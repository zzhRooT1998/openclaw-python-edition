from __future__ import annotations

import asyncio
import inspect
import json
import random
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, TypedDict

from litellm import acompletion

from agents.builtin_tools import Tool
from agents.event_stream import EventStream
from agents.events import MiniAgentEvent, MiniAgentResult
from agents.pruning import prune_context_message
from agents.session import ContentBlock, Message


class ToolContext:
    pass


class PrepareParam(TypedDict):
    messages: list[Message]
    session_key: str
    run_id: str


class PrepareResult(TypedDict, total=False):
    summary: str
    summary_message: Message
    summaryMessage: Message


class ToolApprovalResult(TypedDict):
    approved: bool
    decision: str


class AbortSignal:
    aborted: bool


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


class AgentLoopParam:
    run_id: str
    session_key: str
    agent_id: str
    current_messages: list[Message]
    compaction_summary: Message | None
    system_prompt: str
    tools_for_run: list[Tool]
    tool_context: ToolContext
    model_def: Any
    stream_fn: Any
    api_key: str | None
    temperature: float | None
    reasoning: Any
    max_turns: int
    context_tokens: int

    get_steering_messages: Callable[[], Awaitable[list[Message]]]
    get_follow_up_messages: Optional[Callable[[], Awaitable[list[Message]]]]
    append_message: Callable[[str, Message], Awaitable[None]]
    prepare_compaction: Callable[[PrepareParam], Awaitable[PrepareResult | Any]]
    check_tool_approval: Optional[Callable[[ToolCall], Awaitable[Optional[ToolApprovalResult | Any]]]]
    abort_signal: AbortSignal


@dataclass
class LLMCallResult:
    assistant_content: list[ContentBlock]
    tool_calls: list[ToolCall]
    turn_text_parts: list[str]


def now_ms() -> int:
    return int(time.time() * 1000)


def create_min_agent_stream() -> EventStream[MiniAgentEvent, MiniAgentResult]:
    return EventStream(
        lambda _event: False,
        lambda _event: MiniAgentResult(
            finalText="", turns=0, totalToolCalls=0, messages=[]
        ),
    )


def get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def describe_error(err: BaseException | Any) -> str:
    if isinstance(err, BaseException):
        return str(err)
    return str(err)


def is_rate_limit_error(error_text: str) -> bool:
    text = error_text.lower()
    return "rate limit" in text or "too many requests" in text or "429" in text


def is_context_overflow_error(error_text: str) -> bool:
    text = error_text.lower()
    return (
        "context length" in text
        or "context window" in text
        or "maximum context" in text
        or "token limit" in text
        or "too many tokens" in text
    )


def make_text_block(text: str) -> ContentBlock:
    return ContentBlock(
        type="text",
        text=text,
        id=None,
        name=None,
        input=None,
        tool_use_id=None,
        content=None,
    )


def make_tool_use_block(call: ToolCall) -> ContentBlock:
    return ContentBlock(
        type="tool_use",
        text=None,
        id=call.id,
        name=call.name,
        input=call.input,
        tool_use_id=None,
        content=None,
    )


def make_tool_result_block(call_id: str, name: str, content: str) -> ContentBlock:
    return ContentBlock(
        type="tool_result",
        text=None,
        id=None,
        name=name,
        input=None,
        tool_use_id=call_id,
        content=content,
    )


def skip_tool_call(call: ToolCall) -> ContentBlock:
    return make_tool_result_block(
        call.id,
        call.name,
        "Skipped due to queued user message.",
    )


def convert_messages_litellm(messages: list[Message]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    for message in messages:
        role = get_value(message, "role")
        content = get_value(message, "content")

        if role == "user":
            if isinstance(content, str):
                result.append({"role": "user", "content": content})
                continue

            text_parts: list[dict[str, str]] = []
            for block in content or []:
                block_type = get_value(block, "type")
                if block_type == "text" and get_value(block, "text"):
                    text_parts.append({"type": "text", "text": get_value(block, "text")})
                elif block_type == "tool_result":
                    tool_content = get_value(block, "content") or ""
                    if not isinstance(tool_content, str):
                        tool_content = json.dumps(tool_content, ensure_ascii=False)
                    result.append(
                        {
                            "role": "tool",
                            "tool_call_id": get_value(block, "tool_use_id") or "",
                            "name": get_value(block, "name") or "",
                            "content": tool_content,
                        }
                    )

            if text_parts:
                result.append({"role": "user", "content": text_parts})
            continue

        if role == "assistant":
            if isinstance(content, str):
                result.append({"role": "assistant", "content": content})
                continue

            text_parts: list[dict[str, str]] = []
            tool_calls: list[dict[str, Any]] = []
            for block in content or []:
                block_type = get_value(block, "type")
                if block_type == "text" and get_value(block, "text"):
                    text_parts.append({"type": "text", "text": get_value(block, "text")})
                elif block_type == "tool_use":
                    tool_calls.append(
                        {
                            "id": get_value(block, "id") or "",
                            "type": "function",
                            "function": {
                                "name": get_value(block, "name") or "",
                                "arguments": json.dumps(
                                    get_value(block, "input") or {},
                                    ensure_ascii=False,
                                ),
                            },
                        }
                    )

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": text_parts if text_parts else "",
            }
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            result.append(assistant_msg)

    return result


def convert_tools_litellm(tools: list[Tool]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    for tool in tools:
        name = get_value(tool, "name")
        if not name:
            continue

        description = get_value(tool, "description", "") or ""
        input_schema = (
            get_value(tool, "input_schema")
            or get_value(tool, "inputSchema")
            or get_value(tool, "parameters")
            or {"type": "object", "properties": {}}
        )

        result.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": input_schema,
                },
            }
        )

    return result


def resolve_model_name(model_def: Any) -> str:
    litellm_model = get_value(model_def, "litellm_model")
    if litellm_model:
        return litellm_model

    model_id = (
        get_value(model_def, "model_id")
        or get_value(model_def, "id")
        or get_value(model_def, "model")
    )
    if not model_id:
        raise ValueError("model_def must provide model_id/id/model")

    if "/" in model_id:
        return model_id

    provider = get_value(model_def, "provider", "openai") or "openai"
    return f"{provider}/{model_id}"


def build_litellm_request(
    *,
    system_prompt: str,
    messages_for_model: list[Message],
    tools_for_run: list[Tool],
    model_def: Any,
    api_key: str | None,
    temperature: float | None,
) -> dict[str, Any]:
    request_kwargs: dict[str, Any] = {
        "model": resolve_model_name(model_def),
        "messages": [
            {"role": "system", "content": system_prompt},
            *convert_messages_litellm(messages_for_model),
        ],
        "stream": True,
    }

    if api_key:
        request_kwargs["api_key"] = api_key

    base_url = get_value(model_def, "base_url") or get_value(model_def, "api_base")
    if base_url:
        request_kwargs["api_base"] = base_url

    max_tokens = get_value(model_def, "max_tokens") or get_value(model_def, "maxTokens")
    if max_tokens:
        request_kwargs["max_tokens"] = max_tokens

    if temperature is not None:
        request_kwargs["temperature"] = temperature

    request_tools = convert_tools_litellm(tools_for_run)
    if request_tools:
        request_kwargs["tools"] = request_tools
        request_kwargs["tool_choice"] = "auto"

    return request_kwargs


def safe_prune_context_messages(messages: list[Message], context_tokens: int) -> list[Message]:
    try:
        prune_result = prune_context_message(
            messages=messages,
            context_window_tokens=context_tokens,
            settings=None,
        )
        return prune_result.messages
    except Exception:
        # The Python port's pruning module is still evolving. Falling back here
        # keeps the agent loop usable while pruning internals are completed.
        return messages


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def retry_async(
    fn: Callable[[], Awaitable[LLMCallResult]],
    *,
    stream: EventStream[MiniAgentEvent, MiniAgentResult],
    abort_signal: AbortSignal,
    attempts: int = 3,
    min_delay_ms: int = 300,
    max_delay_ms: int = 30_000,
    jitter: float = 0.1,
) -> LLMCallResult:
    for attempt in range(1, attempts + 1):
        try:
            return await fn()
        except Exception as err:
            if attempt >= attempts or get_value(abort_signal, "aborted", False):
                raise
            if not is_rate_limit_error(describe_error(err)):
                raise

            delay = min(max_delay_ms, min_delay_ms * (2 ** (attempt - 1)))
            if jitter:
                delay = int(delay * (1 + random.uniform(-jitter, jitter)))

            await stream.push(
                {
                    "type": "retry",
                    "attempt": attempt + 1,
                    "delay": delay,
                    "error": describe_error(err),
                }
            )
            await asyncio.sleep(delay / 1000)

    raise RuntimeError("retry_async exhausted unexpectedly")


async def call_litellm_once(
    request_kwargs: dict[str, Any],
    stream: EventStream[MiniAgentEvent, MiniAgentResult],
    abort_signal: AbortSignal,
) -> LLMCallResult:
    assistant_content: list[ContentBlock] = []
    tool_calls: list[ToolCall] = []
    turn_text_parts: list[str] = []

    response = await acompletion(**request_kwargs)
    text_chunks: list[str] = []
    tool_call_chunks: dict[int, dict[str, str]] = {}

    async for chunk in response:
        if get_value(abort_signal, "aborted", False):
            break

        choice = get_value(chunk, "choices", [None])[0]
        delta = get_value(choice, "delta", {})

        text_delta = get_value(delta, "content")
        if text_delta:
            text_chunks.append(text_delta)
            await stream.push({"type": "message_delta", "delta": text_delta})

        thinking_delta = (
            get_value(delta, "reasoning_content")
            or get_value(delta, "reasoning")
            or get_value(delta, "thinking")
        )
        if thinking_delta:
            await stream.push({"type": "thinking_delta", "delta": thinking_delta})

        delta_tool_calls = get_value(delta, "tool_calls") or []
        for delta_tool_call in delta_tool_calls:
            index = get_value(delta_tool_call, "index", 0)
            current = tool_call_chunks.setdefault(
                index,
                {"id": "", "name": "", "arguments": ""},
            )

            call_id = get_value(delta_tool_call, "id")
            if call_id:
                current["id"] = call_id

            function_delta = get_value(delta_tool_call, "function", {})
            name_part = get_value(function_delta, "name")
            if name_part:
                current["name"] += name_part

            args_part = get_value(function_delta, "arguments")
            if args_part:
                current["arguments"] += args_part

    turn_text = "".join(text_chunks)
    if turn_text:
        turn_text_parts.append(turn_text)
        assistant_content.append(make_text_block(turn_text))

    for raw_call in tool_call_chunks.values():
        raw_args = raw_call["arguments"] or "{}"
        try:
            parsed_args = json.loads(raw_args)
        except json.JSONDecodeError:
            parsed_args = {"_raw": raw_args}

        call = ToolCall(
            id=raw_call["id"],
            name=raw_call["name"],
            input=parsed_args,
        )
        assistant_content.append(make_tool_use_block(call))
        tool_calls.append(call)

    return LLMCallResult(
        assistant_content=assistant_content,
        tool_calls=tool_calls,
        turn_text_parts=turn_text_parts,
    )


def find_tool(tools: list[Tool], name: str) -> Tool | None:
    for tool in tools:
        if get_value(tool, "name") == name:
            return tool
    return None


async def execute_tool(tool: Tool | None, call: ToolCall, tool_ctx: ToolContext) -> tuple[str, bool]:
    if tool is None:
        return f"Unknown tool: {call.name}", True

    executor = get_value(tool, "execute")
    if not callable(executor):
        return f"Tool is not executable: {call.name}", True

    try:
        try:
            result = executor(call.input, tool_ctx)
        except TypeError:
            result = executor(call.input)
        result = await maybe_await(result)
        if not isinstance(result, str):
            result = json.dumps(result, ensure_ascii=False)
        return result, False
    except Exception as err:
        return f"Execution error: {describe_error(err)}", True


def get_summary_message(compaction_result: Any) -> Message | None:
    if not compaction_result:
        return None
    summary = get_value(compaction_result, "summary")
    summary_message = (
        get_value(compaction_result, "summary_message")
        or get_value(compaction_result, "summaryMessage")
    )
    if summary and summary_message:
        return summary_message
    return None


async def maybe_check_tool_approval(params: AgentLoopParam, call: ToolCall) -> Any | None:
    checker = get_value(params, "check_tool_approval")
    if checker is None:
        return None
    return await checker(call)


def run_agent_loop(params: AgentLoopParam) -> EventStream[MiniAgentEvent, MiniAgentResult]:
    stream = create_min_agent_stream()

    async def _runner() -> None:
        run_id = params.run_id
        session_key = params.session_key
        current_messages = params.current_messages
        compaction_summary = params.compaction_summary
        system_prompt = params.system_prompt
        tools_for_run = params.tools_for_run
        tool_ctx = params.tool_context
        model_def = params.model_def
        api_key = params.api_key
        temperature = params.temperature
        max_turns = params.max_turns
        context_tokens = params.context_tokens
        get_steering_messages = params.get_steering_messages
        get_follow_up_messages = get_value(params, "get_follow_up_messages")
        append_message = params.append_message
        prepare_compaction = params.prepare_compaction
        abort_signal = params.abort_signal

        turns = 0
        total_tool_calls = 0
        final_text = ""
        overflow_compaction_attempted = False

        try:
            pending_messages = await get_steering_messages()
            break_outer = False

            while not break_outer:
                has_more_tool_calls = True

                while has_more_tool_calls or len(pending_messages) > 0:
                    if turns >= max_turns:
                        break_outer = True
                        break
                    if get_value(abort_signal, "aborted", False):
                        break_outer = True
                        break

                    turns += 1
                    await stream.push({"type": "turn_start", "turn": turns})

                    if pending_messages:
                        for message in pending_messages:
                            await append_message(session_key, message)
                            current_messages.append(message)
                        pending_messages = []

                    messages_for_model = safe_prune_context_messages(
                        current_messages,
                        context_tokens,
                    )
                    if compaction_summary is not None:
                        messages_for_model = [compaction_summary, *messages_for_model]

                    request_kwargs = build_litellm_request(
                        system_prompt=system_prompt,
                        messages_for_model=messages_for_model,
                        tools_for_run=tools_for_run,
                        model_def=model_def,
                        api_key=api_key,
                        temperature=temperature,
                    )

                    try:
                        llm_result = await retry_async(
                            lambda: call_litellm_once(
                                request_kwargs,
                                stream,
                                abort_signal,
                            ),
                            stream=stream,
                            abort_signal=abort_signal,
                        )
                    except Exception as llm_error:
                        error_text = describe_error(llm_error)
                        if (
                            is_context_overflow_error(error_text)
                            and not overflow_compaction_attempted
                        ):
                            overflow_compaction_attempted = True
                            await stream.push(
                                {
                                    "type": "context_overflow_compact",
                                    "error": error_text,
                                }
                            )
                            overflow_prep = await prepare_compaction(
                                {
                                    "messages": current_messages,
                                    "session_key": session_key,
                                    "run_id": run_id,
                                }
                            )
                            next_summary = get_summary_message(overflow_prep)
                            if next_summary is not None:
                                compaction_summary = next_summary
                                turns -= 1
                                continue
                        raise

                    assistant_msg = Message(
                        role="assistant",
                        content=llm_result.assistant_content,
                        timestamp=now_ms(),
                    )
                    await append_message(session_key, assistant_msg)
                    current_messages.append(assistant_msg)

                    turn_text = "".join(llm_result.turn_text_parts)
                    if turn_text:
                        final_text = turn_text
                        await stream.push(
                            {
                                "type": "message_end",
                                "message": assistant_msg,
                                "text": turn_text,
                            }
                        )

                    tool_calls = llm_result.tool_calls
                    has_more_tool_calls = len(tool_calls) > 0

                    if not has_more_tool_calls:
                        await stream.push({"type": "turn_end", "turn": turns})
                        pending_messages = await get_steering_messages()
                        continue

                    tool_results: list[ContentBlock] = []
                    steering_messages: list[Message] | None = None

                    for index, call in enumerate(tool_calls):
                        tool = find_tool(tools_for_run, call.name)
                        await stream.push(
                            {
                                "type": "tool_execution_start",
                                "toolCallId": call.id,
                                "toolName": call.name,
                                "args": call.input,
                            }
                        )

                        approval = await maybe_check_tool_approval(params, call)
                        if approval is not None:
                            decision = get_value(approval, "decision", "deny")
                            approved = bool(get_value(approval, "approved", False))
                            await stream.push(
                                {
                                    "type": "tool_approval_request",
                                    "toolCallId": call.id,
                                    "toolName": call.name,
                                    "args": call.input,
                                }
                            )
                            await stream.push(
                                {
                                    "type": "tool_approval_resolved",
                                    "toolCallId": call.id,
                                    "toolName": call.name,
                                    "decision": decision,
                                }
                            )
                            if not approved:
                                result = "Tool execution denied by user."
                                total_tool_calls += 1
                                await stream.push(
                                    {
                                        "type": "tool_execution_end",
                                        "toolCallId": call.id,
                                        "toolName": call.name,
                                        "result": result,
                                        "isError": True,
                                    }
                                )
                                tool_results.append(
                                    make_tool_result_block(call.id, call.name, result)
                                )
                                steering = await get_steering_messages()
                                if steering:
                                    steering_messages = steering
                                    for skipped in tool_calls[index + 1 :]:
                                        await stream.push(
                                            {
                                                "type": "tool_skipped",
                                                "toolCallId": skipped.id,
                                                "toolName": skipped.name,
                                            }
                                        )
                                        tool_results.append(skip_tool_call(skipped))
                                    await stream.push(
                                        {
                                            "type": "steering",
                                            "pendingCount": len(steering),
                                        }
                                    )
                                    break
                                continue

                        result, is_error = await execute_tool(tool, call, tool_ctx)
                        total_tool_calls += 1
                        await stream.push(
                            {
                                "type": "tool_execution_end",
                                "toolCallId": call.id,
                                "toolName": call.name,
                                "result": result[:500] + "..." if len(result) > 500 else result,
                                "isError": is_error,
                            }
                        )
                        tool_results.append(
                            make_tool_result_block(call.id, call.name, result)
                        )

                        steering = await get_steering_messages()
                        if steering:
                            steering_messages = steering
                            for skipped in tool_calls[index + 1 :]:
                                await stream.push(
                                    {
                                        "type": "tool_skipped",
                                        "toolCallId": skipped.id,
                                        "toolName": skipped.name,
                                    }
                                )
                                tool_results.append(skip_tool_call(skipped))
                            await stream.push(
                                {
                                    "type": "steering",
                                    "pendingCount": len(steering),
                                }
                            )
                            break

                    result_msg = Message(
                        role="user",
                        content=tool_results,
                        timestamp=now_ms(),
                    )
                    await append_message(session_key, result_msg)
                    current_messages.append(result_msg)

                    await stream.push({"type": "turn_end", "turn": turns})
                    if steering_messages:
                        pending_messages = steering_messages
                    else:
                        pending_messages = await get_steering_messages()

                if break_outer:
                    break

                if get_follow_up_messages is not None:
                    follow_up = await get_follow_up_messages()
                    if follow_up:
                        pending_messages = follow_up
                        continue

                break

            await stream.push(
                {
                    "type": "agent_end",
                    "runId": run_id,
                    "messages": current_messages,
                }
            )
            await stream.end(
                MiniAgentResult(
                    finalText=final_text,
                    turns=turns,
                    totalToolCalls=total_tool_calls,
                    messages=current_messages,
                )
            )
        except Exception as err:
            await stream.push(
                {
                    "type": "agent_error",
                    "runId": run_id,
                    "error": describe_error(err),
                }
            )
            await stream.end(
                MiniAgentResult(
                    finalText=final_text,
                    turns=turns,
                    totalToolCalls=total_tool_calls,
                    messages=current_messages,
                )
            )

    asyncio.create_task(_runner())
    return stream