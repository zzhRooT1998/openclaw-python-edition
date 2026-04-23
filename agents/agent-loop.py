import json
from typing import Awaitable, Callable, TypedDict, Any, Optional, List, Dict

from agents.agent import ModelDef, StreamFunction, ThinkLevel
from agents.builtin_tools import Tool
from agents.event_stream import EventStream
from agents.events import MiniAgentResult, MiniAgentEvent
from agents.pruning import prune_context_message
from agents.session import Message


class ToolContext:
    pass

class PrepareParam(TypedDict):
    messages: list[Message]
    session_key: str
    run_id:str
class PrepareResult(TypedDict, total=False):
    summary: str
    summary_message: str
class ToolCall:
    id: str
    name: str
    input: Any
class ToolApprovalResult(TypedDict):
    approved: bool
    decision: str
class AbortSignal:
    aborted: bool
class AgentLoopParam:
    run_id: str
    session_key: str
    agent_id: str
    current_messages: list[Message]
    compaction_summary: Message | None
    system_prompt: str
    tools_for_run: list[Tool]
    tool_context: ToolContext
    model_def: ModelDef
    stream_fn: StreamFunction
    api_key: str
    temperature: int
    reasoning: ThinkLevel
    max_turns: int
    context_tokens: int

    get_steering_messages: Callable[[], Awaitable[list[Message]]]
    get_follow_up_messages: Callable[[], Awaitable[list[Message]]]
    append_message: Callable[[str, Message], Awaitable[None]]
    prepare_compaction: Callable[[PrepareParam], Awaitable[None]]
    check_tool_approval: Optional[Callable[[ToolCall], Awaitable[Optional[ToolApprovalResult]]]]
    abort_signal: AbortSignal

def create_min_agent_stream() ->EventStream[MiniAgentEvent, MiniAgentResult]:
    return EventStream(
        lambda _event: False,
        lambda _event: MiniAgentResult(
            finalText="", turns=0, totalToolCalls=0, messages=[]
        ),
    )


def convert_messages_litellm(messages: List[Message]) -> List[dict[str, Any]]:
    result: List[dict[str, Any]] = []

    for message in messages:
        if message.role == "user":
            if isinstance(message.content, str):
                result.append(
                    {
                        "role": "user",
                        "content": message.content,
                        "timestamp": message.timestamp,
                    }
                )
                continue

            text_parts: List[Dict[str, str]] = []
            for block in message.content or []:
                block_type = block.type
                if block_type == "text" and block.text:
                    text_parts.append({
                        "type": "text",
                        "text": block.text,
                    })
                elif block_type == "tool_result":
                    tool_content = block.content or ""
                    if not isinstance(tool_content, str):
                        tool_content = json.dumps(tool_content, ensure_ascii=False)
                    result.append({
                        "role": "tool",
                        "tool_call_id": block.tool_use_id,
                        "content": tool_content,
                    })
            if text_parts:
                result.append({
                    "role": "user",
                    "content": text_parts
                })
        elif message.role == "assistant":
            if isinstance(message.content, str):
                result.append({
                    "role": "assistant",
                    "content": message.content,
                })
                continue
            text_parts: List[Dict[str, str]] = []
            tool_calls: List[Dict[str, Any]] = []

            for block in message.content or []:
                block_type = block.type

                if block_type == "text" and block.text:
                    text_parts.append({
                        "type": "text",
                        "text": block.text,
                    })
                elif block_type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input, {}, ensure_ascii=False),
                        }
                    })
            assistant_msg: Dict[str, Any] = {
                "role": "assistant",
            }
            if text_parts:
                assistant_msg["content"] = text_parts
            else:
                assistant_msg["content"] = ""
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            result.append(assistant_msg)
        else:
            if isinstance(message.content, str):
                result.append({
                    "role": "assistant",
                })
    return result


def convert_tools_litellm(tools: List[Tool]) -> list[dict[str, Any]]:
    result: List[dict[str, Any]] = []

    for tool in tools:
        name = getattr(tool, "name", None)
        description = getattr(tool, "description", {""})
        input_schema = (
            getattr(tool, "input_schema", None)
            or getattr(tool, "inputSchema", None)
            or getattr(tool, "parameters", None)
        )
        if not name:
            continue
        result.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": input_schema or {
                    "type": "object",
                    "properties": {}
                }
            }
        })
    return result


def run_agent_loop(params: AgentLoopParam):
    stream = create_min_agent_stream()

    async def _runner():
        run_id = params.run_id
        session_key = params.session_key
        agent_id = params.agent_id
        current_messages = params.current_messages
        compaction_summary = params.compaction_summary
        system_prompt = params.system_prompt
        tools_for_run = params.tools_for_run
        tool_ctx = params.tool_ctx
        model_def = params.model_def
        stream_fn = params.stream_fn
        api_key = params.api_key
        temperature = params.temperature
        reasoning = params.reasoning
        max_turns = params.max_turns
        context_tokens = params.context_tokens
        get_steering_messages = params.get_steering_messages
        get_follow_up_messages = params.get_follow_up_messages
        append_message = params.append_message
        prepare_compaction = params.prepare_compaction
        abort_signal = params.abort_signal

        turns = 0
        total_tool_calls = 0
        final_text = ""
        overflow_compaction_attempted = False


        try:
            pending_messages = await get_steering_messages()
            while True:
                has_more_tool_calls = True

                while (has_more_tool_calls | pending_messages.len > 0):
                    if turns > max_turns:
                        break
                    if abort_signal.aborted:
                        break
                    turns += 1
                    stream.push({"type":"turn_start", "turn": turns})

                    if len(pending_messages) > 0:
                        for message in pending_messages:
                            await append_message(session_key, message)
                            current_messages.append(message)
                        pending_messages = list()

                        prune_result = prune_context_message(messages=current_messages, context_window_tokens=context_tokens)
                        messages_for_model = prune_result.messages
                        if compaction_summary is not None:
                            messages_for_model = [compaction_summary] + messages_for_model

                        #构造LiteLLM message
                        litellm_messages = convert_messages_litellm(messages_for_model)
                        request_messages = [
                            {"role": "system", "content": system_prompt},
                            *litellm_messages
                        ]

                        request_tools = convert_tools_litellm(tools_for_run)

                        request_kwargs = {
                            "model": model_def.model_id,
                            "messages": request_messages,
                            "stream": True,
                            "api_key": api_key,
                            "tools": request_tools,
                            "tool_choice": "auto"
                        }

                        if model_def.base_url:
                            request_kwargs["api_base"] = model_def.base_url
                        if temperature is not None:
                            request_kwargs["temperature"] = temperature

                        

        except Exception as err:
            print(err)

