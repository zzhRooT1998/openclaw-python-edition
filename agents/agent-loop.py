from typing import Awaitable, Callable, TypedDict, Any, Optional

from agents.agent import ModelDef, StreamFunction, ThinkLevel
from agents.builtin_tools import Tool
from agents.event_stream import EventStream
from agents.events import MiniAgentResult, MiniAgentEvent
from agents.pruning import prune_context_message


class Message:
    pass
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
        except Exception as err:
            print(err)

