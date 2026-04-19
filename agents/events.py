from typing import TypedDict, Union, Literal, Optional, Any, List

from agents.session_tool_result_guard import Message


class AgentStartEvent(TypedDict):
    type: Literal["agent_start"]
    runId: str
    sessionKey: str
    agentId: str
    model: str


class AgentEndEvent(TypedDict):
    type: Literal["agent_end"]
    runId: str
    messages: List["Message"]


class AgentErrorEvent(TypedDict):
    type: Literal["agent_error"]
    runId: str
    error: str


class TurnStartEvent(TypedDict):
    type: Literal["turn_start"]
    turn: int


class TurnEndEvent(TypedDict):
    type: Literal["turn_end"]
    turn: int


class MessageStartEvent(TypedDict):
    type: Literal["message_start"]
    message: "Message"


class MessageDeltaEvent(TypedDict):
    type: Literal["message_delta"]
    delta: str


class MessageEndEvent(TypedDict):
    type: Literal["message_end"]
    message: "Message"
    text: str


class ThinkingDeltaEvent(TypedDict):
    type: Literal["thinking_delta"]
    delta: str


class ToolExecutionStartEvent(TypedDict):
    type: Literal["tool_execution_start"]
    toolCallId: str
    toolName: str
    args: Any


class ToolExecutionEndEvent(TypedDict):
    type: Literal["tool_execution_end"]
    toolCallId: str
    toolName: str
    result: str
    isError: bool


class ToolSkippedEvent(TypedDict):
    type: Literal["tool_skipped"]
    toolCallId: str
    toolName: str


class ToolApprovalRequestEvent(TypedDict):
    type: Literal["tool_approval_request"]
    toolCallId: str
    toolName: str
    args: Any


class ToolApprovalResolvedEvent(TypedDict):
    type: Literal["tool_approval_resolved"]
    toolCallId: str
    toolName: str
    decision: Literal["allow-once", "allow-always", "deny"]


class SteeringEvent(TypedDict):
    type: Literal["steering"]
    pendingCount: int


class CompactionEvent(TypedDict):
    type: Literal["compaction"]
    summaryChars: int
    droppedMessages: int


class ContextOverflowCompactEvent(TypedDict):
    type: Literal["context_overflow_compact"]
    error: str


class RetryEvent(TypedDict):
    type: Literal["retry"]
    attempt: int
    delay: float
    error: str


class SubagentSummaryEvent(TypedDict, total=False):
    type: Literal["subagent_summary"]
    childSessionKey: str
    label: str
    task: str
    summary: str


class SubagentErrorEvent(TypedDict, total=False):
    type: Literal["subagent_error"]
    childSessionKey: str
    label: str
    task: str
    error: str

MiniAgentEvent = Union[
    AgentStartEvent,
    AgentEndEvent,
    AgentErrorEvent,
    TurnStartEvent,
    TurnEndEvent,
    MessageStartEvent,
    MessageDeltaEvent,
    MessageEndEvent,
    ThinkingDeltaEvent,
    ToolExecutionStartEvent,
    ToolExecutionEndEvent,
    ToolSkippedEvent,
    ToolApprovalRequestEvent,
    ToolApprovalResolvedEvent,
    SteeringEvent,
    CompactionEvent,
    ContextOverflowCompactEvent,
    RetryEvent,
    SubagentSummaryEvent,
    SubagentErrorEvent,
]


from dataclasses import dataclass, field


@dataclass
class MiniAgentResult:
    finalText: str = ""
    turns: int = 0
    totalToolCalls: int = 0
    messages: List["Message"] = field(default_factory=list)