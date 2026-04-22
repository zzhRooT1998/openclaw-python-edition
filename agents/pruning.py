import json
import logging
import re
from dataclasses import dataclass, replace
from platform import android_ver
from typing import Optional, Callable, List, Any

from agents.session import Message, ContentBlock


@dataclass()
class SoftTrim:
    max_chars: int
    head_chars: int
    tail_chars: int
@dataclass()
class HardClear:
    enabled: bool
    placeholder: str

@dataclass()
class ContextPruningToolMatch:
    allow: Optional[list[str]]
    deny: Optional[list[str]]
@dataclass
class ContextPruningSettings:
    max_history_share: float
    keep_last_assistants: int
    soft_trim_ratio: float

    hard_clear_ratio: float
    min_prunable_tool_chars: int

    soft_trim: SoftTrim

    hard_clear: HardClear
    tools:ContextPruningToolMatch


@dataclass
class PruneResult:
    messages: list[Message]
    dropped_messages: list[Message]
    trimmed_tool_results: int
    hard_cleared_tool_results: int
    total_chars: int
    kept_chars: int
    dropped_chars: int
    budget_chars: int

DEFAULT_CONTEXT_PRUNING_SETTINGS = ContextPruningSettings(
    max_history_share=0.5,
    keep_last_assistants=3,
    soft_trim_ratio=0.3,
    hard_clear_ratio=0.5,
    min_prunable_tool_chars=50_000,
    soft_trim=SoftTrim(
        max_chars=4000,
        head_chars=1500,
        tail_chars=1500,
    ),
    hard_clear=HardClear(
        enabled=True,
        placeholder="[Old tool result content cleared]",
    ),
    tools={},
)
def clamp_share(value: Optional[float], default: float) -> float:
    if value is None:
        return default
    return max(0.0, min(1.0, float(value)))

def clamp_positive_int(value: Optional[int], default: int) -> int:
    if value is None:
        return default
    return max(1, int(value))

def resolve_pruning_settings(
    raw: Optional[ContextPruningSettings]
) -> ContextPruningSettings:

    if raw is None:
        return DEFAULT_CONTEXT_PRUNING_SETTINGS

    d = DEFAULT_CONTEXT_PRUNING_SETTINGS

    return ContextPruningSettings(
        max_history_share=clamp_share(
            getattr(raw, "max_history_share", None),
            d.max_history_share,
        ),
        keep_last_assistants=clamp_positive_int(
            getattr(raw, "keep_last_assistants", None),
            d.keep_last_assistants,
        ),
        soft_trim_ratio=clamp_share(
            getattr(raw, "soft_trim_ratio", None),
            d.soft_trim_ratio,
        ),
        hard_clear_ratio=clamp_share(
            getattr(raw, "hard_clear_ratio", None),
            d.hard_clear_ratio,
        ),
        min_prunable_tool_chars=clamp_positive_int(
            getattr(raw, "min_prunable_tool_chars", None),
            d.min_prunable_tool_chars,
        ),
        soft_trim=SoftTrim(
            max_chars=clamp_positive_int(
                getattr(raw.soft_trim, "max_chars", None) if raw.soft_trim else None,
                d.soft_trim.max_chars,
            ),
            head_chars=clamp_positive_int(
                getattr(raw.soft_trim, "head_chars", None) if raw.soft_trim else None,
                d.soft_trim.head_chars,
            ),
            tail_chars=clamp_positive_int(
                getattr(raw.soft_trim, "tail_chars", None) if raw.soft_trim else None,
                d.soft_trim.tail_chars,
            ),
        ),
        hard_clear=HardClear(
            enabled=(
                raw.hard_clear.enabled
                if raw.hard_clear and raw.hard_clear.enabled is not None
                else d.hard_clear.enabled
            ),
            placeholder=(
                raw.hard_clear.placeholder
                if raw.hard_clear and raw.hard_clear.placeholder is not None
                else d.hard_clear.placeholder
            ),
        ),
        tools=raw.tools if raw.tools is not None else d.tools,
    )


def match_glob(value: str, pattern: str) -> bool:
    if pattern == "*":
        return True

    if "*" not in pattern:
        return value == pattern

    # 转换成正则
    def replace(ch: str) -> str:
        if ch == "*":
            return ".*"
        return "\\" + ch

    escaped = "".join(
        replace(ch) if ch in r".*+?^${}()|[]\\" else ch
        for ch in pattern
    )

    regex = f"^{escaped}$"
    return re.match(regex, value) is not None


def make_tool_prunable_predicate(match: Optional[ContextPruningToolMatch]) -> Callable[[str], bool]:
    if match is None:
        return lambda _: True
    deny = match.deny if match.deny is not None else []
    allow = match.allow if match.allow is not None else []

    def predicate(tool_name: str) -> bool:
        normalized = tool_name.strip().lower()
        if any(match_glob(normalized, pattern) for pattern in deny):
            return False
        if len(allow) == 0:
            return True

        return any(match_glob(normalized, pattern) for pattern in allow)
    return predicate

CHARS_PER_TOKEN_ESTIMATE = 4

def estimate_messages_chars(messages: list[Message]) -> int:
    return sum(estimate_message_chars(message) for message in messages)

def estimate_block_chars(block: ContentBlock) -> int:
    if block.type == "text":
        return len(block.text) if block.text else 0
    if block.type == 'tool_use':
        base = len(block.name) if block.name else 0
        try:
            input_str = json.dumps(block.input) if block.input else ""
            return base + len(input_str) + 16
        except Exception:
            logging.exception("Failed to serialize block.input")
            return base + 128

    if block.type == "tool_result":
        return len(block.content) if block.content else 0
    return 0
def estimate_message_chars(message: Message) -> int:
    if isinstance(message.content, str):
        return len(message.content)
    total = 0
    for block in message.content:
        total += estimate_block_chars(block)
    return total

#三层递进上下文修剪
#soft trim -> hard clear -> message drop
def is_tool_result_protected(block):
    #目前block只支持text、tool_result、tool_use，
    # openclaw还支持image，此处预留扩展
    return False


def soft_trim_tool_result_block(block: ContentBlock, soft_trim: SoftTrim, is_prunable:Callable[[str], bool]) -> dict[str, Any]:
    if not block.type == "tool_result":
     return {"block":block, "trimmed": False}

    if is_tool_result_protected(block):
        return {"block":block, "trimmed": False}

    if block.name and not is_prunable(block.name):
        return {"block":block, "trimmed": False}

    raw = block.content if isinstance(block.content, str) else ""
    raw_len = len(raw)
    if raw_len <= soft_trim.max_chars:
        return {"block":block, "trimmed": False}

    head_chars = max(0, soft_trim.head_chars)
    tail_chars = max(0, soft_trim.tail_chars)
    if raw_len < soft_trim.max_chars:
        return {"block":block, "trimmed": False}

    head = raw[:head_chars]
    tail = raw[-tail_chars:]
    trimmed_context = (
        f"{head}\n...\n{tail}\n\n"
        f"[Tool result trimmed: kept first {head_chars} chars"
        f"and last {tail_chars} chars of {raw_len} chars.]"
    )
    return {"block":replace(block, content=trimmed_context), "trimmed":True}



@dataclass()
class ApplySoftTrimResult:
    messages: List[Message]
    trimmed_tool_results: int


def clone_message(message, next_blocks):
    pass


def apply_soft_trim(messages: List[Message], settings: ContextPruningSettings, is_prunable:Callable[[str], bool]) -> ApplySoftTrimResult:
    trimmed_tool_results = 0
    output: List[Message] = []

    for message in messages:
        if isinstance(message.content, str):
            output.append(message)
            continue

        did_change = False
        next_blocks: List[ContentBlock] = []

        for block in message.content:
            result = soft_trim_tool_result_block(block, settings.soft_trim, is_prunable)
            if result["trimmed"]:
                trimmed_tool_results += 1
                did_change = True
            next_blocks.append(result["block"])

        output.append(clone_message(message, next_blocks) if did_change else message)

    return ApplySoftTrimResult(
        messages=output,
        trimmed_tool_results=trimmed_tool_results
    )


def count_prunable_tool_chars(messages: List[Message], is_prunable: Callable[[str], bool]) -> int:
    total = 0
    for message in messages:
        if isinstance(message.content, str):
            continue
        for block in message.content:
            if block.type != "tool_result":
                continue
            if is_tool_result_protected(block):
                continue
            if block.name and not is_prunable(block.name):
                continue
            text = block.content if isinstance(block.content, str) else ""
            total += len(text)
    return total


def apply_hard_clear(messages: List[Message], settings: ContextPruningSettings, is_prunable:Callable[[str], bool], chat_window: int):
    if not settings.hard_clear.enabled:
        return {"messages":messages, "hard_cleared_tool_results": 0}
    total_chars = estimate_messages_chars(messages)

    ratio = total_chars / chat_window
    if ratio < settings.hard_clear_ratio:
        return {"messages":messages, "hard_cleared_tool_results": 0}

    prunable_chars = count_prunable_tool_chars(messages, is_prunable)
    if prunable_chars < settings.min_prunable_tool_chars:
        return {"messages":messages, "hard_cleared_tool_results": 0}

    hard_cleared_tool_results = 0
    output: List[Message] = []

    for message in messages:
        if isinstance(message.content, str):
            output.append(message)
            continue

        did_change = False
        next_blocks: List[ContentBlock] = []

        for block in message.content:
            if (
                    block.type == "tool_result"
                    and not is_tool_result_protected(block)
                    and isinstance(block.content, str)
                    and len(block.content) > 0
            ):
                can_prune = not block.name or is_prunable(block.name)
                if can_prune:
                    current_ratio = total_chars / chat_window
                    if current_ratio < settings.hard_clear_ratio:
                        next_blocks.append(block)
                        continue
                    before_len = len(block.content)
                    cleared_block = replace(block, content=settings.hard_clear.placeholder)
                    next_blocks.append(cleared_block)
                    total_chars -= before_len - len(settings.hard_clear.placeholder)
                    hard_cleared_tool_results += 1
                    did_change = True
                    continue

            next_blocks.append(block)
        output.append(clone_message(message, next_blocks) if did_change else message)
    return {"messages": output, "hard_cleared_tool_results": hard_cleared_tool_results}


def find_assistant_cutoff_index(messages: List[Message], keep_last_assistants: int) -> int | None:
    if keep_last_assistants <= 0:
        return len(messages)
    remaining = keep_last_assistants
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role != "assistant":
            continue
        remaining -= 1
        if remaining == 0:
            return i
    return None


def slice_within_budget(messages: List[Message], budget_chars: int) -> List[Message]:
    kept: List[Message] = []
    used = 0
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        chars = estimate_message_chars(msg)
        if used + chars > budget_chars and len(kept) > 0:
            break
        kept.append(msg)
        used += chars

    kept.reverse()
    return kept


def prune_context_message(messages: list[Message], context_window_tokens: int, settings: Optional[ContextPruningSettings]) -> PruneResult:
    settings = resolve_pruning_settings(settings)
    context_tokens = max(1, int(context_window_tokens))
    chat_window = context_tokens * CHARS_PER_TOKEN_ESTIMATE
    budget_chars = max(1, int(chat_window * settings.max_history_share))
    is_prunable = make_tool_prunable_predicate(settings.tools)

    current = messages
    trimmed_tool_results = 0
    hard_cleared_tool_results = 0

    #layer 1: soft trim - 比例超过 softTrimRatio 时触发
    total_chars = estimate_messages_chars(current)
    #当前字符串 / token总数转换成字符数量得到ratio
    ratio = total_chars / chat_window
    if ratio > settings.soft_trim_ratio:
        trim_result = apply_soft_trim(current, settings, is_prunable)
        current = trim_result.messages
        trimmed_tool_results = trim_result.trimmed_tool_results

    #layer 2: hard clear - soft trim 后仍然超标时触发
    after_soft_trim_chars = estimate_messages_chars(current)
    after_soft_trim_ratio = after_soft_trim_chars / chat_window
    if after_soft_trim_ratio > settings.soft_trim_ratio:
        clear_result = apply_hard_clear(current, settings, is_prunable, chat_window)
        current = clear_result["messages"]
        hard_cleared_tool_results = clear_result["hard_cleared_tool_results"]

    #layer 3: message drop - 超出 history budget 时丢弃消息
    after_clear_chars = estimate_messages_chars(current)
    if after_clear_chars <= budget_chars:
        return PruneResult(
            messages=current,
            dropped_messages=[],
            trimmed_tool_results=trimmed_tool_results,
            hard_cleared_tool_results=hard_cleared_tool_results,
            total_chars=after_clear_chars,
            kept_chars=after_clear_chars,
            dropped_chars=0,
            budget_chars=budget_chars
        )


    cutoff_index = find_assistant_cutoff_index(current, settings.keep_last_assistants)
    protected_index = cutoff_index if cutoff_index else 0
    protected_messages = current[protected_index:]
    protected_chars = estimate_messages_chars(protected_messages)

    kept: List[Message]
    if protected_chars > budget_chars:
        kept = slice_within_budget(current, budget_chars)
    else:
        kept = protected_messages.copy()
        remaining = budget_chars - protected_chars
        for i in range (protected_index - 1, -1, -1):
            msg = current[i]
            msg_chars = estimate_message_chars(msg)
            if msg_chars > remaining:
                break
            kept.insert(0, msg)
            remaining -= msg_chars

    kept_set = set(kept)
    dropped_messages =  [msg for msg in current if msg not in kept_set]
    kept_chars = estimate_messages_chars(kept)
    dropped_chars = max(0, after_clear_chars - kept_chars)


    return PruneResult(
        messages=kept,
        dropped_messages=dropped_messages,
        trimmed_tool_results=trimmed_tool_results,
        hard_cleared_tool_results=hard_cleared_tool_results,
        total_chars=after_clear_chars,
        kept_chars=kept_chars,
        dropped_chars=dropped_chars,
        budget_chars=budget_chars
    )