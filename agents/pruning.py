from dataclasses import dataclass
from typing import Optional

from agents.session import Message


@dataclass()
class SoftTrim:
    max_chars: int
    head_chars: int
    tail_chars: int
@dataclass()
class HardClear:
    enabled: bool
    placeholder: str
@dataclass
class ContextPruningSettings:
    max_history_share: float
    keep_last_assistants: int
    soft_trim_ratio: float

    hard_clear_ratio: float
    min_prunable_tool_chars: int

    soft_trim: SoftTrim

    hard_clear: HardClear
    tools:dict


@dataclass
class PruneResult:
    messages: list[Message]
    droped_messages: list[Message]
    trimmed_tool_results: int
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


def make_tool_prunable_predicate():
    pass