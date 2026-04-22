from dataclasses import dataclass
from typing import Union, Literal, List, Optional, AnyStr, Any


@dataclass
class ContentBlock:
    type: Literal["text", "tool_use", "tool_result"]
    text: Optional[str]
    id: Optional[str]
    name: Optional[str]
    input: Optional[dict[str, Any]]
    tool_use_id: Optional[str]
    content: Optional[str]
@dataclass
class Message:
    content: Union[str, List[ContentBlock]]
    role: Literal["user", "assistant"]
    timestamp: int
    pass