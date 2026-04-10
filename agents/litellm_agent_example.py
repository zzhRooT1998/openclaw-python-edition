from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable

from system_prompt import SYSTEM_PROMPT


@dataclass
class ModelDef:
    """
    A lightweight Python equivalent of the TypeScript `Model` shape.
    This keeps provider/model metadata together instead of scattering it
    across function calls.
    """

    id: str
    provider: str = "openai"
    api_base: str | None = None
    api_key: str | None = None
    max_tokens: int = 8192
    temperature: float | None = None
    reasoning: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def litellm_model_name(self) -> str:
        """
        LiteLLM commonly uses `<provider>/<model>` to disambiguate calls.
        For OpenAI-compatible endpoints like DashScope, `openai/<model>` is
        the simplest mapping.
        """
        return f"{self.provider}/{self.id}"


@dataclass
class AgentConfig:
    provider: str = "openai"
    model: str = "qwen-plus-0112"
    api_key: str | None = None
    base_url: str | None = None
    system_prompt: str = SYSTEM_PROMPT
    tools: list[dict[str, Any]] = field(default_factory=list)
    temperature: float | None = None
    reasoning: str | None = None
    max_turns: int = 20


@dataclass
class RunResult:
    run_id: str
    text: str
    turns: int
    tool_calls: int
    skill_triggered: str | None = None
    memories_used: int = 0


class LiteLLMAgent:
    """
    A minimal Python agent skeleton that mirrors the structure of
    `src/agent.ts` without bringing over every subsystem yet.

    Included:
    - AgentConfig -> constructor config
    - ModelDef -> provider/model/api_base/api_key bundle
    - build_system_prompt()
    - run() -> one chat-completion roundtrip
    - optional streaming aggregation

    Not included yet:
    - session persistence
    - tool execution loop
    - context compaction
    - memory / skills / heartbeat
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.model_def = ModelDef(
            id=config.model,
            provider=config.provider,
            api_base=config.base_url,
            api_key=config.api_key or self._get_env_api_key(config.provider),
            max_tokens=8192,
            temperature=config.temperature,
            reasoning=config.reasoning,
        )

    @staticmethod
    def _get_env_api_key(provider: str) -> str | None:
        provider_to_env = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GEMINI_API_KEY",
        }
        env_name = provider_to_env.get(provider, "OPENAI_API_KEY")
        return os.getenv(env_name)

    def build_system_prompt(self) -> str:
        return self.config.system_prompt

    def _build_messages(
        self,
        user_message: str,
        history: Iterable[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.build_system_prompt()}
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return messages

    def run(
        self,
        user_message: str,
        history: Iterable[dict[str, Any]] | None = None,
        stream: bool = True,
        print_stream: bool = False,
    ) -> RunResult:
        try:
            from litellm import completion
        except ImportError as exc:
            raise RuntimeError(
                "litellm is not installed. Run `pip install litellm` first."
            ) from exc

        if not self.model_def.api_key:
            raise ValueError(
                f"Missing API key for provider={self.model_def.provider}. "
                "Pass `api_key=` in AgentConfig or export the provider env var."
            )

        messages = self._build_messages(user_message=user_message, history=history)
        request_kwargs: dict[str, Any] = {
            "model": self.model_def.litellm_model_name,
            "messages": messages,
            "stream": stream,
            "api_key": self.model_def.api_key,
        }

        if self.model_def.api_base:
            request_kwargs["api_base"] = self.model_def.api_base
        if self.model_def.temperature is not None:
            request_kwargs["temperature"] = self.model_def.temperature
        if self.model_def.max_tokens:
            request_kwargs["max_tokens"] = self.model_def.max_tokens
        if self.config.tools:
            request_kwargs["tools"] = self.config.tools

        response = completion(**request_kwargs)
        run_id = str(uuid.uuid4())

        if not stream:
            choice = response.choices[0].message
            content = choice.content or ""
            tool_calls = len(choice.tool_calls or [])
            return RunResult(
                run_id=run_id,
                text=content,
                turns=1,
                tool_calls=tool_calls,
            )

        chunks: list[str] = []
        tool_calls = 0

        for chunk in response:
            delta = chunk.choices[0].delta
            content_part = getattr(delta, "content", None)
            if content_part:
                chunks.append(content_part)
                if print_stream:
                    print(content_part, end="", flush=True)

            delta_tool_calls = getattr(delta, "tool_calls", None)
            if delta_tool_calls:
                tool_calls = max(tool_calls, len(delta_tool_calls))

        if print_stream:
            print()

        return RunResult(
            run_id=run_id,
            text="".join(chunks),
            turns=1,
            tool_calls=tool_calls,
        )


def main() -> None:
    """
    Minimal runnable demo for DashScope/Qwen through LiteLLM.

    Required environment variables:
    - OPENAI_API_KEY

    Optional:
    - OPENCLAW_PY_MODEL
    - OPENCLAW_PY_BASE_URL
    """
    agent = LiteLLMAgent(
        AgentConfig(
            provider="openai",
            model=os.getenv("OPENCLAW_PY_MODEL", "qwen-plus-0112"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv(
                "OPENCLAW_PY_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            reasoning=None,
        )
    )

    result = agent.run(
        user_message="用一句话介绍你自己。",
        stream=True,
        print_stream=True,
    )
    print(f"\nrun_id={result.run_id}")
    print(f"turns={result.turns}, tool_calls={result.tool_calls}")


if __name__ == "__main__":
    main()
