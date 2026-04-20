from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT_ENV_FILE = PROJECT_ROOT / ".env"


def load_env_file_if_present(env_file: Path) -> None:
    if not env_file.is_file():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        env_key = key.strip()
        env_value = value.strip()
        if not env_key:
            continue
        if len(env_value) >= 2 and env_value[0] == env_value[-1] and env_value[0] in {"'", '"'}:
            env_value = env_value[1:-1]
        os.environ.setdefault(env_key, env_value)


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _active_profile(prefix: str) -> str | None:
    profile = os.environ.get(f"{prefix}_PROFILE", "").strip().lower()
    return profile or None


def _profile_env_value(prefix: str, key: str, default: str | None = None) -> str | None:
    profile = _active_profile(prefix)
    if profile:
        profiled_key = f"{prefix}_{profile.upper()}_{key}"
        profiled_value = os.environ.get(profiled_key)
        if profiled_value is not None and profiled_value.strip() != "":
            return profiled_value.strip()
    value = os.environ.get(f"{prefix}_{key}")
    if value is None:
        return default
    stripped = value.strip()
    return stripped if stripped != "" else default


def _profile_env_flag(prefix: str, key: str, default: bool = False) -> bool:
    profile = _active_profile(prefix)
    if profile:
        profiled_key = f"{prefix}_{profile.upper()}_{key}"
        profiled_value = os.environ.get(profiled_key)
        if profiled_value is not None:
            return profiled_value.strip().lower() in {"1", "true", "yes", "on"}
    return env_flag(f"{prefix}_{key}", default=default)


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    prefix: str
    enabled: bool
    configured: bool
    base_url: str
    model: str
    api_key: str
    timeout_seconds: float
    mode: str = "openai_compatible_chat_completion"
    default_extra_body: dict[str, Any] | None = None


class LLMJSONParseError(ValueError):
    def __init__(self, message: str, *, raw_text: str, payload: dict[str, Any]) -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.payload = payload


class LLMProvider(ABC):
    @property
    @abstractmethod
    def prefix(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def provider_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def enabled(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def configured(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def model(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def timeout_seconds(self) -> float:
        raise NotImplementedError

    @property
    def error(self) -> str | None:
        return None

    @abstractmethod
    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float = 0,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def json(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float = 0,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Some providers (DeepSeek, OpenAI) require the word "json" in the
        # prompt when response_format={"type": "json_object"} is used.
        has_json_hint = any("json" in m.get("content", "").lower() for m in messages)
        if not has_json_hint:
            messages = [{**m} for m in messages]
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] = messages[0]["content"].rstrip() + "\nRespond in JSON."
            else:
                messages.insert(0, {"role": "system", "content": "Respond in JSON."})
        payload = self.chat(
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
            extra_body=extra_body,
        )
        raw_text = extract_message_text(payload)
        try:
            return extract_json_object(raw_text)
        except ValueError as exc:
            raise LLMJSONParseError(str(exc), raw_text=raw_text, payload=payload) from exc

    def summary(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "enabled": self.enabled,
            "configured": self.configured,
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
            "error": self.error,
            "active_profile": _active_profile(self.prefix),
        }


@dataclass(frozen=True)
class UnsupportedProviderConfig:
    prefix: str
    provider_name: str
    enabled: bool
    configured: bool
    error: str


class OpenAICompatibleProvider(LLMProvider):
    def __init__(self, config: OpenAICompatibleConfig) -> None:
        self.config = config

    @property
    def prefix(self) -> str:
        return self.config.prefix

    @property
    def provider_name(self) -> str:
        return "openai_compatible"

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def configured(self) -> bool:
        return self.config.configured

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def timeout_seconds(self) -> float:
        return self.config.timeout_seconds

    def summary(self) -> dict[str, Any]:
        summary = super().summary()
        summary.update(
            {
                "base_url": self.config.base_url,
                "mode": self.config.mode,
                "default_extra_body": self.config.default_extra_body,
            }
        )
        return summary

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float = 0,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            raise ValueError(f"{self.prefix}_ENABLED is not enabled")
        if not self.configured:
            raise ValueError(f"{self.prefix} requires BASE_URL and MODEL when enabled")

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
        }
        if self.config.default_extra_body:
            payload.update(self.config.default_extra_body)
        if response_format is not None:
            payload["response_format"] = response_format
        if extra_body:
            payload.update(extra_body)

        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        response = requests.post(
            build_chat_completions_url(self.config.base_url),
            headers=headers,
            json=payload,
            timeout=self.config.timeout_seconds,
        )
        if not response.ok:
            raise requests.HTTPError(
                f"{response.status_code} {response.reason} for url: {response.url}\n{response.text}",
                response=response,
            )
        return response.json()


class UnsupportedLLMProvider(LLMProvider):
    def __init__(self, config: UnsupportedProviderConfig) -> None:
        self.config = config

    @property
    def prefix(self) -> str:
        return self.config.prefix

    @property
    def provider_name(self) -> str:
        return self.config.provider_name

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def configured(self) -> bool:
        return self.config.configured

    @property
    def model(self) -> str:
        return ""

    @property
    def timeout_seconds(self) -> float:
        return 0.0

    @property
    def error(self) -> str | None:
        return self.config.error

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float = 0,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise ValueError(self.config.error)


def build_openai_compatible_config(prefix: str, *, enabled_default: bool = False) -> OpenAICompatibleConfig:
    load_env_file_if_present(ROOT_ENV_FILE)
    enabled = _profile_env_flag(prefix, "ENABLED", default=enabled_default)
    base_url = _profile_env_value(prefix, "BASE_URL", "") or ""
    model = _profile_env_value(prefix, "MODEL", "") or ""
    api_key = _profile_env_value(prefix, "API_KEY", "") or ""
    timeout_seconds = float(_profile_env_value(prefix, "TIMEOUT_SECONDS", "20") or "20")
    default_extra_body: dict[str, Any] = {}
    if "api.minimaxi.com" in base_url.lower() and _profile_env_flag(prefix, "REASONING_SPLIT", default=True):
        default_extra_body["reasoning_split"] = True
    return OpenAICompatibleConfig(
        prefix=prefix,
        enabled=enabled,
        configured=enabled and bool(base_url and model),
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        default_extra_body=default_extra_body or None,
    )


def build_llm_provider(prefix: str, *, provider_default: str = "openai_compatible", enabled_default: bool = False) -> LLMProvider:
    load_env_file_if_present(ROOT_ENV_FILE)
    provider_name = (_profile_env_value(prefix, "PROVIDER", provider_default) or provider_default).strip().lower()
    if provider_name in {"openai_compatible", "openai"}:
        return OpenAICompatibleProvider(build_openai_compatible_config(prefix, enabled_default=enabled_default))
    return UnsupportedLLMProvider(
        UnsupportedProviderConfig(
            prefix=prefix,
            provider_name=provider_name,
            enabled=_profile_env_flag(prefix, "ENABLED", default=enabled_default),
            configured=False,
            error=f"unsupported LLM provider: {provider_name}",
        )
    )


def build_chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


def extract_message_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("LLM response is missing choices")

    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        joined = "\n".join(part for part in parts if part.strip())
        if joined:
            return joined
    raise ValueError("LLM response content is empty")


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    if start >= 0:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(cleaned)):
            char = cleaned[index]
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    parsed = json.loads(cleaned[start : index + 1])
                    if isinstance(parsed, dict):
                        return parsed
                    break

    raise ValueError("LLM output is not a valid JSON object")


def call_openai_compatible_chat(
    config: OpenAICompatibleConfig,
    *,
    messages: list[dict[str, str]],
    temperature: float = 0,
    response_format: dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    provider = OpenAICompatibleProvider(config)
    return provider.chat(
        messages=messages,
        temperature=temperature,
        response_format=response_format,
        extra_body=extra_body,
    )


def call_openai_compatible_json(
    config: OpenAICompatibleConfig,
    *,
    messages: list[dict[str, str]],
    temperature: float = 0,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    provider = OpenAICompatibleProvider(config)
    return provider.json(
        messages=messages,
        temperature=temperature,
        extra_body=extra_body,
    )