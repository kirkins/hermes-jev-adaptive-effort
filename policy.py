"""Jev-selected reasoning effort, pinned once for each Hermes user turn."""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

if __package__:
    from .jev_client import Decision, JevError, decide
else:  # Direct validation/import of a standalone plugin directory.
    from jev_client import Decision, JevError, decide


logger = logging.getLogger(__name__)

EFFORT_DESCRIPTIONS = {
    "minimal": "The smallest non-zero reasoning budget for a nearly mechanical request.",
    "low": "Only genuinely trivial, casual, or simple read-only work with little ambiguity.",
    "medium": "Straightforward bounded work that benefits from some deliberate reasoning.",
    "high": "Ordinary bounded coding or analysis that needs careful multi-step reasoning.",
    "xhigh": "Demanding multi-step work that needs more deliberation than high without the full max budget.",
    "max": (
        "Complex, ambiguous, agentic, consequential, debugging, planning, or security-sensitive work; "
        "also use when uncertain."
    ),
}
EFFORT_ORDER = ("minimal", "low", "medium", "high", "xhigh", "max")
VALID_MODES = {"disabled", "shadow", "enforce"}


@dataclass(frozen=True)
class PinnedDecision:
    choice: str
    probability: float
    accepted: bool
    reason: str
    input_tokens: int = 0
    output_tokens: int = 0


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        value = part.get("text")
        if isinstance(value, str):
            parts.append(value)
    return "\n".join(parts)


def _offered_efforts(values: Any) -> dict[str, str]:
    normalized = {
        str(value).strip().lower()
        for value in (values or ())
        if str(value).strip().lower() in EFFORT_DESCRIPTIONS
    }
    return {
        effort: EFFORT_DESCRIPTIONS[effort]
        for effort in EFFORT_ORDER
        if effort in normalized
    }


def _cautious_choice(choice: str, offered: dict[str, str]) -> str:
    levels = tuple(offered)
    if choice not in levels:
        return levels[-1]
    index = levels.index(choice)
    return levels[min(index + 1, len(levels) - 1)]


def _openrouter_key() -> str:
    from agent.secret_scope import get_secret

    return str(get_secret("OPENROUTER_API_KEY", "") or "").strip()


class JevEffortPolicy:
    def __init__(
        self,
        *,
        mode: str,
        timeout_seconds: float,
        min_probability: float,
        max_user_chars: int,
        cache_entries: int,
        decide_fn: Callable[..., Decision] = decide,
        secret_fn: Callable[[], str] = _openrouter_key,
    ):
        self.mode = mode if mode in VALID_MODES else "shadow"
        self.timeout_seconds = min(10.0, max(0.1, float(timeout_seconds)))
        self.min_probability = min(1.0, max(0.0, float(min_probability)))
        self.max_user_chars = min(20_000, max(256, int(max_user_chars)))
        self.cache_entries = min(4_096, max(8, int(cache_entries)))
        self._decide_fn = decide_fn
        self._secret_fn = secret_fn
        self._lock = threading.RLock()
        self._cache: OrderedDict[tuple[str, str], PinnedDecision] = OrderedDict()
        self._pending: dict[tuple[str, str], threading.Event] = {}

    def _remember(self, key: tuple[str, str], decision: PinnedDecision) -> None:
        self._cache[key] = decision
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_entries:
            self._cache.popitem(last=False)

    def _failure_decision(
        self,
        *,
        has_conversation_history: bool,
        previous_effort: Any,
        current_effort: str,
        offered: dict[str, str],
        reason: str,
    ) -> PinnedDecision:
        effort = str(previous_effort or "").strip().lower()
        if effort not in offered and has_conversation_history and current_effort in offered:
            effort = current_effort
        if effort in offered:
            return PinnedDecision(effort, 0.0, True, f"reuse_last_after_{reason}")
        return PinnedDecision(tuple(offered)[-1], 0.0, False, reason)

    def _classify(
        self,
        *,
        user_text: str,
        has_conversation_history: bool,
        previous_effort: Any,
        current_effort: str,
        model: str,
        provider: str,
        api_mode: str,
        offered: dict[str, str],
    ) -> PinnedDecision:
        if not user_text.strip():
            return self._failure_decision(
                has_conversation_history=has_conversation_history,
                previous_effort=previous_effort,
                current_effort=current_effort,
                offered=offered,
                reason="missing_user_text",
            )
        try:
            api_key = self._secret_fn()
        except Exception:
            return self._failure_decision(
                has_conversation_history=has_conversation_history,
                previous_effort=previous_effort,
                current_effort=current_effort,
                offered=offered,
                reason="credential_unavailable",
            )
        if not api_key:
            return self._failure_decision(
                has_conversation_history=has_conversation_history,
                previous_effort=previous_effort,
                current_effort=current_effort,
                offered=offered,
                reason="missing_openrouter_credentials",
            )
        from agent.redact import redact_sensitive_text

        redacted = redact_sensitive_text(
            user_text,
            force=True,
            redact_url_credentials=True,
        )
        bounded = redacted[-self.max_user_chars :]
        state = {
            "user_request_untrusted": bounded,
            "selected_model": model,
            "provider": provider,
            "api_mode": api_mode,
            "fixed_policy": {
                "model_must_not_change": True,
                "one_effort_decision_per_user_turn": True,
                "reasoning_must_remain_enabled": True,
            },
        }
        try:
            result = self._decide_fn(
                api_key=api_key,
                timeout_seconds=self.timeout_seconds,
                state=state,
                offered=offered,
            )
        except JevError as exc:
            reason = exc.args[0] if exc.args and isinstance(exc.args[0], str) else "jev_error"
            return self._failure_decision(
                has_conversation_history=has_conversation_history,
                previous_effort=previous_effort,
                current_effort=current_effort,
                offered=offered,
                reason=reason,
            )
        except Exception:
            return self._failure_decision(
                has_conversation_history=has_conversation_history,
                previous_effort=previous_effort,
                current_effort=current_effort,
                offered=offered,
                reason="unexpected_error",
            )
        confident = result.probability >= self.min_probability
        choice = result.choice if confident else _cautious_choice(result.choice, offered)
        return PinnedDecision(
            choice,
            result.probability,
            True,
            "accepted" if confident else "low_probability_step_up",
            int(result.usage.get("input_tokens", 0)),
            int(result.usage.get("output_tokens", 0)),
        )

    def _pinned(self, key: tuple[str, str], **kwargs: Any) -> PinnedDecision:
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached
            pending = self._pending.get(key)
            if pending is None:
                pending = threading.Event()
                self._pending[key] = pending
                owner = True
            else:
                owner = False
        if not owner:
            pending.wait(self.timeout_seconds + 0.25)
            with self._lock:
                cached = self._cache.get(key)
            if cached is not None:
                return cached
            return self._failure_decision(
                has_conversation_history=kwargs["has_conversation_history"],
                previous_effort=kwargs["previous_effort"],
                current_effort=kwargs["current_effort"],
                offered=kwargs["offered"],
                reason="concurrent_timeout",
            )
        try:
            pinned = self._classify(**kwargs)
        finally:
            with self._lock:
                if "pinned" not in locals():
                    pinned = self._failure_decision(
                        has_conversation_history=kwargs["has_conversation_history"],
                        previous_effort=kwargs["previous_effort"],
                        current_effort=kwargs["current_effort"],
                        offered=kwargs["offered"],
                        reason="unexpected_error",
                    )
                self._remember(key, pinned)
                self._pending.pop(key).set()
        return pinned

    def middleware(self, effort: Any = None, **context: Any):
        if self.mode == "disabled":
            return None
        current_effort = str(effort or "").strip().lower()
        if current_effort not in EFFORT_DESCRIPTIONS:
            return None
        if context.get("reasoning_effort_updates_supported") is not True:
            return None
        offered = _offered_efforts(context.get("supported_reasoning_efforts"))
        if len(offered) < 2:
            return None
        session_id = str(context.get("session_id") or "")
        turn_id = str(context.get("turn_id") or "")
        if not session_id or not turn_id:
            return None
        model = str(context.get("model") or "")
        provider = str(context.get("provider") or "")
        api_mode = str(context.get("api_mode") or "")
        started = time.monotonic()
        pinned = self._pinned(
            (session_id, turn_id),
            user_text=_text_content(context.get("user_message")),
            has_conversation_history=context.get("has_conversation_history") is True,
            previous_effort=context.get("previous_effort"),
            current_effort=current_effort,
            model=model,
            provider=provider,
            api_mode=api_mode,
            offered=offered,
        )
        logger.info(
            "jev-adaptive-effort mode=%s model=%s choice=%s probability=%.3f accepted=%s "
            "reason=%s latency_ms=%d input_tokens=%d output_tokens=%d",
            self.mode,
            model,
            pinned.choice,
            pinned.probability,
            pinned.accepted,
            pinned.reason,
            int((time.monotonic() - started) * 1000),
            pinned.input_tokens,
            pinned.output_tokens,
        )
        if self.mode != "enforce" or not pinned.accepted or pinned.choice not in offered:
            return None
        if pinned.choice == current_effort:
            return None
        return {
            "effort": pinned.choice,
            "source": "jev-adaptive-effort",
            "reason": "cache-preserving effort selected for this user turn",
        }
