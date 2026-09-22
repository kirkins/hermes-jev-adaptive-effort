"""Hermes registration entry point for the Jev adaptive-effort policy."""

from __future__ import annotations

if __package__:
    from .policy import JevEffortPolicy
else:  # Direct validation/import of a standalone plugin directory.
    from policy import JevEffortPolicy


def register(ctx) -> None:
    policy = JevEffortPolicy(
        mode=str(ctx.get_config("mode", "shadow")).lower(),
        timeout_seconds=ctx.get_config("timeout_seconds", 2.5),
        min_probability=ctx.get_config("min_probability", 0.65),
        max_user_chars=ctx.get_config("max_user_chars", 6000),
        cache_entries=ctx.get_config("cache_entries", 256),
    )
    ctx.register_middleware("reasoning_effort", policy.middleware)
