from __future__ import annotations

import logging
import threading
import time


def _context(**overrides):
    return {
        "effort": "max",
        "user_message": "Please fix the failing test",
        "has_conversation_history": False,
        "previous_effort": None,
        "session_id": "session-1",
        "turn_id": "turn-1",
        "model": "gpt-5.6-luna-1",
        "provider": "openai-codex",
        "api_mode": "codex_responses",
        "reasoning_effort_updates_supported": True,
        "supported_reasoning_efforts": ("low", "medium", "high", "xhigh", "max"),
    } | overrides


def _policy(package, decide_fn, *, mode="enforce", probability=0.65, secret="openrouter-key"):
    policy_module = __import__(package.__name__ + ".policy", fromlist=["*"])
    return policy_module.JevEffortPolicy(
        mode=mode,
        timeout_seconds=1,
        min_probability=probability,
        max_user_chars=6000,
        cache_entries=32,
        decide_fn=decide_fn,
        secret_fn=lambda: secret,
    )


def _decision(package, choice="high", probability=0.9):
    client = __import__(package.__name__ + ".jev_client", fromlist=["*"])
    return client.Decision(
        choice,
        probability,
        {"input_tokens": 11, "output_tokens": 2, "cost": 0.001},
        client.MODEL,
        "TypeSafe",
    )


def test_enforce_selects_effort_without_changing_model(plugin_package):
    calls = []

    def decide_fn(**kwargs):
        calls.append(kwargs)
        return _decision(plugin_package, "high")

    policy = _policy(plugin_package, decide_fn)
    context = _context()
    result = policy.middleware(**context)

    assert result["effort"] == "high"
    assert context["model"] == "gpt-5.6-luna-1"
    assert calls[0]["api_key"] == "openrouter-key"
    assert "conversation_history" not in calls[0]["state"]


def test_unsupported_route_never_calls_jev(plugin_package):
    calls = []
    policy = _policy(plugin_package, lambda **kwargs: calls.append(kwargs))
    result = policy.middleware(
        **_context(
            reasoning_effort_updates_supported=False,
            supported_reasoning_efforts=(),
        )
    )
    assert result is None
    assert calls == []


def test_low_probability_steps_up_one_supported_level(plugin_package):
    policy = _policy(
        plugin_package,
        lambda **_: _decision(plugin_package, "low", 0.64),
    )
    assert policy.middleware(**_context())["effort"] == "medium"


def test_xhigh_is_a_first_class_supported_choice(plugin_package):
    policy = _policy(
        plugin_package,
        lambda **_: _decision(plugin_package, "xhigh", 0.9),
    )
    assert policy.middleware(**_context())["effort"] == "xhigh"


def test_failure_reuses_last_thread_effort(plugin_package):
    client = __import__(plugin_package.__name__ + ".jev_client", fromlist=["*"])

    def fail(**_kwargs):
        raise client.JevError("rate_limited")

    policy = _policy(plugin_package, fail)
    result = policy.middleware(
        **_context(
            effort="max",
            has_conversation_history=True,
            previous_effort="medium",
        )
    )
    assert result["effort"] == "medium"
    assert policy._cache[("session-1", "turn-1")].reason == "reuse_last_after_rate_limited"


def test_first_turn_failure_keeps_highest_supported_baseline(plugin_package):
    policy = _policy(plugin_package, lambda **_: None, secret="")
    assert policy.middleware(**_context()) is None
    pinned = policy._cache[("session-1", "turn-1")]
    assert pinned.choice == "max"
    assert pinned.accepted is False
    assert pinned.reason == "missing_openrouter_credentials"


def test_shadow_calls_jev_but_does_not_apply_choice(plugin_package):
    calls = []

    def decide_fn(**kwargs):
        calls.append(kwargs)
        return _decision(plugin_package, "low")

    policy = _policy(plugin_package, decide_fn, mode="shadow")
    assert policy.middleware(**_context()) is None
    assert len(calls) == 1


def test_decision_is_pinned_once_per_turn(plugin_package):
    calls = []

    def decide_fn(**kwargs):
        calls.append(kwargs)
        return _decision(plugin_package, "low")

    policy = _policy(plugin_package, decide_fn)
    assert policy.middleware(**_context())["effort"] == "low"
    assert policy.middleware(**_context())["effort"] == "low"
    assert len(calls) == 1


def test_concurrent_same_turn_is_billed_once(plugin_package):
    calls = 0
    lock = threading.Lock()

    def decide_fn(**_kwargs):
        nonlocal calls
        with lock:
            calls += 1
        time.sleep(0.05)
        return _decision(plugin_package, "high")

    policy = _policy(plugin_package, decide_fn)
    results = []
    threads = [threading.Thread(target=lambda: results.append(policy.middleware(**_context()))) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert calls == 1
    assert [result["effort"] for result in results] == ["high"] * 4


def test_prompt_is_redacted_and_key_is_never_logged(plugin_package, caplog):
    captured = {}
    secret = "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789"

    def decide_fn(**kwargs):
        captured.update(kwargs)
        return _decision(plugin_package, "high")

    policy = _policy(plugin_package, decide_fn, secret="openrouter-key")
    caplog.set_level(logging.INFO)
    policy.middleware(**_context(user_message="API_KEY=" + secret))

    assert secret not in captured["state"]["user_request_untrusted"]
    assert "openrouter-key" not in caplog.text
    assert secret not in caplog.text
