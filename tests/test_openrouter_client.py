from __future__ import annotations

import json

import pytest


class FakeResponse:
    def __init__(self, body: dict, *, content_type: str = "application/json", status: int = 200):
        self.status = status
        self.headers = {"Content-Type": content_type}
        self._body = json.dumps(body).encode()

    def read(self, _limit: int) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class FakeOpener:
    def __init__(self, response):
        self.response = response
        self.request = None
        self.timeout = None

    def open(self, request, timeout):
        self.request = request
        self.timeout = timeout
        return self.response


def _response(client, choice="high"):
    probabilities = {"low": 0.05, "medium": 0.1, "high": 0.8, "max": 0.05}
    return {
        "answers": {
            "effort": {
                "type": "choice",
                "choice": choice,
                "probabilities": probabilities,
            }
        },
        "usage": {"input_tokens": 11, "output_tokens": 2, "cost": 0.000001},
        "model": client.MODEL,
        "provider": "TypeSafe",
    }


def test_direct_openrouter_request_is_bounded_and_authenticated(plugin_package):
    client = __import__(plugin_package.__name__ + ".jev_client", fromlist=["*"])
    opener = FakeOpener(FakeResponse(_response(client)))
    offered = {
        "low": "small",
        "medium": "bounded",
        "high": "careful",
        "max": "complex",
    }

    decision = client.decide(
        api_key="or-secret-value",
        timeout_seconds=2.5,
        state={"user_request_untrusted": "fix this"},
        offered=offered,
        opener=opener,
    )

    assert opener.request.full_url == "https://openrouter.ai/api/alpha/decisions"
    assert opener.request.method == "POST"
    assert opener.request.get_header("Authorization") == "Bearer or-secret-value"
    assert opener.request.get_header("Content-type") == "application/json"
    assert opener.request.get_header("User-agent") == "hermes-jev-adaptive-effort/0.1.1"
    payload = json.loads(opener.request.data)
    assert payload["model"] == "typesafe/jev-1.13"
    assert payload["questions"]["effort"]["criteria"] == offered
    assert "or-secret-value" not in json.dumps(payload)
    assert opener.timeout == 2.5
    assert decision.choice == "high"
    assert decision.probability == 0.8


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["answers"]["effort"]["probabilities"].pop("max"),
        lambda value: value["answers"]["effort"].update({"choice": "xhigh"}),
        lambda value: value["answers"]["effort"]["probabilities"].update({"high": float("nan")}),
        lambda value: value.update({"model": "typesafe/jev-latest"}),
        lambda value: value.update({"usage": {"input_tokens": -1, "output_tokens": 0}}),
    ],
)
def test_response_contract_fails_closed(plugin_package, mutation):
    client = __import__(plugin_package.__name__ + ".jev_client", fromlist=["*"])
    value = _response(client)
    mutation(value)
    with pytest.raises(client.JevError, match="invalid_response"):
        client.validate_response(
            value,
            {"low": "", "medium": "", "high": "", "max": ""},
        )


def test_rejects_non_json_response(plugin_package):
    client = __import__(plugin_package.__name__ + ".jev_client", fromlist=["*"])
    opener = FakeOpener(FakeResponse(_response(client), content_type="text/html"))
    with pytest.raises(client.JevError, match="invalid_content_type"):
        client.decide(
            api_key="secret",
            timeout_seconds=1,
            state={"user_request_untrusted": "hello"},
            offered={"low": "", "max": ""},
            opener=opener,
        )


def test_accepts_confidence_without_probability_vector(plugin_package):
    client = __import__(plugin_package.__name__ + ".jev_client", fromlist=["*"])
    value = _response(client)
    answer = value["answers"]["effort"]
    answer.pop("probabilities")
    answer["confidence"] = 0.72
    decision = client.validate_response(
        value,
        {"low": "", "medium": "", "high": "", "max": ""},
    )
    assert decision.choice == "high"
    assert decision.probability == 0.72


def test_choice_without_confidence_is_treated_as_uncertain(plugin_package):
    client = __import__(plugin_package.__name__ + ".jev_client", fromlist=["*"])
    value = _response(client)
    value["answers"]["effort"].pop("probabilities")
    decision = client.validate_response(
        value,
        {"low": "", "medium": "", "high": "", "max": ""},
    )
    assert decision.choice == "high"
    assert decision.probability == 0.0
