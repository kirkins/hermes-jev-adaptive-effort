# Jev Adaptive Effort for Hermes

An opt-in Hermes plugin that asks [TypeSafe Jev 1.13 on OpenRouter](https://openrouter.ai/typesafe/jev-1.13/api) to choose the reasoning effort for each user turn. It never changes the selected model and only runs when Hermes says the active transport can change effort without invalidating the conversation's cached prefix.

This plugin is the reference consumer for the provider-neutral `reasoning_effort` middleware proposed in [NousResearch/hermes-agent#118985](https://github.com/NousResearch/hermes-agent/pull/118985). Until that PR ships in a Hermes release, the plugin requires a Hermes build containing that PR and its cache-preservation dependencies.

## What it does

At the start of a supported user turn, the plugin sends the current user message—after Hermes secret redaction and a configurable length bound—to OpenRouter's experimental Decisions API. Jev selects one of the effort levels that the active Hermes transport declares safe.

- The model/provider never changes.
- Reasoning is never disabled.
- Unsupported routes do not call OpenRouter.
- The full conversation transcript is never given to the plugin or OpenRouter.
- A choice is pinned once per user turn, preventing duplicate billed calls during retries or concurrent delivery.
- A timeout, malformed response, missing credential, or API failure retains the last effort used in that thread. On the first turn it retains the highest supported baseline, normally `max`.
- A valid choice below `min_probability` is hedged upward by one supported effort level.

## Install

While the catalog PR is pending:

```bash
hermes plugins install https://github.com/kirkins/hermes-jev-adaptive-effort --enable
```

If `OPENROUTER_API_KEY` is already configured in the active Hermes profile, the plugin reuses it and the installer does not prompt again. Otherwise Hermes prompts for the key during installation and stores it in that profile's `.env`. Create a key at [OpenRouter](https://openrouter.ai/settings/keys).

The public default is `shadow`: Jev decisions are logged but not applied. After observing the choices, enable enforcement in `~/.hermes/config.yaml`:

```yaml
plugins:
  enabled:
    - jev-adaptive-effort
  entries:
    jev-adaptive-effort:
      settings:
        mode: enforce
        timeout_seconds: 2.5
        min_probability: 0.65
        max_user_chars: 6000
        cache_entries: 256
```

Restart Hermes after changing plugin settings.

Once accepted into the Hermes catalog, installation becomes:

```bash
hermes plugins install jev-adaptive-effort --enable
```

## Supported routes

Hermes core—not this plugin—decides whether a route supports cache-preserving effort updates and which levels are legal. With the dependency PRs, qualified routes include official ChatGPT Responses routes for GPT-5.6 Luna/Terra/Sol and GPT-6 Astra, plus supported Anthropic Messages routes. Unknown routes and ordinary Chat Completions routes fail closed.

## Privacy and cost

Each supported turn in `shadow` or `enforce` mode makes a billed OpenRouter request using your `OPENROUTER_API_KEY`. The request contains:

- The current user message after Hermes secret redaction, truncated to `max_user_chars`.
- The already-selected model, provider, and API mode.
- Fixed instructions that prohibit model switching and disabling reasoning.
- The effort choices supplied by Hermes.

It does not contain the conversation transcript, system prompt, tool results, OpenRouter key, or prior assistant messages. The plugin does not emit OpenRouter attribution headers.

OpenRouter currently exposes Decisions under an `alpha` endpoint. The API may change or be removed. Failures leave the thread on its previous effort rather than changing the model or rewriting its cached prefix.

## Modes

- `disabled`: no Jev request and no change.
- `shadow`: call Jev and log the bounded decision metadata, but keep the current effort.
- `enforce`: apply a valid decision through Hermes's durable cache-preserving mechanism.

Logs contain the chosen effort, probability, categorical failure reason, latency, and token counts. They do not contain prompts, response bodies, or credentials.

## Development

The test suite expects a Hermes checkout containing PR #118985 on `PYTHONPATH`:

```bash
PYTHONPATH=/path/to/hermes-agent /path/to/hermes-agent/.venv/bin/python -m pytest -q
PYTHONPATH=/path/to/hermes-agent /path/to/hermes-agent/.venv/bin/ruff check .
/path/to/hermes-agent/.venv/bin/hermes plugins validate --install-deps .
/path/to/hermes-agent/.venv/bin/hermes plugins doctor . --ci
```

No live OpenRouter request runs in CI.

## License

MIT
