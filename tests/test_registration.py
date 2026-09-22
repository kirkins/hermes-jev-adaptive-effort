from __future__ import annotations

import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_declares_actual_capabilities():
    manifest = yaml.safe_load((ROOT / "plugin.yaml").read_text())
    assert manifest["name"] == "jev-adaptive-effort"
    assert manifest["provides_middleware"] == ["reasoning_effort"]
    assert [item["name"] for item in manifest["requires_env"]] == ["OPENROUTER_API_KEY"]


def test_register_exposes_only_reasoning_effort_middleware(plugin_package):
    registered = []

    class Context:
        def get_config(self, _name, default=None):
            return default

        def register_middleware(self, kind, callback):
            registered.append((kind, callback))

    plugin_package.register(Context())
    assert len(registered) == 1
    assert registered[0][0] == "reasoning_effort"
    assert callable(registered[0][1])


def test_real_hermes_discovery_loads_plugin(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    plugin_home = hermes_home / "plugins" / "jev-adaptive-effort"
    plugin_home.parent.mkdir(parents=True)
    shutil.copytree(
        ROOT,
        plugin_home,
        ignore=shutil.ignore_patterns(".git", ".pytest_cache", ".ruff_cache", "__pycache__"),
    )
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "plugins": {
                    "enabled": ["jev-adaptive-effort"],
                    "entries": {
                        "jev-adaptive-effort": {"settings": {"mode": "disabled"}},
                    },
                }
            }
        )
    )
    empty_bundled = tmp_path / "bundled"
    empty_bundled.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(empty_bundled))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    from hermes_cli.plugins import PluginManager

    manager = PluginManager()
    manager.discover_and_load()
    loaded = manager._plugins["jev-adaptive-effort"]
    assert loaded.enabled is True
    assert loaded.error is None
    assert loaded.middleware_registered == ["reasoning_effort"]
