from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def plugin_package():
    package = "_test_jev_adaptive_effort"
    for name in list(sys.modules):
        if name == package or name.startswith(package + "."):
            sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(
        package,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[package] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    yield module
    for name in list(sys.modules):
        if name == package or name.startswith(package + "."):
            sys.modules.pop(name, None)
