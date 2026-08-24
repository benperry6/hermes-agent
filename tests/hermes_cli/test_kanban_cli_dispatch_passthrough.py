"""Regression tests for #33488 (CLI max_in_progress / max_spawn / per-profile
config passthrough) and #29415 (kanban_swarm humanizer skill ref).

These two fixes are bundled because they're both small, both touch the
kanban dispatcher's CLI surface, and they each guard against a silent
operator footgun that only manifests in long-running setups.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def isolated_kanban_home(monkeypatch):
    """Spin up a fresh HERMES_HOME with a clean kanban DB."""
    test_home = tempfile.mkdtemp(prefix="kanban_cli_passthrough_")
    os.makedirs(os.path.join(test_home, "profiles", "default"), exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", test_home)
    for mod in list(sys.modules.keys()):
        if mod.startswith("hermes_cli") or mod.startswith("hermes_state") or mod == "hermes_constants":
            del sys.modules[mod]
    yield test_home


def test_cli_dispatch_passes_max_in_progress_from_config(isolated_kanban_home, monkeypatch):
    """#33488: hermes kanban dispatch must pass kanban.max_in_progress from
    config to dispatch_once. Without this, the global concurrency cap is
    unreachable from the CLI even though it works from the gateway."""
    from hermes_cli import kanban as kb_cli
    from hermes_cli import kanban_db

    # Configure max_in_progress in the loaded config.
    fake_config = {
        "kanban": {
            "max_in_progress": 3,
            "max_spawn": 5,
            "default_assignee": "default",
            "max_in_progress_per_profile": 2,
        }
    }
    monkeypatch.setattr(
        "hermes_cli.config.load_config", lambda: fake_config
    )

    captured = {}

    def fake_dispatch_once(conn, **kwargs):
        captured.update(kwargs)
        return kanban_db.DispatchResult()

    monkeypatch.setattr(kanban_db, "dispatch_once", fake_dispatch_once)

    args = argparse.Namespace(dry_run=True, max=None, failure_limit=2, json=False)
    kb_cli._cmd_dispatch(args)

    # Every config value must have reached dispatch_once.
    assert captured.get("max_in_progress") == 3, (
        f"CLI must pass kanban.max_in_progress from config; got {captured.get('max_in_progress')!r}"
    )
    assert captured.get("max_spawn") == 5, (
        f"CLI must pass kanban.max_spawn from config when --max is not provided; got {captured.get('max_spawn')!r}"
    )
    assert captured.get("default_assignee") == "default"
    assert captured.get("max_in_progress_per_profile") == 2


def test_cli_max_flag_overrides_config_max_spawn(isolated_kanban_home, monkeypatch):
    """--max on the CLI takes precedence over kanban.max_spawn in config.
    The CLI flag is the explicit operator signal; config is the default."""
    from hermes_cli import kanban as kb_cli
    from hermes_cli import kanban_db

    fake_config = {"kanban": {"max_spawn": 10}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: fake_config)

    captured = {}
    monkeypatch.setattr(
        kanban_db, "dispatch_once",
        lambda conn, **kw: (captured.update(kw), kanban_db.DispatchResult())[1],
    )

    args = argparse.Namespace(dry_run=True, max=2, failure_limit=2, json=False)
    kb_cli._cmd_dispatch(args)

    assert captured.get("max_spawn") == 2, (
        f"CLI --max=2 must override config kanban.max_spawn=10; got {captured.get('max_spawn')!r}"
    )


@pytest.mark.parametrize(
    "configured, expected_key",
    [
        (600, 600),
        (0, 0),
        ("banana", "default"),
        (True, "default"),
        (45, "default"),
    ],
)
def test_cli_dispatch_passes_the_resolved_no_progress_timeout(
    isolated_kanban_home, monkeypatch, configured, expected_key,
):
    """``hermes kanban dispatch`` must resolve ``kanban.no_progress_timeout_seconds``
    through the same validated parser the gateway and daemon use: zero
    disables, invalid values fall back to the default (never to 0)."""
    from hermes_cli import kanban as kb_cli
    from hermes_cli import kanban_db

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"kanban": {"no_progress_timeout_seconds": configured}},
    )
    captured = {}
    monkeypatch.setattr(
        kanban_db, "dispatch_once",
        lambda conn, **kw: (captured.update(kw), kanban_db.DispatchResult())[1],
    )
    args = argparse.Namespace(dry_run=True, max=None, failure_limit=2, json=False)
    kb_cli._cmd_dispatch(args)

    expected = (
        kanban_db.DEFAULT_NO_PROGRESS_TIMEOUT_SECONDS
        if expected_key == "default" else expected_key
    )
    assert captured.get("no_progress_timeout_seconds") == expected


def test_cli_dispatch_without_config_uses_the_same_default(isolated_kanban_home, monkeypatch):
    from hermes_cli import kanban as kb_cli
    from hermes_cli import kanban_db

    def _boom():
        raise RuntimeError("config unreadable")

    monkeypatch.setattr("hermes_cli.config.load_config", _boom)
    captured = {}
    monkeypatch.setattr(
        kanban_db, "dispatch_once",
        lambda conn, **kw: (captured.update(kw), kanban_db.DispatchResult())[1],
    )
    kb_cli._cmd_dispatch(argparse.Namespace(dry_run=True, max=None, failure_limit=2, json=False))
    assert captured.get("no_progress_timeout_seconds") == (
        kanban_db.DEFAULT_NO_PROGRESS_TIMEOUT_SECONDS
    )


def test_cli_dispatch_reports_deferred_no_progress_in_json_and_text(
    isolated_kanban_home, monkeypatch, capsys,
):
    """The detected/deferred outcome must reach the operator on both output
    shapes; dropping it would make a held card look like an idle tick."""
    import json as _json

    from hermes_cli import kanban as kb_cli
    from hermes_cli import kanban_db

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"kanban": {}})
    monkeypatch.setattr(
        kanban_db, "dispatch_once",
        lambda conn, **kw: kanban_db.DispatchResult(no_progress_deferred=["t_held"]),
    )

    kb_cli._cmd_dispatch(argparse.Namespace(dry_run=True, max=None, failure_limit=2, json=True))
    payload = _json.loads(capsys.readouterr().out)
    assert payload["no_progress_deferred"] == ["t_held"]

    kb_cli._cmd_dispatch(argparse.Namespace(dry_run=True, max=None, failure_limit=2, json=False))
    text = capsys.readouterr().out
    assert "No progress (deferred): 1" in text
    assert "t_held" in text
