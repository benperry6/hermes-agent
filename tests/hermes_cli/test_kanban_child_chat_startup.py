"""Child ``hermes chat`` must drop inherited Kanban lifecycle ownership.

Defense in depth for the same incident as
``tests/tools/test_kanban_child_chat_env_isolation.py``: even if a child
process is launched with a raw ``os.environ`` copy (outside the terminal
sanitize path), ``cmd_chat`` must refuse to become the board worker unless
it carries the one-shot native launch proof emitted by the dispatcher.
"""
from __future__ import annotations

import os
import sys
import types
from argparse import Namespace

import pytest


_OWNERSHIP_KEYS = (
    "HERMES_KANBAN_TASK",
    "HERMES_KANBAN_RUN_ID",
    "HERMES_KANBAN_WORKSPACE",
    "HERMES_KANBAN_WORKSPACES_ROOT",
    "HERMES_KANBAN_CLAIM_LOCK",
    "HERMES_KANBAN_WORKER_LAUNCH",
)


def _chat_args(**overrides):
    base = {
        "continue_last": None,
        "model": None,
        "provider": None,
        "resume": None,
        "no_restore_cwd": False,
        "toolsets": None,
        "skills": None,
        "tui": False,
        "tui_dev": False,
        "cli": True,
        "verbose": None,
        "quiet": True,
        "query": "hello",
        "query_file": None,
        "image": None,
        "worktree": False,
        "checkpoints": False,
        "pass_session_id": False,
        "max_turns": None,
        "ignore_rules": False,
        "ignore_user_config": False,
        "safe_mode": False,
        "compact": False,
        "source": None,
        "kanban_worker_launch": None,
        "yolo": False,
        "accept_hooks": False,
        "in_dir": None,
        "reasoning": None,
        "run_budget": None,
    }
    base.update(overrides)
    return Namespace(**base)


@pytest.fixture
def main_mod(monkeypatch):
    import hermes_cli.main as mod

    monkeypatch.setattr(mod, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(mod, "_sync_bundled_skills_for_startup", lambda: None)
    monkeypatch.setattr(mod, "_termux_should_prefetch_update_check", lambda: False)
    monkeypatch.setattr(mod, "_pin_kanban_board_env", lambda: None)
    monkeypatch.setattr(mod, "_confirm_startup_expensive_model_override", lambda args: None)
    monkeypatch.setattr(mod, "_resolve_session_by_name_or_id", lambda val: val)
    monkeypatch.setattr(mod, "_oneshot_cleanup_done", False)
    return mod


@pytest.fixture
def fake_cli(monkeypatch):
    captured = {}

    def fake_cli_main(**kwargs):
        captured.update(kwargs)
        captured["env"] = {
            key: os.environ.get(key) for key in (
                *_OWNERSHIP_KEYS,
                "HERMES_SESSION_SOURCE",
                "HERMES_KANBAN_BOARD",
                "HERMES_KANBAN_DB",
            )
        }

    monkeypatch.setitem(sys.modules, "cli", types.SimpleNamespace(main=fake_cli_main))
    import cli

    monkeypatch.setattr(cli, "main", fake_cli_main)
    return captured


def _inherit_worker_identity(
    monkeypatch, tmp_path, task_id="t_parent", launch_marker=None,
):
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "18")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(tmp_path / "workspaces"))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "lock-parent")
    if launch_marker is not None:
        monkeypatch.setenv("HERMES_KANBAN_WORKER_LAUNCH", launch_marker)
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "kanban")


def test_source_tool_child_chat_drops_inherited_ownership(
    main_mod, fake_cli, monkeypatch, tmp_path
):
    _inherit_worker_identity(monkeypatch, tmp_path, launch_marker="t_parent")
    main_mod.cmd_chat(
        _chat_args(
            source="tool",
            kanban_worker_launch="t_parent",
            query="Benchmark Browser Use on books.toscrape.com",
            quiet=True,
        )
    )

    env = fake_cli["env"]
    for key in _OWNERSHIP_KEYS:
        assert env.get(key) is None, key
    assert env["HERMES_SESSION_SOURCE"] == "tool"
    assert env["HERMES_KANBAN_BOARD"] == "default"
    assert env["HERMES_KANBAN_DB"] == str(tmp_path / "kanban.db")


def test_validator_import_failure_fails_closed(
    main_mod, monkeypatch, tmp_path
):
    import builtins

    _inherit_worker_identity(
        monkeypatch,
        tmp_path,
        task_id="t_board_worker",
        launch_marker="t_board_worker",
    )
    real_import = builtins.__import__

    def _fail_validator_import(name, *args, **kwargs):
        if name == "agent.delegation_context":
            raise ImportError("validator unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fail_validator_import)
    main_mod._drop_inherited_kanban_lifecycle(_chat_args(
        source="kanban",
        kanban_worker_launch="t_board_worker",
    ))

    for key in _OWNERSHIP_KEYS:
        assert os.environ.get(key) is None, key
    assert os.environ["HERMES_KANBAN_BOARD"] == "default"
    assert os.environ["HERMES_KANBAN_DB"] == str(tmp_path / "kanban.db")


def test_inherited_kanban_source_without_worker_prompt_drops_ownership(
    main_mod, fake_cli, monkeypatch, tmp_path
):
    _inherit_worker_identity(monkeypatch, tmp_path)
    main_mod.cmd_chat(_chat_args(source=None, query="do a local model benchmark"))

    env = fake_cli["env"]
    for key in _OWNERSHIP_KEYS:
        assert env.get(key) is None, key
    # Inherited source tag is not enough to keep ownership.
    assert env["HERMES_SESSION_SOURCE"] == "kanban"


def test_explicit_board_worker_launch_keeps_ownership_for_any_query_wording(
    main_mod, fake_cli, monkeypatch, tmp_path
):
    _inherit_worker_identity(
        monkeypatch,
        tmp_path,
        task_id="t_board_worker",
        launch_marker="launch-nonce-a",
    )
    main_mod.cmd_chat(
        _chat_args(
            source=None,
            query="execute the assigned board card now",
            kanban_worker_launch="launch-nonce-a",
        )
    )

    env = fake_cli["env"]
    assert env["HERMES_KANBAN_TASK"] == "t_board_worker"
    assert env["HERMES_KANBAN_RUN_ID"] == "18"
    assert env["HERMES_KANBAN_CLAIM_LOCK"] == "lock-parent"
    assert env["HERMES_KANBAN_WORKER_LAUNCH"] is None
    assert env["HERMES_SESSION_SOURCE"] == "kanban"


@pytest.mark.parametrize(
    "source, launch_marker, launch_arg",
    [
        ("kanban", "launch-nonce-a", None),
        ("kanban", None, "launch-nonce-a"),
        ("kanban", "", "launch-nonce-a"),
        ("kanban", "launch-nonce-a", "launch-nonce-b"),
        # The task id is public lifecycle data, not a launch proof.
        ("kanban", "t_board_worker", "t_board_worker"),
        ("kanban", "launch-nonce-a", ""),
        ("tool", "launch-nonce-a", "launch-nonce-a"),
        (" kanban ", "launch-nonce-a", "launch-nonce-a"),
    ],
)
def test_source_task_marker_and_flag_must_all_match(
    main_mod, fake_cli, monkeypatch, tmp_path,
    source, launch_marker, launch_arg,
):
    _inherit_worker_identity(
        monkeypatch,
        tmp_path,
        task_id="t_board_worker",
        launch_marker=launch_marker,
    )
    main_mod.cmd_chat(
        # Human text that exactly resembles the legacy worker prompt is not
        # machine authority.
        _chat_args(
            source=source,
            query="work kanban task t_board_worker",
            kanban_worker_launch=launch_arg,
        )
    )

    env = fake_cli["env"]
    for key in _OWNERSHIP_KEYS:
        assert env.get(key) is None, key


def test_consumed_marker_cannot_authorize_a_nested_chat(
    main_mod, fake_cli, monkeypatch, tmp_path
):
    _inherit_worker_identity(
        monkeypatch,
        tmp_path,
        task_id="t_board_worker",
        launch_marker="launch-nonce-a",
    )
    main_mod.cmd_chat(_chat_args(
        source="kanban",
        query="arbitrary worker wording",
        kanban_worker_launch="launch-nonce-a",
    ))
    assert fake_cli["env"]["HERMES_KANBAN_TASK"] == "t_board_worker"
    assert os.environ.get("HERMES_KANBAN_WORKER_LAUNCH") is None

    # A raw-env child can inherit the runtime task keys. Even if it reconstructs
    # both proof positions from the public task id, startup must reject it.
    monkeypatch.setenv("HERMES_KANBAN_WORKER_LAUNCH", "t_board_worker")
    main_mod.cmd_chat(_chat_args(
        source="kanban",
        query="work kanban task t_board_worker",
        kanban_worker_launch="t_board_worker",
    ))
    for key in _OWNERSHIP_KEYS:
        assert fake_cli["env"].get(key) is None, key


def test_internal_worker_launch_flag_is_accepted_but_hidden_from_help():
    from hermes_cli._parser import build_top_level_parser

    parser, _subparsers, chat_parser = build_top_level_parser()
    args = parser.parse_args([
        "chat", "--kanban-worker-launch", "launch-nonce-a", "-q", "anything",
    ])

    assert args.kanban_worker_launch == "launch-nonce-a"
    assert "--kanban-worker-launch" not in chat_parser.format_help()
