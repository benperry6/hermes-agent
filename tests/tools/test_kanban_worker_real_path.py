"""Kanban workers must keep the native single-query approval policy.

The dispatcher launches every worker through ``hermes chat -q``.  The child
therefore uses ``HERMES_SINGLE_QUERY_SESSION`` and
``approvals.single_query_mode``; dispatcher-level approval bypass state must
not replace that profile-local worker policy.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def _make_task(kb):
    return kb.Task(
        id="t_real_path",
        title="real path approval",
        body=None,
        assignee="worker",
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind="dir",
        workspace_path=None,
        claim_lock="lock",
        claim_expires=None,
        tenant=None,
        current_run_id=7,
    )


def _capture_worker(monkeypatch, tmp_path):
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "worker"
    profile.mkdir(parents=True)
    profile.joinpath("config.yaml").write_text(
        "approvals:\n  single_query_mode: deny\n",
        encoding="utf-8",
    )
    root.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_YOLO_MODE", "1")

    from hermes_cli import kanban_db as kb

    captured = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env") or {})
        return FakeProc()

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    # Keep the interception scoped to _default_spawn so the behavior test can
    # launch a fresh Python child with the captured worker environment.
    with monkeypatch.context() as spawn_patch:
        spawn_patch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
        spawn_patch.setattr(subprocess, "Popen", fake_popen)
        kb._default_spawn(_make_task(kb), str(workspace))
    return captured


def test_dispatcher_yolo_is_not_inherited_by_worker(monkeypatch, tmp_path):
    captured = _capture_worker(monkeypatch, tmp_path)

    assert "HERMES_YOLO_MODE" not in captured["env"]
    assert captured["cmd"][-3:-1] == ["chat", "-q"]


def test_worker_single_query_deny_survives_dispatcher_yolo(monkeypatch, tmp_path):
    captured = _capture_worker(monkeypatch, tmp_path)
    worker_env = captured["env"]
    # cli.main sets this immediately after entering the -q path.  Import the
    # approval module only after the worker env is installed so its process
    # scoped YOLO flag is frozen exactly as it would be in the real child.
    worker_env["HERMES_SINGLE_QUERY_SESSION"] = "1"

    code = """
import json
from tools.approval import (
    check_all_command_guards,
    check_dangerous_command,
    check_execute_code_guard,
    request_tool_approval,
)
results = {
    "dangerous": check_dangerous_command("rm -rf /tmp/stuff", "local"),
    "combined": check_all_command_guards("rm -rf /tmp/stuff", "local"),
    "execute_code": check_execute_code_guard("import os", "local"),
    "plugin": request_tool_approval("write_file", "writes protected files"),
}
print(json.dumps(results))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        env=worker_env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    results = json.loads(completed.stdout)
    assert set(results) == {"dangerous", "combined", "execute_code", "plugin"}
    for gate, result in results.items():
        assert result["approved"] is False, f"{gate} bypassed worker deny: {result}"
        assert "single_query_mode" in (result.get("message") or ""), result
