"""Runtime → board progress/liveness bridge (tools/kanban_tools.py).

``AIAgent._touch_activity`` fires on every stream chunk, every API wait and
every retry notice, and bridges into the kanban board so a dispatcher-spawned
worker is not reclaimed mid-flight. That bridge renewed the *only* lease the
watchdogs looked at, so a worker that reasoned for thousands of tokens with
zero tool calls held its claim forever.

These tests pin the split:

* stream/API activity            → liveness only
* VERIFIED tool execution        → progress (via the tool middleware): the
  call was dispatched (not refused), it is a tool that can change the task's
  world (not read-only, not agent bookkeeping) and its result is a success
* failed / refused / read-only / bookkeeping tool calls → nothing
* ``kanban_heartbeat``           → liveness + commentary, NEVER progress: its
  note is free text the model writes, so gating the lease on it would be a
  spelling test, not a guard
* a durable board transition through the worker toolset → progress

Everything runs against real imports and a temp ``HERMES_HOME``.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tools import kanban_tools as kt


@pytest.fixture
def worker_env(monkeypatch, tmp_path):
    """A dispatcher-spawned worker: isolated home, claimed running task,
    the env vars the dispatcher pins at spawn time."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="worker-test", assignee="test-worker")
        host = kb._claimer_id().split(":", 1)[0]
        kb.claim_task(conn, tid, claimer=f"{host}:worker")
        run_id = kb.get_task(conn, tid).current_run_id
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", f"{host}:worker")
    _reset_rate_limits(monkeypatch)
    return tid


def _reset_rate_limits(monkeypatch):
    """The bridges rate-limit their DB writes; tests drive them back-to-back."""
    monkeypatch.setattr(kt, "_auto_heartbeat_last_attempt", None, raising=False)
    monkeypatch.setattr(kt, "_auto_progress_last_attempt", None, raising=False)
    monkeypatch.setattr(kt, "_AUTO_HEARTBEAT_MIN_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(kt, "_AUTO_PROGRESS_MIN_INTERVAL_SECONDS", 0.0, raising=False)


def _lease(tid):
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        return conn.execute(
            "SELECT last_heartbeat_at, last_progress_at, claim_expires "
            "FROM tasks WHERE id = ?",
            (tid,),
        ).fetchone()
    finally:
        conn.close()


def _rewind(tid, *, heartbeat, progress):
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        conn.execute(
            "UPDATE tasks SET last_heartbeat_at = ?, last_progress_at = ? "
            "WHERE id = ?",
            (heartbeat, progress, tid),
        )
        conn.commit()
    finally:
        conn.close()


OK_TERMINAL = json.dumps({"exit_code": 0, "stdout": "ok", "stderr": ""})
FAILED_TERMINAL = json.dumps({"exit_code": 1, "stdout": "", "stderr": "boom"})
OK_WRITE = json.dumps({"success": True, "bytes_written": 42, "path": "a.py"})
FAILED_WRITE = json.dumps({"error": "Permission denied: a.py"})


# ---------------------------------------------------------------------------
# Liveness bridge: unchanged, and it never touches progress
# ---------------------------------------------------------------------------

def test_stream_activity_renews_liveness_only(worker_env):
    """Reproduces the masking: thousands of stream tokens keep the claim
    alive but must leave the progress lease untouched."""
    old = int(time.time()) - 9000
    _rewind(worker_env, heartbeat=old, progress=old)

    for _ in range(10):
        kt.heartbeat_current_worker_from_env()

    row = _lease(worker_env)
    assert row["last_heartbeat_at"] > old
    assert row["last_progress_at"] == old


def _activity_stub():
    """Minimal object carrying only what ``_touch_activity`` reads."""
    import run_agent as _ra

    class _Stub:
        session_id = ""
        _session_db = None
        _last_activity_ts = 0.0
        _last_activity_desc = ""
        _last_activity_provenance = None
        _session_activity_last_persist_mono = 0.0

    stub = _Stub()
    stub._touch_activity = _ra.AIAgent._touch_activity.__get__(stub)
    stub._persist_session_activity_if_due = (
        _ra.AIAgent._persist_session_activity_if_due.__get__(stub)
    )
    return stub


def test_touch_activity_stream_chunk_does_not_renew_progress(worker_env):
    """E2E through the real ``AIAgent._touch_activity``: the exact call the
    streaming path makes on every chunk."""
    old = int(time.time()) - 9000
    _rewind(worker_env, heartbeat=old, progress=old)

    agent = _activity_stub()
    for _ in range(20):
        agent._touch_activity("receiving stream response")
        agent._touch_activity("waiting for non-streaming API response")

    row = _lease(worker_env)
    assert row["last_heartbeat_at"] > old, "liveness must still be bridged"
    assert row["last_progress_at"] == old, (
        "stream tokens must never renew the progress lease"
    )


# ---------------------------------------------------------------------------
# Progress bridge: only a verified tool execution counts
# ---------------------------------------------------------------------------

def test_verified_tool_success_renews_progress(worker_env):
    old = int(time.time()) - 9000
    _rewind(worker_env, heartbeat=old, progress=old)

    assert kt.note_tool_progress_from_env("terminal", OK_TERMINAL) is True
    assert _lease(worker_env)["last_progress_at"] > old

    _rewind(worker_env, heartbeat=old, progress=old)
    assert kt.note_tool_progress_from_env("write_file", OK_WRITE) is True
    assert _lease(worker_env)["last_progress_at"] > old


@pytest.mark.parametrize(
    "tool_name, result",
    [
        ("terminal", FAILED_TERMINAL),
        ("write_file", FAILED_WRITE),
        ("patch", json.dumps({"success": False, "error": "hunk failed"})),
        ("execute_code", "Error executing tool 'execute_code': boom"),
        ("terminal", json.dumps({"error": "Tool execution cancelled by user interrupt",
                                 "status": "cancelled"})),
    ],
)
def test_failed_tool_call_does_not_renew_progress(worker_env, tool_name, result):
    old = int(time.time()) - 9000
    _rewind(worker_env, heartbeat=old, progress=old)

    ok, reason = kt.tool_call_is_progress_evidence(tool_name, result)
    assert (ok, reason) == (False, "failed")
    assert kt.note_tool_progress_from_env(tool_name, result) is False
    assert _lease(worker_env)["last_progress_at"] == old


def test_refused_tool_call_does_not_renew_progress(worker_env):
    """A call blocked by scope/plugin/guardrail policy never ran."""
    old = int(time.time()) - 9000
    _rewind(worker_env, heartbeat=old, progress=old)

    ok, reason = kt.tool_call_is_progress_evidence(
        "terminal", json.dumps({"error": "blocked"}), blocked=True,
    )
    assert (ok, reason) == (False, "refused")
    assert kt.note_tool_progress_from_env("terminal", OK_TERMINAL, blocked=True) is False
    assert _lease(worker_env)["last_progress_at"] == old


def test_unobserved_result_does_not_renew_progress(worker_env):
    """'A tool was attempted' is not evidence; only an observed success is."""
    old = int(time.time()) - 9000
    _rewind(worker_env, heartbeat=old, progress=old)
    assert kt.tool_call_is_progress_evidence("terminal", None) == (False, "unverified")
    assert kt.note_tool_progress_from_env("terminal") is False
    assert kt.note_tool_progress_from_env("terminal", None) is False
    assert _lease(worker_env)["last_progress_at"] == old


@pytest.mark.parametrize(
    "tool_name", ["read_file", "search_files", "web_search", "web_extract",
                  "session_search", "browser_snapshot", "read_terminal",
                  "kanban_show", "kanban_list", "kanban_attachments"],
)
def test_read_only_tool_does_not_renew_progress(worker_env, tool_name):
    """Reading cannot change the task's world, so a loop of reads is still a
    loop. The classification is the one the interrupt-safety code already
    uses (``agent.tool_result_classification.NO_EFFECT_TOOL_NAMES``)."""
    old = int(time.time()) - 9000
    _rewind(worker_env, heartbeat=old, progress=old)

    ok, reason = kt.tool_call_is_progress_evidence(tool_name, json.dumps({"content": "x"}))
    assert (ok, reason) == (False, "read_only")
    assert kt.note_tool_progress_from_env(tool_name, json.dumps({"content": "x"})) is False
    assert _lease(worker_env)["last_progress_at"] == old


@pytest.mark.parametrize("tool_name", ["kanban_heartbeat", "todo", "memory"])
def test_bookkeeping_tool_does_not_renew_progress(worker_env, tool_name):
    """Tools whose only effect is the agent's own state (the lease, its todo
    list, its memory store) can be called at will by a looping model and
    say nothing about the task."""
    old = int(time.time()) - 9000
    _rewind(worker_env, heartbeat=old, progress=old)

    ok, reason = kt.tool_call_is_progress_evidence(tool_name, json.dumps({"ok": True}))
    assert (ok, reason) == (False, "non_task")
    assert kt.note_tool_progress_from_env(tool_name, json.dumps({"ok": True})) is False
    assert _lease(worker_env)["last_progress_at"] == old
    assert tool_name in kt.NON_PROGRESS_TOOL_NAMES


def test_progress_evidence_predicate_is_strict_in_order(worker_env):
    """Refusal beats classification beats result: a blocked read-only call
    reports ``refused``, and a successful mutating call is ``verified``."""
    assert kt.tool_call_is_progress_evidence("read_file", "x", blocked=True) == (False, "refused")
    assert kt.tool_call_is_progress_evidence("terminal", OK_TERMINAL) == (True, "verified")
    assert kt.tool_call_is_progress_evidence("patch", json.dumps({"success": True})) == (True, "verified")
    # Multimodal results are dicts, not strings — a success shape.
    assert kt.tool_call_is_progress_evidence(
        "browser_navigate", {"_multimodal": True, "text": "ok"},
    ) == (True, "verified")


@pytest.mark.parametrize(
    "tool_name, action",
    [
        ("process", "list"), ("process", "poll"), ("process", "log"),
        ("process", "wait"),
        ("computer_use", "capture"), ("computer_use", "cua_browser_state"),
        ("cronjob", "list"), ("delegate_task", "list"),
    ],
)
def test_read_only_action_of_a_multiplexed_tool_is_not_progress(
    worker_env, tool_name, action,
):
    assert kt.tool_call_is_progress_evidence(
        tool_name, {"status": "ok"}, tool_args={"action": action},
    ) == (False, "read_only")


@pytest.mark.parametrize(
    "result",
    [
        {"_multimodal": True, "error": "permission denied", "status": "error"},
        {"ok": False, "action": "click", "message": "No active window"},
        {"ok": False, "status": "refused", "code": "typed_browser_unavailable"},
        {"effect": "suspected_noop"},
        {"success": False},
        {"verified": False},
        {"exit_code": 1},
    ],
)
def test_mapping_and_multimodal_failures_are_not_progress(worker_env, result):
    assert kt.tool_call_is_progress_evidence("computer_use", result) == (
        False, "failed",
    )


def test_bridge_is_inert_outside_a_worker(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    _reset_rate_limits(monkeypatch)
    assert kt.note_tool_progress_from_env("terminal", OK_TERMINAL) is False


def test_wrong_run_cannot_renew_progress(worker_env, monkeypatch):
    """A worker whose run was reclaimed must not resurrect the lease of the
    run that replaced it."""
    old = int(time.time()) - 9000
    _rewind(worker_env, heartbeat=old, progress=old)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "999999")
    kt.note_tool_progress_from_env("terminal", OK_TERMINAL)
    assert _lease(worker_env)["last_progress_at"] == old


def test_wrong_claim_cannot_renew_progress(worker_env, monkeypatch):
    old = int(time.time()) - 9000
    _rewind(worker_env, heartbeat=old, progress=old)
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "otherhost:intruder")
    kt.note_tool_progress_from_env("terminal", OK_TERMINAL)
    assert _lease(worker_env)["last_progress_at"] == old


def test_wrong_task_cannot_renew_progress(worker_env, monkeypatch):
    """The env names a task this process does not own (not running, or a
    different card): nothing moves, on either card."""
    from hermes_cli import kanban_db as kb

    old = int(time.time()) - 9000
    _rewind(worker_env, heartbeat=old, progress=old)
    conn = kb.connect()
    try:
        other = kb.create_task(conn, title="someone else's", assignee="x")
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", other)
    kt.note_tool_progress_from_env("terminal", OK_TERMINAL)
    assert _lease(worker_env)["last_progress_at"] == old
    assert _lease(other)["last_progress_at"] is None


def test_non_dispatcher_owned_context_cannot_renew_progress(worker_env):
    """A cron job fired in-process from a worker inherits the worker's env
    but does not own its task; its tool calls are not the worker's work."""
    from agent.delegation_context import non_dispatcher_owned_context

    old = int(time.time()) - 9000
    _rewind(worker_env, heartbeat=old, progress=old)
    with non_dispatcher_owned_context():
        assert kt.note_tool_progress_from_env("terminal", OK_TERMINAL) is False
    assert _lease(worker_env)["last_progress_at"] == old


def test_bridge_never_raises(monkeypatch, tmp_path):
    """A bridge failure must never break the agent loop."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_does_not_exist")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "nope"))
    _reset_rate_limits(monkeypatch)
    assert kt.note_tool_progress_from_env("terminal", OK_TERMINAL) in (True, False)
    assert kt.note_tool_progress_from_env("terminal", object()) in (True, False)


# ---------------------------------------------------------------------------
# Rate limiter: thread-safe, independent of the liveness limiter
# ---------------------------------------------------------------------------

def test_progress_write_is_rate_limited(worker_env, monkeypatch):
    """A tool call every few seconds for hours must not become a DB write
    every few seconds for hours — and a failed call must not consume the
    window a later success needs."""
    monkeypatch.setattr(kt, "_AUTO_PROGRESS_MIN_INTERVAL_SECONDS", 3600.0)
    assert kt.note_tool_progress_from_env("terminal", FAILED_TERMINAL) is False
    assert kt.note_tool_progress_from_env("terminal", OK_TERMINAL) is True
    assert kt.note_tool_progress_from_env("terminal", OK_TERMINAL) is False


@pytest.mark.parametrize("uptime", [0.0, 1.0, 3599.999])
def test_progress_limiter_admits_the_first_event_at_any_uptime(
    worker_env, monkeypatch, uptime,
):
    """None is the never-attempted sentinel; every float is a real stamp. A
    worker launched right after boot must not lose its first event merely
    because monotonic uptime is smaller than the window."""
    monkeypatch.setattr(kt, "_AUTO_PROGRESS_MIN_INTERVAL_SECONDS", 3600.0)
    monkeypatch.setattr(time, "monotonic", lambda: uptime)
    assert kt.note_tool_progress_from_env("terminal", OK_TERMINAL) is True
    assert kt._auto_progress_last_attempt == uptime
    assert kt.note_tool_progress_from_env("terminal", OK_TERMINAL) is False


@pytest.mark.parametrize("uptime", [0.0, 1.0, 3599.999])
def test_heartbeat_limiter_admits_the_first_event_at_any_uptime(
    worker_env, monkeypatch, uptime,
):
    monkeypatch.setattr(kt, "_AUTO_HEARTBEAT_MIN_INTERVAL_SECONDS", 3600.0)
    monkeypatch.setattr(time, "monotonic", lambda: uptime)
    assert kt.heartbeat_current_worker_from_env() is True
    assert kt._auto_heartbeat_last_attempt == uptime
    assert kt.heartbeat_current_worker_from_env() is False


def test_progress_limiter_state_is_guarded_by_a_lock(worker_env, monkeypatch):
    """Concurrent tool calls complete on worker threads and all touch the
    module-level limiter timestamp. Read-compare-write without a lock lets
    two threads both observe the old value and both write, which is how a
    limiter silently stops limiting."""
    assert isinstance(kt._auto_progress_lock, type(threading.Lock()))
    monkeypatch.setattr(kt, "_AUTO_PROGRESS_MIN_INTERVAL_SECONDS", 3600.0)

    admitted: list[bool] = []
    barrier = threading.Barrier(8)

    def _hit():
        barrier.wait()
        admitted.append(kt.note_tool_progress_from_env("terminal", OK_TERMINAL))

    threads = [threading.Thread(target=_hit) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert sum(1 for a in admitted if a) == 1, admitted


def test_a_recent_liveness_write_never_suppresses_a_progress_stamp(worker_env, monkeypatch):
    monkeypatch.setattr(kt, "_AUTO_HEARTBEAT_MIN_INTERVAL_SECONDS", 3600.0)
    monkeypatch.setattr(kt, "_AUTO_PROGRESS_MIN_INTERVAL_SECONDS", 3600.0)
    old = int(time.time()) - 9000
    _rewind(worker_env, heartbeat=old, progress=old)
    assert kt.heartbeat_current_worker_from_env() is True
    assert kt.note_tool_progress_from_env("terminal", OK_TERMINAL) is True
    row = _lease(worker_env)
    assert row["last_heartbeat_at"] > old and row["last_progress_at"] > old


def test_limiter_sentinel_is_none_on_fresh_import():
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    probe = (
        "from tools import kanban_tools as kt; "
        "assert kt._auto_progress_last_attempt is None; "
        "assert kt._auto_heartbeat_last_attempt is None"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=repo_root,
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# The tool executor is the verified middleware
# ---------------------------------------------------------------------------

def _make_agent(monkeypatch):
    """Minimal AIAgent-like stub (mirrors tests/run_agent/test_tool_activity_heartbeat.py)."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "")
    import run_agent as _ra

    class _Stub:
        _interrupt_requested = False
        _interrupt_message = None
        log_prefix = ""
        quiet_mode = True
        verbose_logging = False
        log_prefix_chars = 200
        _checkpoint_mgr = MagicMock(enabled=False)
        tool_progress_callback = None
        tool_start_callback = None
        tool_complete_callback = None
        tool_progress_mode = "off"
        _todo_store = MagicMock()
        _session_db = None
        valid_tool_names = set()
        _turns_since_memory = 0
        _iters_since_skill = 0
        _current_tool = None
        _last_activity = 0.0
        session_id = ""
        _current_turn_id = ""
        _current_api_request_id = ""

        def __init__(self):
            self._tool_worker_threads: set = set()
            self._tool_worker_threads_lock = threading.Lock()
            self._active_children_lock = threading.Lock()

        def _touch_activity(self, desc):
            self._last_activity = time.time()

        def _vprint(self, msg, force=False):
            pass

        def _safe_print(self, msg):
            pass

        def _should_emit_quiet_tool_messages(self):
            return False

        def _should_start_quiet_spinner(self):
            return False

        def _has_stream_consumers(self):
            return False

        def _tool_result_content_for_active_model(self, name, result):
            return result

        def _record_file_mutation_result(self, *a, **kw):
            pass

        def _apply_pending_steer_to_tool_results(self, *a, **kw):
            pass

    stub = _Stub()
    stub._subdirectory_hints = MagicMock()
    stub._subdirectory_hints.check_tool_call = lambda *a, **kw: None
    stub._flush_messages_to_session_db = lambda *a, **kw: None
    stub._append_guardrail_observation = lambda name, result, *a, **kw: result
    stub.interrupt = _ra.AIAgent.interrupt.__get__(stub)
    stub.clear_interrupt = _ra.AIAgent.clear_interrupt.__get__(stub)
    stub._guardrail_block_result = lambda d: json.dumps({"error": "blocked"})
    stub._tool_guardrails = MagicMock(
        before_call=lambda name, args: MagicMock(allows_execution=True)
    )
    return stub


def _run_middleware(
    agent, *, function_name, execute, scope_block=None, function_args=None,
):
    import agent.tool_executor as te

    return te._run_agent_tool_execution_middleware(
        agent,
        function_name=function_name,
        function_args=function_args or {"command": "true"},
        effective_task_id="task",
        tool_call_id="tc1",
        execute=execute,
        scope_block=scope_block,
        display_index=1,
    )


def test_executor_notes_progress_for_a_verified_success(worker_env, monkeypatch):
    old = int(time.time()) - 9000
    _rewind(worker_env, heartbeat=old, progress=old)
    agent = _make_agent(monkeypatch)

    managed = _run_middleware(
        agent, function_name="terminal", execute=lambda _args: OK_TERMINAL,
    )
    assert managed.dispatched is True and managed.blocked is False
    assert _lease(worker_env)["last_progress_at"] > old


def test_executor_does_not_note_progress_for_a_failed_result(worker_env, monkeypatch):
    old = int(time.time()) - 9000
    _rewind(worker_env, heartbeat=old, progress=old)
    agent = _make_agent(monkeypatch)

    managed = _run_middleware(
        agent, function_name="terminal", execute=lambda _args: FAILED_TERMINAL,
    )
    assert managed.dispatched is True
    assert _lease(worker_env)["last_progress_at"] == old


def test_executor_does_not_note_progress_when_the_tool_raises(worker_env, monkeypatch):
    old = int(time.time()) - 9000
    _rewind(worker_env, heartbeat=old, progress=old)
    agent = _make_agent(monkeypatch)

    def _boom(_args):
        raise RuntimeError("tool exploded")

    with pytest.raises(RuntimeError):
        _run_middleware(agent, function_name="terminal", execute=_boom)
    assert _lease(worker_env)["last_progress_at"] == old


def test_executor_does_not_note_progress_for_a_refused_call(worker_env, monkeypatch):
    old = int(time.time()) - 9000
    _rewind(worker_env, heartbeat=old, progress=old)
    agent = _make_agent(monkeypatch)
    calls: list = []

    managed = _run_middleware(
        agent, function_name="terminal",
        execute=lambda _args: calls.append(1) or OK_TERMINAL,
        scope_block="terminal is not available in this scope",
    )
    assert managed.blocked is True and calls == []
    assert _lease(worker_env)["last_progress_at"] == old


def test_executor_does_not_note_progress_for_a_read_only_tool(worker_env, monkeypatch):
    old = int(time.time()) - 9000
    _rewind(worker_env, heartbeat=old, progress=old)
    agent = _make_agent(monkeypatch)
    _run_middleware(
        agent, function_name="read_file",
        execute=lambda _args: json.dumps({"content": "hello"}),
    )
    assert _lease(worker_env)["last_progress_at"] == old


def test_executor_passes_action_args_to_progress_classifier(worker_env, monkeypatch):
    old = int(time.time()) - 9000
    _rewind(worker_env, heartbeat=old, progress=old)
    agent = _make_agent(monkeypatch)
    _run_middleware(
        agent,
        function_name="process",
        function_args={"action": "poll", "session_id": "proc_1"},
        execute=lambda _args: json.dumps({"status": "running"}),
    )
    assert _lease(worker_env)["last_progress_at"] == old


def test_executor_bridge_is_inert_outside_a_worker_and_never_raises(monkeypatch, tmp_path):
    import agent.tool_executor as te

    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    te._note_verified_tool_progress("terminal", OK_TERMINAL, {})  # no env: no-op
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_does_not_exist")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "nope"))
    te._note_verified_tool_progress("terminal", OK_TERMINAL, {})  # must not raise


# ---------------------------------------------------------------------------
# kanban_heartbeat: liveness + commentary, NEVER progress
# ---------------------------------------------------------------------------

def test_heartbeat_tool_renews_liveness_and_records_the_note_but_never_progress(worker_env):
    """A/B/A/B defeats a previous-note dedupe; twenty unique notes defeat any
    'is this new?' test. It must not matter — no note renews progress."""
    from hermes_cli import kanban_db as kb

    old = int(time.time()) - 9000
    _rewind(worker_env, heartbeat=old, progress=old)

    notes = [("compiling module A" if i % 2 == 0 else "compiling module B") for i in range(8)]
    notes += [f"step {i}: analysed subsystem {i} and chose approach {i}" for i in range(20)]
    notes.append("wrote migrations/003_users.sql; ran pytest -q; 41 passed")
    for note in notes:
        d = json.loads(kt._handle_heartbeat({"note": note}))
        assert d.get("ok") is True, note
        assert _lease(worker_env)["last_progress_at"] == old, note
    assert _lease(worker_env)["last_heartbeat_at"] > old

    conn = kb.connect()
    try:
        recorded = [
            json.loads(r["payload"])["note"]
            for r in conn.execute(
                "SELECT payload FROM task_events WHERE task_id = ? "
                "AND kind = 'heartbeat' AND payload IS NOT NULL ORDER BY id",
                (worker_env,),
            )
        ]
    finally:
        conn.close()
    assert recorded == notes


def test_heartbeat_schema_says_liveness_only(worker_env):
    """The schema is the contract the model reads. It must not tell the
    model that writing a good enough note keeps the claim."""
    desc = kt.KANBAN_HEARTBEAT_SCHEMA["description"].lower()
    note_desc = (
        kt.KANBAN_HEARTBEAT_SCHEMA["parameters"]["properties"]["note"]["description"].lower()
    )
    assert "liveness" in desc
    assert "progress" in desc
    for lure in ("renews the progress", "renews progress", "evidence"):
        assert lure not in desc + " " + note_desc, lure


# ---------------------------------------------------------------------------
# Board transitions through the worker toolset
# ---------------------------------------------------------------------------

def test_worker_comment_renews_progress(worker_env):
    old = int(time.time()) - 9000
    _rewind(worker_env, heartbeat=old, progress=old)
    kt._handle_comment({"task_id": worker_env, "body": "hotspot: a.py — churn"})
    assert _lease(worker_env)["last_progress_at"] > old


# ---------------------------------------------------------------------------
# E2E: agent activity -> board leases -> dispatcher decision
# ---------------------------------------------------------------------------

def _dispatch(monkeypatch):
    """One real dispatcher tick beside a live (mocked) host-local worker.
    Any attempt to signal a process fails the test."""
    from hermes_cli import kanban_db as kb

    def _boom(*_a, **_kw):
        raise AssertionError("no-progress detection must never signal a worker")

    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(kb, "_terminate_reclaimed_worker", _boom)
    monkeypatch.setattr(kb.os, "kill", _boom, raising=False)
    conn = kb.connect()
    try:
        return kb.dispatch_once(conn, spawn_fn=lambda *a, **kw: None)
    finally:
        conn.close()


def _set_pid(tid, pid=4242):
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        kb._set_worker_pid(conn, tid, pid)
    finally:
        conn.close()


def _configured_limit():
    from hermes_cli import kanban_db as kb
    from hermes_cli.config import load_config
    return kb.resolve_no_progress_timeout_seconds(
        (load_config().get("kanban") or {}).get("no_progress_timeout_seconds")
    )


def test_e2e_reasoning_only_worker_is_detected_and_deferred(worker_env, monkeypatch):
    """The reproduced incident, end to end: a worker streams for a long time,
    calls no tool, transitions nothing — and the dispatcher records exactly
    that, holds the claim, and touches nothing else."""
    from hermes_cli import kanban_db as kb

    _set_pid(worker_env)
    agent = _activity_stub()
    for _ in range(50):
        agent._touch_activity("receiving stream response")

    now = int(time.time())
    _rewind(worker_env, heartbeat=now, progress=now - _configured_limit() - 60)

    result = _dispatch(monkeypatch)
    assert result.no_progress_deferred == [worker_env]
    assert result.crashed == [] and result.stale == [] and result.timed_out == []
    assert result.auto_blocked == [] and result.reclaimed == 0

    conn = kb.connect()
    try:
        task = kb.get_task(conn, worker_env)
        assert task.status == "running"
        assert task.consecutive_failures == 0
        assert task.worker_pid == 4242
        assert task.claim_expires >= now + kb.NO_PROGRESS_DEFER_GRACE_SECONDS
        runs = conn.execute(
            "SELECT status, outcome FROM task_runs WHERE task_id = ?", (worker_env,),
        ).fetchall()
        assert [(r["status"], r["outcome"]) for r in runs] == [("running", None)]
        receipts = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'no_progress_deferred'",
            (worker_env,),
        ).fetchall()
        assert len(receipts) == 1
        payload = json.loads(receipts[0]["payload"])
        assert payload["liveness"] == "fresh"
        assert payload["reason"] == "authority_unavailable"
        assert kb.list_comments(conn, worker_env) == []
    finally:
        conn.close()


def test_e2e_worker_with_a_verified_tool_call_is_not_reported(worker_env, monkeypatch):
    """Same elapsed wall-clock, one real verified tool call: quiet."""
    from hermes_cli import kanban_db as kb

    _set_pid(worker_env)
    now = int(time.time())
    _rewind(worker_env, heartbeat=now, progress=now - _configured_limit() - 60)

    import agent.tool_executor as te
    te._note_verified_tool_progress("terminal", OK_TERMINAL, {})

    result = _dispatch(monkeypatch)
    assert result.no_progress_deferred == []
    conn = kb.connect()
    try:
        assert kb.get_task(conn, worker_env).status == "running"
    finally:
        conn.close()


def test_e2e_worker_with_only_failed_tool_calls_is_reported(worker_env, monkeypatch):
    _set_pid(worker_env)
    now = int(time.time())
    _rewind(worker_env, heartbeat=now, progress=now - _configured_limit() - 60)

    import agent.tool_executor as te
    for _ in range(5):
        te._note_verified_tool_progress("terminal", FAILED_TERMINAL, {})

    assert _dispatch(monkeypatch).no_progress_deferred == [worker_env]
