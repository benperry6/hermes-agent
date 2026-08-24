"""Linux-only Kanban worker identity and quiescence contracts."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _columns(conn, table: str) -> set[str]:
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}


def _synthetic_identity(pid: int = 4242) -> dict:
    return {
        "v": kb.WORKER_IDENTITY_VERSION,
        "scheme": "linux_proc",
        "pid": pid,
        "boot_id": kb._read_linux_boot_id(),
        "pid_namespace": os.readlink("/proc/self/ns/pid"),
        "starttime": 99,
        "pgid": pid,
    }


def _persist_worker_identity(conn, task_id: str, pid: int = 4242) -> dict:
    identity = _synthetic_identity(pid)
    row = conn.execute(
        "SELECT current_run_id FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()
    assert row is not None and row["current_run_id"] is not None
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET worker_pid = ? WHERE id = ?", (pid, task_id),
        )
        conn.execute(
            "UPDATE task_runs SET worker_pid = ?, worker_identity = ? WHERE id = ?",
            (pid, json.dumps(identity), row["current_run_id"]),
        )
    return identity


@pytest.mark.linux_only
def test_worker_identity_is_run_scoped_without_legacy_backfill(kanban_home):
    with kb.connect() as conn:
        assert "worker_identity" in _columns(conn, "task_runs")
        assert "worker_identity" not in _columns(conn, "tasks")


@pytest.mark.linux_only
def test_worker_identity_migration_never_blesses_existing_pid(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="legacy", assignee="worker")
        claimed = kb.claim_task(conn, task_id, claimer="legacy:worker")
        assert claimed is not None
        conn.execute(
            "UPDATE task_runs SET worker_pid = 4242 WHERE id = ?",
            (claimed.current_run_id,),
        )
        conn.execute("ALTER TABLE task_runs DROP COLUMN worker_identity")
        conn.commit()

        kb._migrate_add_optional_columns(conn)

        assert "worker_identity" in _columns(conn, "task_runs")
        row = conn.execute(
            "SELECT worker_pid, worker_identity FROM task_runs WHERE id = ?",
            (claimed.current_run_id,),
        ).fetchone()
        assert row["worker_pid"] == 4242
        assert row["worker_identity"] is None


@pytest.mark.linux_only
@pytest.mark.live_system_guard_bypass
def test_capture_worker_identity_uses_exact_linux_birth_and_group_proof():
    assert hasattr(kb, "capture_worker_identity")
    assert hasattr(kb, "verify_worker_identity")
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    try:
        identity = kb.capture_worker_identity(proc.pid)
        assert identity is not None
        assert identity["scheme"] == "linux_proc"
        assert identity["pid"] == proc.pid
        assert identity["boot_id"] == Path(
            "/proc/sys/kernel/random/boot_id"
        ).read_text(encoding="utf-8").strip()
        assert identity["pid_namespace"] == os.readlink(
            f"/proc/{proc.pid}/ns/pid"
        )
        assert identity["starttime"] > 0
        assert identity["pgid"] == proc.pid
        assert kb.verify_worker_identity(proc.pid, json.dumps(identity)) == (
            True,
            "exact_match",
        )

        recycled = dict(identity, starttime=identity["starttime"] + 1)
        assert kb.verify_worker_identity(proc.pid, recycled) == (
            False,
            "identity_mismatch",
        )
    finally:
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


@pytest.mark.linux_only
@pytest.mark.live_system_guard_bypass
def test_capture_worker_identity_rejects_non_group_leader():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
    )
    try:
        assert os.getpgid(proc.pid) != proc.pid
        assert kb.capture_worker_identity(proc.pid) is None
    finally:
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


@pytest.mark.linux_only
def test_set_worker_pid_captures_before_writer_txn_and_persists_on_run(
    kanban_home, monkeypatch
):
    assert hasattr(kb, "capture_worker_identity")
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="identity", assignee="worker")
        claimed = kb.claim_task(conn, task_id, claimer="host:worker")
        assert claimed is not None
        observed = {}
        identity = {
            "v": 1,
            "scheme": "linux_proc",
            "pid": 4242,
            "boot_id": "boot",
            "pid_namespace": "pid:[42]",
            "starttime": 99,
            "pgid": 4242,
        }

        def _capture(pid):
            observed["in_transaction"] = conn.in_transaction
            observed["pid"] = pid
            return identity

        monkeypatch.setattr(kb, "capture_worker_identity", _capture)
        kb._set_worker_pid(conn, task_id, 4242)

        row = conn.execute(
            "SELECT worker_pid, worker_identity FROM task_runs WHERE id = ?",
            (claimed.current_run_id,),
        ).fetchone()
        assert observed == {"in_transaction": False, "pid": 4242}
        assert row["worker_pid"] == 4242
        assert json.loads(row["worker_identity"]) == identity


@pytest.mark.linux_only
@pytest.mark.live_system_guard_bypass
def test_native_spawn_persists_group_leader_identity_and_fences_until_exit(
    kanban_home, monkeypatch, tmp_path
):
    helper = tmp_path / "native_worker.py"
    ready = tmp_path / "native-worker-ready"
    helper.write_text(
        "from pathlib import Path\n"
        "import sys, time\n"
        "Path(sys.argv[1]).write_text('ready', encoding='utf-8')\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(
        kb, "_resolve_hermes_argv",
        lambda: [sys.executable, str(helper), str(ready)],
    )
    monkeypatch.setattr(kb, "_resolve_worker_cli_toolsets", lambda _home: None)
    monkeypatch.setattr(kb, "_retag_legacy_worker_sessions", lambda _root: None)

    pid = None
    reaped = False
    try:
        with kb.connect() as conn:
            task_id = kb.create_task(
                conn, title="native spawn", assignee="default",
                workspace_kind="scratch", workspace_path=str(workspace),
            )
            claimed = kb.claim_task(conn, task_id, claimer="host:worker")
            assert claimed is not None and claimed.current_run_id is not None

            pid = kb._default_spawn(claimed, str(workspace))
            assert pid is not None
            deadline = time.time() + 10
            while time.time() < deadline and not ready.exists():
                time.sleep(0.02)
            assert ready.read_text(encoding="utf-8") == "ready"
            assert os.getpgid(pid) == pid

            kb._set_worker_pid(conn, task_id, pid)
            run = conn.execute(
                "SELECT worker_pid, worker_identity FROM task_runs WHERE id = ?",
                (claimed.current_run_id,),
            ).fetchone()
            identity = json.loads(run["worker_identity"])
            assert run["worker_pid"] == pid
            assert identity["pid"] == pid
            assert identity["pgid"] == pid
            assert kb.verify_worker_identity(pid, run["worker_identity"]) == (
                True, "exact_match",
            )

            assert kb.request_review(
                conn, task_id, expected_run_id=claimed.current_run_id,
            )
            before = conn.execute(
                "SELECT status, claim_lock, worker_pid, current_run_id "
                "FROM tasks WHERE id = ?", (task_id,),
            ).fetchone()
            assert tuple(before) == (
                "review", "host:worker", pid, claimed.current_run_id,
            )
            assert kb._release_quiesced_worker_fences(conn) == []
            assert tuple(conn.execute(
                "SELECT status, claim_lock, worker_pid, current_run_id "
                "FROM tasks WHERE id = ?", (task_id,),
            ).fetchone()) == tuple(before)

            os.killpg(pid, signal.SIGTERM)  # windows-footgun: ok — linux_only
            deadline = time.time() + 10
            while time.time() < deadline:
                waited, _status = os.waitpid(pid, os.WNOHANG)
                if waited == pid:
                    reaped = True
                    break
                time.sleep(0.02)
            assert reaped
            assert kb._release_quiesced_worker_fences(conn) == [task_id]
            assert tuple(conn.execute(
                "SELECT claim_lock, worker_pid, current_run_id FROM tasks "
                "WHERE id = ?", (task_id,),
            ).fetchone()) == (None, None, None)
    finally:
        if pid is not None and not reaped:
            try:
                os.killpg(pid, signal.SIGKILL)  # windows-footgun: ok — linux_only
            except ProcessLookupError:
                pass
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass


@pytest.mark.linux_only
@pytest.mark.live_system_guard_bypass
def test_worker_scope_waits_for_real_process_group_descendants():
    assert hasattr(kb, "_worker_scope_is_quiescent")
    code = (
        "import subprocess, sys, time; "
        "child=subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(60)']); "
        "print(child.pid, flush=True); time.sleep(60)"
    )
    leader = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    identity = None
    try:
        assert leader.stdout is not None
        child_pid = int(leader.stdout.readline().strip())
        identity = kb.capture_worker_identity(leader.pid)
        assert identity is not None
        assert not kb._worker_scope_is_quiescent(identity, leader.pid)

        os.kill(leader.pid, signal.SIGTERM)
        leader.wait(timeout=10)
        assert kb._pid_alive(child_pid)
        assert not kb._worker_scope_is_quiescent(identity, leader.pid)

        os.killpg(identity["pgid"], signal.SIGKILL)  # windows-footgun: ok
        deadline = time.time() + 10
        while time.time() < deadline:
            if kb._worker_scope_is_quiescent(identity, leader.pid):
                break
            time.sleep(0.05)
        assert kb._worker_scope_is_quiescent(identity, leader.pid)
    finally:
        if identity is not None:
            try:
                os.killpg(identity["pgid"], signal.SIGKILL)  # windows-footgun: ok
            except ProcessLookupError:
                pass
        try:
            leader.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.kill(leader.pid, signal.SIGKILL)  # windows-footgun: ok
            leader.wait(timeout=10)


@pytest.mark.linux_only
def test_review_handoff_retains_authority_until_group_is_quiescent(
    kanban_home, monkeypatch
):
    assert hasattr(kb, "_release_quiesced_worker_fences")
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="handoff", assignee="builder")
        claimed = kb.claim_task(conn, task_id, claimer="host:builder")
        assert claimed is not None
        _persist_worker_identity(conn, task_id)
        before = conn.execute(
            "SELECT claim_lock, worker_pid FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()

        monkeypatch.setattr(kb, "_worker_scope_is_quiescent", lambda *_args: False)
        assert kb.request_review(
            conn,
            task_id,
            reviewer="reviewer",
            summary="ready",
            expected_run_id=claimed.current_run_id,
        )
        fenced = conn.execute(
            "SELECT status, claim_lock, worker_pid FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        assert tuple(fenced) == ("review", before["claim_lock"], before["worker_pid"])
        assert kb.claim_review_task(conn, task_id) is None
        assert kb._release_quiesced_worker_fences(conn) == []

        monkeypatch.setattr(kb, "_worker_scope_is_quiescent", lambda *_args: True)
        assert kb._release_quiesced_worker_fences(conn) == [task_id]
        released = conn.execute(
            "SELECT claim_lock, worker_pid FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        assert tuple(released) == (None, None)
        assert kb.claim_review_task(conn, task_id) is not None


@pytest.mark.linux_only
def test_request_changes_retains_reviewer_fence_until_quiescent(
    kanban_home, monkeypatch
):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="review rework", assignee="builder")
        implementation = kb.claim_task(conn, task_id, claimer="host:builder")
        assert implementation is not None
        _persist_worker_identity(conn, task_id)
        assert kb.request_review(
            conn,
            task_id,
            reviewer="reviewer",
            summary="ready",
            expected_run_id=implementation.current_run_id,
        )
        monkeypatch.setattr(kb, "_worker_scope_is_quiescent", lambda *_args: True)
        assert kb._release_quiesced_worker_fences(conn) == [task_id]
        review = kb.claim_review_task(conn, task_id, claimer="host:reviewer")
        assert review is not None
        _persist_worker_identity(conn, task_id, pid=4343)
        before = conn.execute(
            "SELECT claim_lock, worker_pid FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()

        monkeypatch.setattr(kb, "_worker_scope_is_quiescent", lambda *_args: False)
        assert kb.request_changes(
            conn,
            task_id,
            reason="fix boundary",
            expected_run_id=review.current_run_id,
        ) == (True, "builder")
        fenced = conn.execute(
            "SELECT status, claim_lock, worker_pid FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        assert tuple(fenced) == ("ready", before["claim_lock"], before["worker_pid"])
        assert kb.claim_task(conn, task_id) is None
        assert kb._release_quiesced_worker_fences(conn) == []

        monkeypatch.setattr(kb, "_worker_scope_is_quiescent", lambda *_args: True)
        assert kb._release_quiesced_worker_fences(conn) == [task_id]
        assert kb.claim_task(conn, task_id) is not None


@pytest.mark.linux_only
def test_block_retains_fence_and_unblock_refuses_until_quiescent(
    kanban_home, monkeypatch
):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="blocked", assignee="worker")
        claimed = kb.claim_task(conn, task_id, claimer="host:worker")
        assert claimed is not None
        _persist_worker_identity(conn, task_id)
        before = conn.execute(
            "SELECT claim_lock, worker_pid FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()

        assert kb.block_task(
            conn,
            task_id,
            reason="needs input",
            expected_run_id=claimed.current_run_id,
        )
        fenced = conn.execute(
            "SELECT status, claim_lock, worker_pid FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        assert tuple(fenced) == ("blocked", before["claim_lock"], before["worker_pid"])
        assert kb.unblock_task(conn, task_id) is False

        monkeypatch.setattr(kb, "_worker_scope_is_quiescent", lambda *_args: True)
        assert kb._release_quiesced_worker_fences(conn) == [task_id]
        assert kb.unblock_task(conn, task_id) is True


@pytest.mark.linux_only
def test_completed_parent_does_not_dispatch_child_before_fence_release(
    kanban_home, monkeypatch
):
    with kb.connect() as conn:
        parent_id = kb.create_task(conn, title="parent", assignee="worker")
        child_id = kb.create_task(
            conn, title="child", assignee="worker", parents=[parent_id],
        )
        claimed = kb.claim_task(conn, parent_id, claimer="host:worker")
        assert claimed is not None
        _persist_worker_identity(conn, parent_id)

        assert kb.complete_task(
            conn,
            parent_id,
            summary="done",
            expected_run_id=claimed.current_run_id,
        )
        parent = conn.execute(
            "SELECT status, claim_lock FROM tasks WHERE id = ?", (parent_id,),
        ).fetchone()
        child = conn.execute(
            "SELECT status FROM tasks WHERE id = ?", (child_id,),
        ).fetchone()
        assert tuple(parent) == ("done", "host:worker")
        assert child["status"] == "todo"

        monkeypatch.setattr(kb, "_worker_scope_is_quiescent", lambda *_args: True)
        assert kb._release_quiesced_worker_fences(conn) == [parent_id]
        assert kb.recompute_ready(conn) == 1
        child = conn.execute(
            "SELECT status FROM tasks WHERE id = ?", (child_id,),
        ).fetchone()
        assert child["status"] == "ready"


@pytest.mark.linux_only
def test_reopen_review_refuses_retained_worker_fence(kanban_home, monkeypatch):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="reopen", assignee="builder")
        claimed = kb.claim_task(conn, task_id, claimer="host:builder")
        assert claimed is not None
        _persist_worker_identity(conn, task_id)
        assert kb.request_review(
            conn,
            task_id,
            reviewer="reviewer",
            expected_run_id=claimed.current_run_id,
        )
        assert kb.reopen_review_task(conn, task_id) is False

        monkeypatch.setattr(kb, "_worker_scope_is_quiescent", lambda *_args: True)
        assert kb._release_quiesced_worker_fences(conn) == [task_id]
        assert kb.reopen_review_task(conn, task_id) is True


@pytest.mark.linux_only
@pytest.mark.live_system_guard_bypass
def test_manual_reclaim_revalidates_and_kills_real_process_group(kanban_home):
    code = (
        "import subprocess, sys, time; "
        "child=subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(60)']); "
        "print(child.pid, flush=True); time.sleep(60)"
    )
    leader = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    identity = None
    try:
        assert leader.stdout is not None
        child_pid = int(leader.stdout.readline().strip())
        identity = kb.capture_worker_identity(leader.pid)
        assert identity is not None
        with kb.connect() as conn:
            task_id = kb.create_task(conn, title="reclaim", assignee="worker")
            claimed = kb.claim_task(conn, task_id, claimer="stale-host:worker")
            assert claimed is not None
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET worker_pid = ? WHERE id = ?",
                    (leader.pid, task_id),
                )
                conn.execute(
                    "UPDATE task_runs SET worker_pid = ?, worker_identity = ? "
                    "WHERE id = ?",
                    (leader.pid, json.dumps(identity), claimed.current_run_id),
                )

            assert kb.reclaim_task(conn, task_id, reason="operator") is True
            task = conn.execute(
                "SELECT status, claim_lock FROM tasks WHERE id = ?", (task_id,),
            ).fetchone()
            assert tuple(task) == ("ready", None)
        leader.wait(timeout=10)
        deadline = time.time() + 10
        while time.time() < deadline and kb._pid_alive(child_pid):
            time.sleep(0.05)
        assert not kb._pid_alive(child_pid)
        assert kb._worker_scope_is_quiescent(identity, leader.pid)
    finally:
        if identity is not None:
            try:
                os.killpg(identity["pgid"], signal.SIGKILL)  # windows-footgun: ok
            except ProcessLookupError:
                pass
        try:
            leader.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.kill(leader.pid, signal.SIGKILL)  # windows-footgun: ok
            leader.wait(timeout=10)


@pytest.mark.linux_only
def test_manual_reclaim_without_exact_identity_fails_closed(
    kanban_home, monkeypatch
):
    signalled = []
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="legacy reclaim", assignee="worker")
        claimed = kb.claim_task(
            conn,
            task_id,
            claimer=f"{kb._claimer_id().split(':', 1)[0]}:worker",
        )
        assert claimed is not None
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET worker_pid = 4242 WHERE id = ?", (task_id,),
            )
            conn.execute(
                "UPDATE task_runs SET worker_pid = 4242, worker_identity = NULL "
                "WHERE id = ?",
                (claimed.current_run_id,),
            )
        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)

        assert kb.reclaim_task(
            conn,
            task_id,
            reason="legacy",
            signal_fn=lambda pid, sig: signalled.append((pid, sig)),
        ) is False
        assert signalled == []
        row = conn.execute(
            "SELECT status, claim_lock, worker_pid FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        assert row["status"] == "running"
        assert row["claim_lock"] is not None
        assert row["worker_pid"] == 4242


@pytest.mark.linux_only
def test_crash_and_orphan_recovery_require_quiescent_group_outside_writer(
    kanban_home, monkeypatch
):
    observations = []

    def _dead(_pid):
        observations.append(conn.in_transaction)
        return False

    with kb.connect() as conn:
        crashed_id = kb.create_task(conn, title="crash", assignee="worker")
        crashed_run = kb.claim_task(conn, crashed_id, claimer="host:worker")
        assert crashed_run is not None
        _persist_worker_identity(conn, crashed_id)
        orphan_id = kb.create_task(conn, title="orphan", assignee="worker")
        orphan_run = kb.claim_task(conn, orphan_id, claimer="host:worker")
        assert orphan_run is not None
        _persist_worker_identity(conn, orphan_id, pid=4343)
        old = int(time.time()) - 1000
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET started_at = ? WHERE id IN (?, ?)",
                (old, crashed_id, orphan_id),
            )
            conn.execute(
                "UPDATE task_runs SET started_at = ? WHERE id IN (?, ?)",
                (old, crashed_run.current_run_id, orphan_run.current_run_id),
            )
            conn.execute(
                "UPDATE tasks SET claim_lock = NULL WHERE id = ?", (orphan_id,),
            )
        monkeypatch.setattr(kb, "_pid_alive", _dead)
        monkeypatch.setattr(kb, "_worker_scope_is_quiescent", lambda *_args: False)

        assert kb.detect_crashed_workers(conn) == []
        assert kb.reconcile_orphaned_running(conn) == []
        for task_id in (crashed_id, orphan_id):
            row = conn.execute(
                "SELECT status FROM tasks WHERE id = ?", (task_id,),
            ).fetchone()
            assert row["status"] == "running"
        assert observations and not any(observations)

        monkeypatch.setattr(kb, "_worker_scope_is_quiescent", lambda *_args: True)
        assert kb.reconcile_orphaned_running(conn) == [orphan_id]
        assert kb.detect_crashed_workers(conn) == [crashed_id]


@pytest.mark.linux_only
def test_crash_probe_cannot_release_replacement_run_with_same_pid_and_claim(
    kanban_home, monkeypatch
):
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    with kb.connect() as detector, kb.connect() as writer:
        task_id = kb.create_task(detector, title="crash race", assignee="worker")
        first = kb.claim_task(detector, task_id, claimer="host:worker")
        assert first is not None
        _persist_worker_identity(detector, task_id)
        old = int(time.time()) - 1_000
        with kb.write_txn(detector):
            detector.execute(
                "UPDATE tasks SET started_at = ? WHERE id = ?", (old, task_id),
            )
            detector.execute(
                "UPDATE task_runs SET started_at = ? WHERE id = ?",
                (old, first.current_run_id),
            )

        replacement = {}

        def _replace_during_probe(_identity, _pid):
            now = int(time.time())
            with kb.write_txn(writer):
                writer.execute(
                    "UPDATE task_runs SET status = 'reclaimed', "
                    "outcome = 'reclaimed', ended_at = ? WHERE id = ?",
                    (now, first.current_run_id),
                )
                writer.execute(
                    "UPDATE tasks SET status = 'ready', claim_lock = NULL, "
                    "claim_expires = NULL, worker_pid = NULL, "
                    "current_run_id = NULL WHERE id = ?",
                    (task_id,),
                )
            claimed = kb.claim_task(writer, task_id, claimer="host:worker")
            assert claimed is not None
            _persist_worker_identity(writer, task_id)
            replacement["run_id"] = claimed.current_run_id
            return True

        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
        monkeypatch.setattr(
            kb, "_worker_scope_is_quiescent", _replace_during_probe,
        )

        assert kb.detect_crashed_workers(detector) == []
        row = detector.execute(
            "SELECT status, claim_lock, worker_pid, current_run_id "
            "FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        assert tuple(row) == (
            "running", "host:worker", 4242, replacement["run_id"],
        )
        assert replacement["run_id"] != first.current_run_id


@pytest.mark.linux_only
@pytest.mark.parametrize("path", ["stale_claim", "manual_reclaim", "max_runtime"])
def test_process_verdict_cannot_release_replacement_run(
    kanban_home, monkeypatch, path
):
    with kb.connect() as detector, kb.connect() as writer:
        task_id = kb.create_task(
            detector,
            title=path,
            assignee="worker",
            max_runtime_seconds=1 if path == "max_runtime" else None,
        )
        first = kb.claim_task(detector, task_id, claimer="host:worker")
        assert first is not None
        _persist_worker_identity(detector, task_id)
        old = int(time.time()) - 1_000
        with kb.write_txn(detector):
            detector.execute(
                "UPDATE tasks SET started_at = ?, claim_expires = ? WHERE id = ?",
                (old, old, task_id),
            )
            detector.execute(
                "UPDATE task_runs SET started_at = ?, claim_expires = ? "
                "WHERE id = ?",
                (old, old, first.current_run_id),
            )

        replacement = {}

        def _replace_authority(*_args, **_kwargs):
            now = int(time.time())
            with kb.write_txn(writer):
                writer.execute(
                    "UPDATE task_runs SET status = 'reclaimed', "
                    "outcome = 'reclaimed', ended_at = ? WHERE id = ?",
                    (now, first.current_run_id),
                )
                writer.execute(
                    "UPDATE tasks SET status = 'ready', claim_lock = NULL, "
                    "claim_expires = NULL, worker_pid = NULL, "
                    "current_run_id = NULL WHERE id = ?",
                    (task_id,),
                )
            claimed = kb.claim_task(writer, task_id, claimer="host:worker")
            assert claimed is not None
            _persist_worker_identity(writer, task_id)
            with kb.write_txn(writer):
                writer.execute(
                    "UPDATE tasks SET started_at = ?, claim_expires = ? "
                    "WHERE id = ?",
                    (old, old, task_id),
                )
                writer.execute(
                    "UPDATE task_runs SET started_at = ?, claim_expires = ? "
                    "WHERE id = ?",
                    (old, old, claimed.current_run_id),
                )
            replacement["run_id"] = claimed.current_run_id
            return {
                "prev_pid": 4242,
                "host_local": True,
                "termination_attempted": True,
                "terminated": True,
                "sigkill": False,
            }

        monkeypatch.setattr(
            kb, "verify_worker_identity", lambda *_args: (False, "gone"),
        )
        monkeypatch.setattr(kb, "_terminate_reclaimed_worker", _replace_authority)

        if path == "stale_claim":
            assert kb.release_stale_claims(detector) == 0
        elif path == "manual_reclaim":
            assert kb.reclaim_task(detector, task_id) is False
        else:
            assert kb.enforce_max_runtime(
                detector, signal_fn=lambda *_args: None,
            ) == []

        row = detector.execute(
            "SELECT status, claim_lock, claim_expires, worker_pid, current_run_id "
            "FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        assert tuple(row) == (
            "running", "host:worker", old, 4242, replacement["run_id"],
        )
        assert replacement["run_id"] != first.current_run_id


@pytest.mark.linux_only
@pytest.mark.parametrize("quiescent", [False, True])
def test_fence_probe_cannot_defer_or_release_replacement_run(
    kanban_home, monkeypatch, quiescent
):
    with kb.connect() as detector, kb.connect() as writer:
        task_id = kb.create_task(detector, title="fence race", assignee="worker")
        first = kb.claim_task(detector, task_id, claimer="host:worker")
        assert first is not None
        identity = _persist_worker_identity(detector, task_id)
        original_expiry = 777
        now = int(time.time())
        with kb.write_txn(detector):
            detector.execute(
                "UPDATE task_runs SET status = 'review_requested', "
                "outcome = 'review_requested', ended_at = ?, "
                "claim_lock = NULL, claim_expires = NULL, worker_pid = NULL "
                "WHERE id = ?",
                (now, first.current_run_id),
            )
            detector.execute(
                "UPDATE tasks SET status = 'review', claim_expires = ? "
                "WHERE id = ?",
                (original_expiry, task_id),
            )

        replacement = {}

        def _replace_during_probe(_identity, _pid):
            with kb.write_txn(writer):
                cur = writer.execute(
                    "INSERT INTO task_runs (task_id, profile, status, outcome, "
                    "worker_identity, started_at, ended_at) "
                    "VALUES (?, 'worker', 'review_requested', "
                    "'review_requested', ?, ?, ?)",
                    (task_id, json.dumps(identity), now, now),
                )
                assert cur.lastrowid is not None
                run_id = int(cur.lastrowid)
                writer.execute(
                    "UPDATE tasks SET current_run_id = ?, claim_expires = ? "
                    "WHERE id = ?",
                    (run_id, original_expiry, task_id),
                )
            replacement["run_id"] = run_id
            return quiescent

        monkeypatch.setattr(
            kb, "_worker_scope_is_quiescent", _replace_during_probe,
        )

        assert kb._release_quiesced_worker_fences(detector) == []
        row = detector.execute(
            "SELECT status, claim_lock, claim_expires, worker_pid, current_run_id "
            "FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        assert tuple(row) == (
            "review", "host:worker", original_expiry, 4242,
            replacement["run_id"],
        )
        assert replacement["run_id"] != first.current_run_id


@pytest.mark.linux_only
@pytest.mark.parametrize("quiescent", [False, True])
def test_fence_probe_cannot_mutate_replaced_identity_on_same_run(
    kanban_home, monkeypatch, quiescent
):
    with kb.connect() as detector, kb.connect() as writer:
        task_id = kb.create_task(detector, title="identity race", assignee="worker")
        claimed = kb.claim_task(detector, task_id, claimer="host:worker")
        assert claimed is not None and claimed.current_run_id is not None
        identity = _persist_worker_identity(detector, task_id)
        assert kb.request_review(
            detector, task_id, expected_run_id=claimed.current_run_id,
        )
        with kb.write_txn(detector):
            detector.execute(
                "UPDATE tasks SET claim_expires = 777 WHERE id = ?", (task_id,),
            )
        before = detector.execute(
            "SELECT status, claim_lock, claim_expires, worker_pid, current_run_id "
            "FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        before_events = detector.execute(
            "SELECT COUNT(*) AS n FROM task_events WHERE task_id = ? "
            "AND kind IN ('reclaim_deferred', 'worker_fence_released')",
            (task_id,),
        ).fetchone()["n"]
        replacement_identity = json.dumps(
            dict(identity, starttime=identity["starttime"] + 1),
            sort_keys=True,
        )

        def _replace_identity(_identity, _pid):
            with kb.write_txn(writer):
                writer.execute(
                    "UPDATE task_runs SET worker_identity = ? WHERE id = ?",
                    (replacement_identity, claimed.current_run_id),
                )
            return quiescent

        monkeypatch.setattr(kb, "_worker_scope_is_quiescent", _replace_identity)

        assert kb._release_quiesced_worker_fences(detector) == []
        assert tuple(detector.execute(
            "SELECT status, claim_lock, claim_expires, worker_pid, current_run_id "
            "FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()) == tuple(before)
        assert detector.execute(
            "SELECT worker_identity FROM task_runs WHERE id = ?",
            (claimed.current_run_id,),
        ).fetchone()["worker_identity"] == replacement_identity
        assert detector.execute(
            "SELECT COUNT(*) AS n FROM task_events WHERE task_id = ? "
            "AND kind IN ('reclaim_deferred', 'worker_fence_released')",
            (task_id,),
        ).fetchone()["n"] == before_events


@pytest.mark.linux_only
@pytest.mark.parametrize("transition", ["review", "block", "complete"])
def test_worker_transition_with_legacy_incomplete_claim_fails_closed(
    kanban_home, transition
):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title=transition, assignee="worker")
        claimed = kb.claim_task(conn, task_id, claimer="legacy:worker")
        assert claimed is not None
        if transition == "review":
            result = kb.request_review(
                conn, task_id, expected_run_id=claimed.current_run_id,
            )
        elif transition == "block":
            result = kb.block_task(
                conn, task_id, expected_run_id=claimed.current_run_id,
            )
        else:
            result = kb.complete_task(
                conn, task_id, expected_run_id=claimed.current_run_id,
            )
        assert result is False
        row = conn.execute(
            "SELECT status, claim_lock FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        assert row["status"] == "running"
        assert row["claim_lock"] == "legacy:worker"


@pytest.mark.linux_only
def test_dashboard_cannot_force_release_or_invalidate_live_descendant(
    kanban_home
):
    from plugins.kanban.dashboard import plugin_api

    with kb.connect() as conn:
        direct_id = kb.create_task(conn, title="direct", assignee="worker")
        direct_run = kb.claim_task(conn, direct_id, claimer="host:worker")
        assert direct_run is not None
        _persist_worker_identity(conn, direct_id)
        assert plugin_api._set_status_direct(conn, direct_id, "ready") is False
        direct = conn.execute(
            "SELECT status, claim_lock FROM tasks WHERE id = ?", (direct_id,),
        ).fetchone()
        assert direct["status"] == "running"
        assert direct["claim_lock"] is not None

        parent_id = kb.create_task(conn, title="parent", assignee="planner")
        assert kb.complete_task(conn, parent_id)
        child_id = kb.create_task(
            conn, title="child", assignee="worker", parents=[parent_id],
        )
        child_run = kb.claim_task(conn, child_id, claimer="host:worker")
        assert child_run is not None
        _persist_worker_identity(conn, child_id, pid=4343)

        assert plugin_api._set_status_direct(conn, parent_id, "ready") is False
        parent = conn.execute(
            "SELECT status FROM tasks WHERE id = ?", (parent_id,),
        ).fetchone()
        child = conn.execute(
            "SELECT status, claim_lock FROM tasks WHERE id = ?", (child_id,),
        ).fetchone()
        assert parent["status"] == "done"
        assert child["status"] == "running"
        assert child["claim_lock"] is not None


@pytest.mark.linux_only
@pytest.mark.parametrize("fenced_status", ["review", "done"])
def test_dashboard_direct_status_preserves_nonrunning_worker_fence(
    kanban_home, monkeypatch, fenced_status
):
    from plugins.kanban.dashboard import plugin_api

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title=fenced_status, assignee="worker")
        claimed = kb.claim_task(conn, task_id, claimer="host:worker")
        assert claimed is not None
        _persist_worker_identity(conn, task_id)
        monkeypatch.setattr(kb, "_worker_scope_is_quiescent", lambda *_args: False)
        if fenced_status == "review":
            assert kb.request_review(
                conn, task_id, expected_run_id=claimed.current_run_id,
            )
        else:
            assert kb.complete_task(
                conn, task_id, expected_run_id=claimed.current_run_id,
            )
        before = conn.execute(
            "SELECT status, claim_lock, worker_pid, current_run_id FROM tasks "
            "WHERE id = ?",
            (task_id,),
        ).fetchone()
        assert before["status"] == fenced_status

        assert plugin_api._set_status_direct(conn, task_id, "todo") is False
        assert tuple(conn.execute(
            "SELECT status, claim_lock, worker_pid, current_run_id FROM tasks "
            "WHERE id = ?",
            (task_id,),
        ).fetchone()) == tuple(before)

        assert plugin_api._set_status_direct(
            conn, task_id, fenced_status,
        ) is True
        assert tuple(conn.execute(
            "SELECT status, claim_lock, worker_pid, current_run_id FROM tasks "
            "WHERE id = ?",
            (task_id,),
        ).fetchone()) == tuple(before)


@pytest.mark.linux_only
def test_dashboard_ready_uses_parent_fence_contract(kanban_home, monkeypatch):
    from plugins.kanban.dashboard import plugin_api

    with kb.connect() as conn:
        parent_id = kb.create_task(conn, title="parent fence", assignee="worker")
        parent_run = kb.claim_task(conn, parent_id, claimer="host:worker")
        assert parent_run is not None
        _persist_worker_identity(conn, parent_id)
        child_id = kb.create_task(
            conn, title="child", assignee="worker", parents=[parent_id],
        )
        monkeypatch.setattr(kb, "_worker_scope_is_quiescent", lambda *_args: False)
        assert kb.complete_task(
            conn, parent_id, expected_run_id=parent_run.current_run_id,
        )
        parent = conn.execute(
            "SELECT status, claim_lock FROM tasks WHERE id = ?", (parent_id,),
        ).fetchone()
        assert tuple(parent) == ("done", "host:worker")

        assert plugin_api._set_status_direct(conn, child_id, "ready") is False
        child = conn.execute(
            "SELECT status, claim_lock FROM tasks WHERE id = ?", (child_id,),
        ).fetchone()
        assert tuple(child) == ("todo", None)


def _make_malformed_authority(
    conn, *, status: str, authority_field: str, title: str,
) -> str:
    task_id = kb.create_task(conn, title=title, assignee="worker")
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = ?, claim_lock = NULL, worker_pid = NULL, "
            "current_run_id = NULL WHERE id = ?",
            (status, task_id),
        )
        if authority_field == "claim_lock":
            conn.execute(
                "UPDATE tasks SET claim_lock = 'malformed-lock' WHERE id = ?",
                (task_id,),
            )
        elif authority_field == "worker_pid":
            conn.execute(
                "UPDATE tasks SET worker_pid = 4242 WHERE id = ?", (task_id,),
            )
        else:
            cur = conn.execute(
                "INSERT INTO task_runs (task_id, status, started_at) "
                "VALUES (?, 'running', ?)",
                (task_id, int(time.time())),
            )
            assert cur.lastrowid is not None
            conn.execute(
                "UPDATE tasks SET current_run_id = ? WHERE id = ?",
                (int(cur.lastrowid), task_id),
            )
    return task_id


def _authority_state(conn, task_id: str) -> tuple:
    task = conn.execute(
        "SELECT status, claim_lock, worker_pid, current_run_id FROM tasks "
        "WHERE id = ?",
        (task_id,),
    ).fetchone()
    runs = conn.execute(
        "SELECT id, status, outcome, ended_at FROM task_runs "
        "WHERE task_id = ? ORDER BY id",
        (task_id,),
    ).fetchall()
    return tuple(task), tuple(tuple(run) for run in runs)


@pytest.mark.linux_only
@pytest.mark.parametrize(
    ("authority_field", "expected_probe"),
    [
        ("claim_lock", (None, None)),
        ("worker_pid", (4242, None)),
        ("current_run_id", (None, "current-run-only")),
    ],
)
def test_fence_reconciliation_selects_partial_authority_and_fails_closed(
    kanban_home, monkeypatch, authority_field, expected_probe
):
    with kb.connect() as conn:
        task_id = _make_malformed_authority(
            conn, status="review", authority_field=authority_field,
            title=f"reconcile-{authority_field}",
        )
        if authority_field == "current_run_id":
            run_id = conn.execute(
                "SELECT current_run_id FROM tasks WHERE id = ?", (task_id,),
            ).fetchone()["current_run_id"]
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE task_runs SET worker_identity = ? WHERE id = ?",
                    ("current-run-only", run_id),
                )
        before = _authority_state(conn, task_id)
        before_events = conn.execute(
            "SELECT COUNT(*) AS n FROM task_events WHERE task_id = ?",
            (task_id,),
        ).fetchone()["n"]
        probes = []

        def _reject_partial(pid, identity):
            probes.append((pid, identity))
            return False

        monkeypatch.setattr(kb, "_worker_identity_can_fence", _reject_partial)

        assert kb._release_quiesced_worker_fences(conn) == []
        assert probes == [expected_probe]
        assert _authority_state(conn, task_id) == before
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM task_events WHERE task_id = ?",
            (task_id,),
        ).fetchone()["n"] == before_events


@pytest.mark.linux_only
@pytest.mark.parametrize(
    "authority_field", ["claim_lock", "worker_pid", "current_run_id"],
)
def test_partial_authority_tuple_blocks_all_mutation_paths(
    kanban_home, authority_field
):
    with kb.connect() as conn:
        ready = _make_malformed_authority(
            conn, status="ready", authority_field=authority_field,
            title=f"claim-{authority_field}",
        )
        before = _authority_state(conn, ready)
        assert kb.claim_task(conn, ready) is None
        assert _authority_state(conn, ready) == before

        review = _make_malformed_authority(
            conn, status="review", authority_field=authority_field,
            title=f"review-{authority_field}",
        )
        before = _authority_state(conn, review)
        assert kb.claim_review_task(conn, review) is None
        assert kb.request_review(conn, review) is False
        assert _authority_state(conn, review) == before

        blocked = _make_malformed_authority(
            conn, status="blocked", authority_field=authority_field,
            title=f"unblock-{authority_field}",
        )
        before = _authority_state(conn, blocked)
        assert kb.unblock_task(conn, blocked) is False
        assert _authority_state(conn, blocked) == before

        reopened = _make_malformed_authority(
            conn, status="review", authority_field=authority_field,
            title=f"reopen-{authority_field}",
        )
        before = _authority_state(conn, reopened)
        assert kb.reopen_review_task(conn, reopened) is False
        assert _authority_state(conn, reopened) == before

        promoted = _make_malformed_authority(
            conn, status="todo", authority_field=authority_field,
            title=f"promote-{authority_field}",
        )
        before = _authority_state(conn, promoted)
        assert kb.promote_task(conn, promoted, actor="test")[0] is False
        assert kb.recompute_ready(conn) == 0
        assert _authority_state(conn, promoted) == before

        parent = _make_malformed_authority(
            conn, status="done", authority_field=authority_field,
            title=f"parent-{authority_field}",
        )
        child = kb.create_task(
            conn, title=f"child-{authority_field}", parents=[parent],
        )
        child_created = kb.get_task(conn, child)
        assert child_created is not None and child_created.status == "todo"
        before = _authority_state(conn, parent)
        assert kb._parents_satisfied(conn, child) is False
        assert kb._landing_status_after_parents(conn, child) == "todo"
        assert kb.promote_task(
            conn, child, actor="test", force=True,
        )[0] is False
        assert kb.recompute_ready(conn) == 0
        child_task = kb.get_task(conn, child)
        assert child_task is not None and child_task.status == "todo"
        assert _authority_state(conn, parent) == before

        from plugins.kanban.dashboard import plugin_api

        ancestor = kb.create_task(conn, title=f"ancestor-{authority_field}")
        assert kb.complete_task(conn, ancestor)
        descendant = _make_malformed_authority(
            conn, status="done", authority_field=authority_field,
            title=f"descendant-{authority_field}",
        )
        kb.link_tasks(conn, ancestor, descendant)
        ancestor_before = _authority_state(conn, ancestor)
        descendant_before = _authority_state(conn, descendant)
        assert plugin_api._set_status_direct(conn, ancestor, "ready") is False
        assert _authority_state(conn, ancestor) == ancestor_before
        assert _authority_state(conn, descendant) == descendant_before
