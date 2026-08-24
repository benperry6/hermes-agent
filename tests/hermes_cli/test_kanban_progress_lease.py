"""Progress lease vs liveness lease on the Kanban board (commit A: detection).

A dispatcher-spawned worker could hold its claim forever while doing nothing:
``AIAgent._touch_activity`` fires on every stream chunk and API wait, and the
runtime bridge renews ``tasks.last_heartbeat_at`` + ``claim_expires`` from it.
Every watchdog keyed off that one clock, so a worker reasoning in a loop with
zero tool calls and zero board transitions renewed its own lease indefinitely.

The fix splits the claim into two independent leases:

* **liveness** (``last_heartbeat_at`` + ``claim_expires``) — "the process and
  its provider are responsive". Renewed by any agent activity, stream tokens
  included. Unchanged.
* **progress** (``tasks.last_progress_at``, one authoritative column) —
  "something observable happened outside the model's token stream". Stamped
  by the claim, by a *verified* tool execution seen by the tool middleware,
  and by durable board transitions (``PROGRESS_EVENT_KINDS``). Never by
  model-authored text, ``kanban_heartbeat`` notes included.

This commit only DETECTS an expired progress lease. ``detect_no_progress_running``
records a durable ``no_progress_deferred`` receipt, holds the claim for a
bounded grace and leaves status / claim / PID / run / failure accounting
exactly as they were. It never signals, requeues or counts a failure: the
authority to do that is a separate, later change. These tests pin that
boundary as hard as they pin the lease split.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb

pytestmark = pytest.mark.usefixtures("synthetic_kanban_worker_lifecycle")


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_CLAIM_LOCK", raising=False)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _host() -> str:
    return kb._claimer_id().split(":", 1)[0]


def _running_task(conn, *, title="t", assignee="a", pid=4242, claimer=None):
    """Create + claim a task on this host, with a worker pid recorded."""
    tid = kb.create_task(conn, title=title, assignee=assignee)
    kb.claim_task(conn, tid, claimer=claimer or f"{_host()}:worker")
    if pid is not None:
        kb._set_worker_pid(conn, tid, pid)
    return tid


def _cols(conn, table):
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}


def _row(conn, tid):
    return conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone()


def _run_row(conn, tid):
    return conn.execute(
        "SELECT r.* FROM task_runs r JOIN tasks t ON t.current_run_id = r.id "
        "WHERE t.id = ?",
        (tid,),
    ).fetchone()


def _events(conn, tid, kind):
    return [
        json.loads(r["payload"]) if r["payload"] else {}
        for r in conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? "
            "ORDER BY id",
            (tid, kind),
        )
    ]


def _age_progress(conn, tid, seconds, *, heartbeat_age=0):
    """Fresh liveness, stale progress: the reasoning-loop shape."""
    now = int(time.time())
    conn.execute(
        "UPDATE tasks SET last_heartbeat_at = ?, last_progress_at = ?, "
        "claim_expires = ? WHERE id = ?",
        (now - heartbeat_age, now - seconds, now + 900, tid),
    )
    conn.commit()


def _record_owned_progress(conn, tid):
    row = _row(conn, tid)
    return kb.record_progress(
        conn,
        tid,
        source=kb.PROGRESS_SOURCE_TOOL,
        expected_run_id=row["current_run_id"],
        claim_lock=row["claim_lock"],
    )


def _forbid_signals(monkeypatch):
    """Commit A must never reach for a process. Any attempt is a test failure."""
    def _boom(*_a, **_kw):
        raise AssertionError("no-progress detection must never signal a worker")

    monkeypatch.setattr(kb, "_terminate_reclaimed_worker", _boom)
    monkeypatch.setattr(kb.os, "kill", _boom, raising=False)
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)


def _stub_operator_reclaim(monkeypatch):
    """``reclaim_task`` is an operator action that legitimately signals; in
    tests it must never touch a real pid, so its termination helper is a
    no-op that reports a non-local worker."""
    monkeypatch.setattr(
        kb, "_terminate_reclaimed_worker",
        lambda *a, **kw: {"prev_pid": None, "host_local": True,
                          "termination_attempted": True, "terminated": True,
                          "sigkill": False},
    )


def _snapshot(conn, tid):
    row = _row(conn, tid)
    run = _run_row(conn, tid)
    return {
        "status": row["status"],
        "claim_lock": row["claim_lock"],
        "worker_pid": row["worker_pid"],
        "current_run_id": row["current_run_id"],
        "consecutive_failures": row["consecutive_failures"],
        "last_progress_at": row["last_progress_at"],
        "run_status": run["status"],
        "run_outcome": run["outcome"],
        "run_ended_at": run["ended_at"],
    }


# ---------------------------------------------------------------------------
# Schema + migration + dataclass
# ---------------------------------------------------------------------------

def test_fresh_schema_carries_one_authoritative_progress_column(kanban_home):
    """``tasks.last_progress_at`` is the single source of truth; the run
    history table is deliberately not widened for it."""
    with kb.connect() as conn:
        assert "last_progress_at" in _cols(conn, "tasks")
        assert "last_progress_at" not in _cols(conn, "task_runs")


def test_migration_adds_the_progress_lease_and_preserves_rows(kanban_home):
    """A board that predates the lease migrates in place: additive, rows and
    the in-flight claim preserved, legacy rows NULL (never backfilled to
    'now', which would gift a wedged pre-upgrade worker a lease it never
    earned)."""
    with kb.connect() as conn:
        tid = _running_task(conn)
        other = kb.create_task(conn, title="queued", assignee="a")
        conn.execute("ALTER TABLE tasks DROP COLUMN last_progress_at")
        conn.commit()
        assert "last_progress_at" not in _cols(conn, "tasks")

        kb._migrate_add_optional_columns(conn)

        assert "last_progress_at" in _cols(conn, "tasks")
        assert _row(conn, tid)["status"] == "running"
        assert _row(conn, tid)["claim_lock"] == f"{_host()}:worker"
        assert _row(conn, tid)["last_progress_at"] is None
        assert _row(conn, other)["status"] in {"ready", "todo"}
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 2


def test_migration_is_idempotent(kanban_home):
    with kb.connect() as conn:
        kb._migrate_add_optional_columns(conn)
        kb._migrate_add_optional_columns(conn)
        assert "last_progress_at" in _cols(conn, "tasks")


def test_reopening_a_legacy_board_migrates_it(kanban_home):
    """Restart path: ``connect()`` on an un-migrated board adds the column."""
    with kb.connect() as conn:
        tid = _running_task(conn)
        conn.execute("ALTER TABLE tasks DROP COLUMN last_progress_at")
        conn.commit()
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kb.connect() as conn:
        assert "last_progress_at" in _cols(conn, "tasks")
        assert kb.get_task(conn, tid).last_progress_at is None


def test_task_dataclass_exposes_the_progress_lease(kanban_home):
    """``asdict(Task)`` is what the dashboard and CLI serialise — the field
    has to be on the dataclass or no surface can ever show it. The run
    dataclass stays as it was: one authoritative column."""
    with kb.connect() as conn:
        tid = _running_task(conn)
        task = kb.get_task(conn, tid)
        assert task.last_progress_at is not None
        run = kb.get_run(conn, task.current_run_id)
        assert not hasattr(run, "last_progress_at")


def test_legacy_row_reads_as_none(kanban_home):
    with kb.connect() as conn:
        tid = _running_task(conn)
        conn.execute(
            "UPDATE tasks SET last_progress_at = NULL WHERE id = ?", (tid,)
        )
        conn.commit()
        assert kb.get_task(conn, tid).last_progress_at is None


# ---------------------------------------------------------------------------
# Claim stamps both leases
# ---------------------------------------------------------------------------

def test_claim_stamps_liveness_and_progress(kanban_home):
    """The claim is itself a transition: the attempt starts with a fresh
    progress lease (and, as before, a fresh claim TTL)."""
    before = int(time.time())
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        assert _row(conn, tid)["last_progress_at"] is None
        kb.claim_task(conn, tid, claimer=f"{_host()}:worker")
        row = _row(conn, tid)
        assert row["last_progress_at"] >= before
        assert row["claim_expires"] > before


def test_review_claim_stamps_the_progress_lease(kanban_home):
    before = int(time.time())
    with kb.connect() as conn:
        tid = _running_task(conn)
        assert kb.request_review(
            conn, tid, summary="done, please review",
            expected_run_id=_row(conn, tid)["current_run_id"],
        )
        conn.execute(
            "UPDATE tasks SET last_progress_at = NULL WHERE id = ?", (tid,)
        )
        conn.commit()
        assert kb.get_task(conn, tid).status == "review"
        claimed = kb.claim_review_task(conn, tid, claimer=f"{_host()}:reviewer")
        assert claimed is not None and claimed.status == "running"
        assert _row(conn, tid)["last_progress_at"] >= before


def test_reclaim_then_reclaim_starts_a_fresh_lease(kanban_home, monkeypatch):
    """A second attempt is measured from its own claim, not the previous
    attempt's last action."""
    _stub_operator_reclaim(monkeypatch)
    with kb.connect() as conn:
        tid = _running_task(conn)
        _age_progress(conn, tid, 9000)
        assert kb.reclaim_task(conn, tid, reason="operator") is True
        before = int(time.time())
        kb.claim_task(conn, tid, claimer=f"{_host()}:worker")
        assert _row(conn, tid)["last_progress_at"] >= before


# ---------------------------------------------------------------------------
# Liveness never renews progress
# ---------------------------------------------------------------------------

def test_stream_activity_renews_liveness_but_not_progress(kanban_home):
    """THE regression. ``heartbeat_claim`` / ``heartbeat_worker`` are the
    liveness bridge that ``_touch_activity`` drives on every stream chunk.
    They must move ``last_heartbeat_at`` and ``claim_expires`` — and leave
    ``last_progress_at`` exactly where it was."""
    with kb.connect() as conn:
        tid = _running_task(conn)
        stale = int(time.time()) - 9000
        conn.execute(
            "UPDATE tasks SET last_progress_at = ?, last_heartbeat_at = ?, "
            "claim_expires = ? WHERE id = ?",
            (stale, stale, stale, tid),
        )
        conn.commit()

        for _ in range(5):  # thousands of tokens, no tools
            assert kb.heartbeat_claim(conn, tid, claimer=f"{_host()}:worker")
            assert kb.heartbeat_worker(conn, tid)

        row = _row(conn, tid)
        assert row["last_heartbeat_at"] > stale, "liveness must renew"
        assert row["claim_expires"] > stale, "claim TTL must renew"
        assert row["last_progress_at"] == stale, (
            "raw stream/API activity must NOT renew the progress lease"
        )


def test_heartbeat_notes_never_renew_progress(kanban_home):
    """The note is free text the model writes. A/B/A/B defeats a
    previous-note dedupe; twenty unique notes defeat any 'is it new?' test.
    Neither may move the lease — the boundary is the class of the signal,
    not its content."""
    with kb.connect() as conn:
        tid = _running_task(conn)
        stale = int(time.time()) - 9000
        conn.execute(
            "UPDATE tasks SET last_progress_at = ? WHERE id = ?", (stale, tid)
        )
        conn.commit()
        notes = [("compiling A" if i % 2 == 0 else "compiling B") for i in range(8)]
        notes += [f"step {i}: analysed subsystem {i}, chose approach {i}" for i in range(20)]
        for note in notes:
            assert kb.heartbeat_worker(conn, tid, note=note)
        assert _row(conn, tid)["last_progress_at"] == stale
        # ...and the commentary is still recorded for humans.
        assert len(_events(conn, tid, "heartbeat")) == len(notes)


def test_lease_bookkeeping_events_do_not_renew_progress(kanban_home):
    """``heartbeat`` / ``claim_extended`` / ``claimed`` / recovery receipts are
    bookkeeping, not work. If they renewed progress the guard would renew
    itself."""
    with kb.connect() as conn:
        tid = _running_task(conn)
        stale = int(time.time()) - 9000
        conn.execute(
            "UPDATE tasks SET last_progress_at = ? WHERE id = ?", (stale, tid)
        )
        conn.commit()
        with kb.write_txn(conn):
            for kind in (
                "heartbeat", "claim_extended", "claimed", "spawned",
                "reclaim_deferred", "no_progress_deferred", "stale",
                "reclaimed", "timed_out", "crashed", "respawn_guarded",
            ):
                kb._append_event(conn, tid, kind, {"reason": "test"})
        assert _row(conn, tid)["last_progress_at"] == stale


# ---------------------------------------------------------------------------
# Durable board transitions and verified tool execution renew progress
# ---------------------------------------------------------------------------

def test_progress_event_kinds_is_the_transition_allowlist():
    """Pinned so a bookkeeping kind cannot drift in and a transition cannot
    silently drop out."""
    assert kb.PROGRESS_EVENT_KINDS == frozenset({
        "completed", "review_requested", "changes_requested",
        "review_reopened", "blocked", "unblocked", "commented", "attached",
        "attachment_removed", "linked", "unlinked", "edited", "decomposed",
        "specified",
    })
    for bookkeeping in (
        "heartbeat", "claimed", "claim_extended", "spawned", "promoted",
        "reclaimed", "reclaim_deferred", "no_progress_deferred", "stale",
        "timed_out", "crashed", "gave_up", "spawn_failed", "status",
    ):
        assert bookkeeping not in kb.PROGRESS_EVENT_KINDS, bookkeeping


def test_board_transition_renews_progress(kanban_home):
    """A durable board transition is progress evidence in its own right,
    whichever surface produced it (worker tool, CLI, dashboard)."""
    with kb.connect() as conn:
        tid = _running_task(conn)
        stale = int(time.time()) - 9000
        conn.execute(
            "UPDATE tasks SET last_progress_at = ? WHERE id = ?", (stale, tid)
        )
        conn.commit()
        kb.add_comment(conn, tid, author="worker", body="hotspot: a.py — churn")
        assert _row(conn, tid)["last_progress_at"] > stale

        conn.execute(
            "UPDATE tasks SET last_progress_at = ? WHERE id = ?", (stale, tid)
        )
        conn.commit()
        # ``linked`` is recorded on the child side of the edge.
        parent = kb.create_task(conn, title="parent", assignee="a")
        kb.link_tasks(conn, parent, tid)
        assert _row(conn, tid)["last_progress_at"] > stale


def test_verified_tool_execution_renews_progress(kanban_home):
    with kb.connect() as conn:
        tid = _running_task(conn)
        stale = int(time.time()) - 9000
        conn.execute(
            "UPDATE tasks SET last_progress_at = ? WHERE id = ?", (stale, tid)
        )
        conn.commit()
        run_id = _row(conn, tid)["current_run_id"]

        receipt = kb.record_progress(
            conn, tid, source=kb.PROGRESS_SOURCE_TOOL,
            expected_run_id=run_id, claim_lock=f"{_host()}:worker",
        )

        assert receipt["recorded"] is True
        assert receipt["reason"] == "recorded"
        assert receipt["run_id"] == run_id
        assert _row(conn, tid)["last_progress_at"] > stale
        assert _row(conn, tid)["last_progress_at"] == receipt["last_progress_at"]


def test_record_progress_admits_only_the_verified_tool_source(kanban_home):
    """Enforced inside ``record_progress``, not at its call sites, so a future
    caller that invents a model-driven source gets a refusal rather than a
    lease renewal. Unhashable sources are refused, never raised."""
    assert kb.PROGRESS_RENEWAL_SOURCES == frozenset({kb.PROGRESS_SOURCE_TOOL})
    with kb.connect() as conn:
        tid = _running_task(conn)
        stale = int(time.time()) - 9000
        conn.execute(
            "UPDATE tasks SET last_progress_at = ? WHERE id = ?", (stale, tid)
        )
        conn.commit()
        for bogus in ("checkpoint", "heartbeat", "note", "model", "", None, ["tool"]):
            receipt = kb.record_progress(conn, tid, source=bogus)
            assert receipt["recorded"] is False, bogus
            assert receipt["reason"] == "unsupported_source", bogus
        assert _row(conn, tid)["last_progress_at"] == stale


def test_record_progress_is_bound_to_task_run_and_claim(kanban_home):
    """A worker whose run was reclaimed, or whose claim was taken over, must
    not resurrect the lease of whatever replaced it."""
    with kb.connect() as conn:
        tid = _running_task(conn)
        stale = int(time.time()) - 9000
        conn.execute(
            "UPDATE tasks SET last_progress_at = ? WHERE id = ?", (stale, tid)
        )
        conn.commit()
        run_id = _row(conn, tid)["current_run_id"]
        lock = f"{_host()}:worker"

        wrong_run = kb.record_progress(
            conn, tid, source=kb.PROGRESS_SOURCE_TOOL,
            expected_run_id=run_id + 1000, claim_lock=lock,
        )
        assert wrong_run["recorded"] is False
        assert wrong_run["reason"] == "superseded"

        wrong_claim = kb.record_progress(
            conn, tid, source=kb.PROGRESS_SOURCE_TOOL,
            expected_run_id=run_id, claim_lock=f"{_host()}:someone-else",
        )
        assert wrong_claim["recorded"] is False
        assert wrong_claim["reason"] == "claim_mismatch"

        missing = kb.record_progress(
            conn, "t_does_not_exist", source=kb.PROGRESS_SOURCE_TOOL,
        )
        assert missing["recorded"] is False
        assert missing["reason"] == "not_running"

        queued = kb.create_task(conn, title="queued", assignee="a")
        parked = kb.record_progress(conn, queued, source=kb.PROGRESS_SOURCE_TOOL)
        assert parked["recorded"] is False
        assert parked["reason"] == "not_running"

        assert _row(conn, tid)["last_progress_at"] == stale


def test_tool_progress_requires_the_complete_worker_authority_tuple(kanban_home):
    with kb.connect() as conn:
        tid = _running_task(conn)
        stale = int(time.time()) - 9000
        conn.execute(
            "UPDATE tasks SET last_progress_at = ? WHERE id = ?", (stale, tid)
        )
        conn.commit()
        row = _row(conn, tid)

        for kwargs in (
            {},
            {"expected_run_id": row["current_run_id"]},
            {"claim_lock": row["claim_lock"]},
        ):
            receipt = kb.record_progress(
                conn, tid, source=kb.PROGRESS_SOURCE_TOOL, **kwargs,
            )
            assert receipt["recorded"] is False
            assert receipt["reason"] == "authority_required"
            assert _row(conn, tid)["last_progress_at"] == stale


def test_progress_never_writes_an_event_row(kanban_home):
    """Progress is a column stamp only — a tool call every few seconds for
    hours must not bloat ``task_events``, and no ``progress`` event exists
    to be mined for 'the worker said it did something'."""
    with kb.connect() as conn:
        tid = _running_task(conn)
        before = conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0]
        for _ in range(5):
            _record_owned_progress(conn, tid)
        after = conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0]
        assert after == before
        assert _events(conn, tid, "progress") == []


# ---------------------------------------------------------------------------
# Detection: expiry is deferred, never settled, in this commit
# ---------------------------------------------------------------------------

def test_expired_progress_lease_is_deferred_not_reclaimed(kanban_home, monkeypatch):
    """The reproduced incident at the DB layer: PID alive, heartbeat seconds
    old, progress lease hours old. The detector reports it with a durable
    receipt and holds the claim — and changes nothing else."""
    _forbid_signals(monkeypatch)
    with kb.connect() as conn:
        tid = _running_task(conn)
        _age_progress(conn, tid, 7200)
        before = _snapshot(conn, tid)
        now = int(time.time())

        deferred = kb.detect_no_progress_running(
            conn, no_progress_timeout_seconds=2700,
        )

        assert deferred == [tid]
        after = _snapshot(conn, tid)
        assert after == before, "status/claim/pid/run/failure count must be untouched"
        assert after["status"] == "running"
        assert after["consecutive_failures"] == 0
        assert after["run_outcome"] is None and after["run_ended_at"] is None

        receipts = _events(conn, tid, "no_progress_deferred")
        assert len(receipts) == 1
        payload = receipts[0]
        assert payload["classification"] == "no_progress"
        assert payload["reason"] == "authority_unavailable"
        assert payload["progress_age_seconds"] >= 7200
        assert payload["timeout_seconds"] == 2700
        assert payload["heartbeat_age_seconds"] <= 60
        assert payload["liveness"] == "fresh"
        assert payload["worker_pid"] == 4242
        assert payload["claim_lock"] == f"{_host()}:worker"
        assert payload["host_local"] is True
        assert payload["grace_seconds"] == kb.NO_PROGRESS_DEFER_GRACE_SECONDS
        assert payload["claim_expires_now"] >= now + kb.NO_PROGRESS_DEFER_GRACE_SECONDS

        # The claim is held for the grace window, on the task AND its run.
        assert _row(conn, tid)["claim_expires"] >= now + kb.NO_PROGRESS_DEFER_GRACE_SECONDS
        assert _run_row(conn, tid)["claim_expires"] >= now + kb.NO_PROGRESS_DEFER_GRACE_SECONDS

        # Nothing that belongs to settlement happened.
        assert _events(conn, tid, "no_progress") == []
        assert _events(conn, tid, "reclaimed") == []
        assert _events(conn, tid, "reclaim_deferred") == []
        assert kb.list_comments(conn, tid) == []
        run = conn.execute(
            "SELECT status, outcome FROM task_runs WHERE task_id = ?", (tid,)
        ).fetchall()
        assert [(r["status"], r["outcome"]) for r in run] == [("running", None)]


def test_progress_recorded_after_snapshot_prevents_stale_receipt(
    kanban_home, monkeypatch,
):
    """A second writer can renew progress from the liveness probe window.
    The detector must revalidate that evidence after acquiring its writer lock."""
    with kb.connect() as detector_conn, kb.connect() as progress_conn:
        tid = _running_task(detector_conn)
        _age_progress(detector_conn, tid, 7200)
        row = _row(detector_conn, tid)
        old_progress_at = row["last_progress_at"]
        progress_receipts = []

        def _record_progress_then_report_alive(_pid):
            progress_receipts.append(kb.record_progress(
                progress_conn,
                tid,
                source=kb.PROGRESS_SOURCE_TOOL,
                expected_run_id=row["current_run_id"],
                claim_lock=row["claim_lock"],
            ))
            return True

        monkeypatch.setattr(kb, "_pid_alive", _record_progress_then_report_alive)
        deferred = kb.detect_no_progress_running(
            detector_conn, no_progress_timeout_seconds=60,
        )
        receipts = _events(detector_conn, tid, "no_progress_deferred")

        assert progress_receipts == [{
            "recorded": True,
            "reason": "recorded",
            "source": kb.PROGRESS_SOURCE_TOOL,
            "run_id": row["current_run_id"],
            "last_progress_at": _row(detector_conn, tid)["last_progress_at"],
        }]
        assert _row(detector_conn, tid)["last_progress_at"] > old_progress_at
        assert (deferred, len(receipts)) == ([], 0)


def test_board_transition_after_snapshot_prevents_stale_receipt(
    kanban_home, monkeypatch,
):
    """A durable non-terminal board transition in the same window is fresh
    progress and must defeat the detector's stale snapshot."""
    with kb.connect() as detector_conn, kb.connect() as board_conn:
        tid = _running_task(detector_conn)
        _age_progress(detector_conn, tid, 7200)
        old_progress_at = _row(detector_conn, tid)["last_progress_at"]
        comment_ids = []

        def _comment_then_report_alive(_pid):
            comment_ids.append(kb.add_comment(
                board_conn, tid, author="operator", body="durable progress",
            ))
            return True

        monkeypatch.setattr(kb, "_pid_alive", _comment_then_report_alive)
        deferred = kb.detect_no_progress_running(
            detector_conn, no_progress_timeout_seconds=60,
        )
        receipts = _events(detector_conn, tid, "no_progress_deferred")

        assert len(comment_ids) == 1
        assert [comment.body for comment in kb.list_comments(detector_conn, tid)] == [
            "durable progress"
        ]
        assert _row(detector_conn, tid)["last_progress_at"] > old_progress_at
        assert (deferred, len(receipts)) == ([], 0)


def test_holding_the_claim_never_shortens_it(kanban_home, monkeypatch):
    """A live worker renews its own claim well past the grace; the hold is a
    floor, not a reset, so it cannot bring a healthy claim forward."""
    _forbid_signals(monkeypatch)
    with kb.connect() as conn:
        tid = _running_task(conn)
        _age_progress(conn, tid, 7200)
        far = int(time.time()) + 10 * kb.NO_PROGRESS_DEFER_GRACE_SECONDS
        conn.execute("UPDATE tasks SET claim_expires = ? WHERE id = ?", (far, tid))
        conn.commit()
        assert kb.detect_no_progress_running(conn, no_progress_timeout_seconds=60) == [tid]
        assert _row(conn, tid)["claim_expires"] == far


def test_deferred_receipt_fires_once_per_grace_window(kanban_home, monkeypatch):
    """The stale lease is left alone on purpose (clearing it would reset the
    clock that detected the problem), so the condition re-detects every
    tick. The receipt is bounded: one per grace window per run."""
    _forbid_signals(monkeypatch)
    with kb.connect() as conn:
        tid = _running_task(conn)
        _age_progress(conn, tid, 7200)

        assert kb.detect_no_progress_running(conn, no_progress_timeout_seconds=60) == [tid]
        for _ in range(5):
            assert kb.detect_no_progress_running(conn, no_progress_timeout_seconds=60) == []
        assert len(_events(conn, tid, "no_progress_deferred")) == 1
        assert _row(conn, tid)["status"] == "running"

        # Once the grace has elapsed, the still-expired lease is reported again.
        conn.execute(
            "UPDATE task_events SET created_at = created_at - ? "
            "WHERE task_id = ? AND kind = 'no_progress_deferred'",
            (kb.NO_PROGRESS_DEFER_GRACE_SECONDS + 1, tid),
        )
        conn.commit()
        assert kb.detect_no_progress_running(conn, no_progress_timeout_seconds=60) == [tid]
        assert len(_events(conn, tid, "no_progress_deferred")) == 2
        assert _row(conn, tid)["status"] == "running"
        assert _row(conn, tid)["consecutive_failures"] == 0


def test_the_receipt_window_is_per_run(kanban_home, monkeypatch):
    """A recent receipt for a previous attempt must not suppress the first
    receipt of the attempt that replaced it."""
    _forbid_signals(monkeypatch)
    with kb.connect() as conn:
        tid = _running_task(conn)
        _age_progress(conn, tid, 7200)
        assert kb.detect_no_progress_running(conn, no_progress_timeout_seconds=60) == [tid]
        _stub_operator_reclaim(monkeypatch)
        assert kb.reclaim_task(conn, tid, reason="operator") is True
        _forbid_signals(monkeypatch)
        kb.claim_task(conn, tid, claimer=f"{_host()}:worker")
        kb._set_worker_pid(conn, tid, 4343)
        _age_progress(conn, tid, 7200)
        assert kb.detect_no_progress_running(conn, no_progress_timeout_seconds=60) == [tid]
        assert len(_events(conn, tid, "no_progress_deferred")) == 2


def test_progress_resumed_clears_the_condition(kanban_home, monkeypatch):
    _forbid_signals(monkeypatch)
    with kb.connect() as conn:
        tid = _running_task(conn)
        _age_progress(conn, tid, 7200)
        assert kb.detect_no_progress_running(conn, no_progress_timeout_seconds=60) == [tid]
        _record_owned_progress(conn, tid)
        conn.execute(
            "UPDATE task_events SET created_at = created_at - ? "
            "WHERE task_id = ? AND kind = 'no_progress_deferred'",
            (kb.NO_PROGRESS_DEFER_GRACE_SECONDS + 1, tid),
        )
        conn.commit()
        assert kb.detect_no_progress_running(conn, no_progress_timeout_seconds=60) == []
        assert len(_events(conn, tid, "no_progress_deferred")) == 1


def test_expiry_never_changes_spawnability_or_failure_count(kanban_home, monkeypatch):
    """Through a full dispatcher tick: the deferred card stays ``running``,
    nothing is spawned for it, and the circuit breaker never moves."""
    _forbid_signals(monkeypatch)
    spawned: list = []
    with kb.connect() as conn:
        tid = _running_task(conn)
        _age_progress(conn, tid, 9000)
        for _ in range(3):
            result = kb.dispatch_once(
                conn,
                spawn_fn=lambda task, *a, **kw: spawned.append(task.id),
                no_progress_timeout_seconds=60,
            )
            assert tid not in spawned
            assert result.auto_blocked == []
            assert result.crashed == [] and result.stale == [] and result.timed_out == []
            assert result.reclaimed == 0
        task = kb.get_task(conn, tid)
        assert task.status == "running"
        assert task.consecutive_failures == 0
        assert task.claim_lock == f"{_host()}:worker"
        assert task.worker_pid == 4242
        assert kb.has_spawnable_ready(conn) is False


def test_slow_healthy_api_call_is_not_reported(kanban_home, monkeypatch):
    """A single long tool-free API call keeps its claim: liveness carries it
    and the progress lease has not yet expired."""
    _forbid_signals(monkeypatch)
    with kb.connect() as conn:
        tid = _running_task(conn)
        _age_progress(conn, tid, 1800)
        assert kb.detect_no_progress_running(conn, no_progress_timeout_seconds=2700) == []
        assert _events(conn, tid, "no_progress_deferred") == []


def test_detection_is_disabled_when_timeout_is_zero(kanban_home, monkeypatch):
    _forbid_signals(monkeypatch)
    with kb.connect() as conn:
        tid = _running_task(conn)
        conn.execute(
            "UPDATE tasks SET last_heartbeat_at = ?, last_progress_at = 1 WHERE id = ?",
            (int(time.time()), tid),
        )
        conn.commit()
        assert kb.detect_no_progress_running(conn, no_progress_timeout_seconds=0) == []
        assert kb.detect_no_progress_running(conn, no_progress_timeout_seconds=-5) == []
        assert _events(conn, tid, "no_progress_deferred") == []


def test_legacy_rows_are_measured_from_the_run_start(kanban_home, monkeypatch):
    """A run claimed before the upgrade has ``last_progress_at IS NULL``. It
    is measured from its own start — never treated as infinitely stale."""
    _forbid_signals(monkeypatch)
    with kb.connect() as conn:
        tid = _running_task(conn)
        now = int(time.time())
        conn.execute(
            "UPDATE tasks SET last_progress_at = NULL, last_heartbeat_at = ? "
            "WHERE id = ?",
            (now, tid),
        )
        conn.execute(
            "UPDATE task_runs SET started_at = ? WHERE task_id = ?", (now - 60, tid),
        )
        conn.commit()
        assert kb.detect_no_progress_running(conn, no_progress_timeout_seconds=2700) == []

        conn.execute(
            "UPDATE task_runs SET started_at = ? WHERE task_id = ?", (now - 9000, tid),
        )
        conn.commit()
        assert kb.detect_no_progress_running(conn, no_progress_timeout_seconds=2700) == [tid]
        assert _events(conn, tid, "no_progress_deferred")[0]["last_progress_at"] is None


def test_dead_host_local_worker_is_left_to_the_crash_path(kanban_home, monkeypatch):
    """Classification, not overlap: a dead PID is a crash, and the crash path
    owns it (exit-code forensics this pass cannot see)."""
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    with kb.connect() as conn:
        tid = _running_task(conn)
        _age_progress(conn, tid, 7200)
        assert kb.detect_no_progress_running(conn, no_progress_timeout_seconds=60) == []
        assert _events(conn, tid, "no_progress_deferred") == []


def test_remote_and_custom_spawn_claims_are_safely_deferred(kanban_home, monkeypatch):
    """No authority record exists in this commit, so a remote claim and a
    custom-spawn claim (no pid) are ordinary safe-defer cases — the same
    receipt, not an error, not a reclaim."""
    _forbid_signals(monkeypatch)
    with kb.connect() as conn:
        remote = _running_task(conn, title="remote", pid=None, claimer="otherhost:worker")
        custom = _running_task(conn, title="custom", pid=None)
        # This case intentionally models claims that bypassed the native spawner;
        # undo the opt-in lifecycle fixture's synthetic PID/identity.
        with kb.write_txn(conn):
            for tid in (remote, custom):
                run_id = _row(conn, tid)["current_run_id"]
                conn.execute(
                    "UPDATE tasks SET worker_pid = NULL WHERE id = ?", (tid,),
                )
                conn.execute(
                    "UPDATE task_runs SET worker_pid = NULL, worker_identity = NULL "
                    "WHERE id = ?",
                    (run_id,),
                )
        for tid in (remote, custom):
            _age_progress(conn, tid, 7200)
        deferred = kb.detect_no_progress_running(conn, no_progress_timeout_seconds=60)
        assert sorted(deferred) == sorted([remote, custom])
        for tid, host_local in ((remote, False), (custom, True)):
            payload = _events(conn, tid, "no_progress_deferred")[0]
            assert payload["host_local"] is host_local
            assert payload["worker_pid"] is None
            assert payload["reason"] == "authority_unavailable"
            assert kb.get_task(conn, tid).status == "running"


def test_orphaned_running_rows_are_left_to_reconciliation(kanban_home, monkeypatch):
    """A ``running`` card with no valid claim is ``reconcile_orphaned_running``'s
    case. Holding it would invent a ``claim_expires`` where there was none and
    hand the row to the TTL path later — beside a possibly-live process."""
    _forbid_signals(monkeypatch)
    with kb.connect() as conn:
        no_lock = _running_task(conn, title="no-lock")
        no_expiry = _running_task(conn, title="no-expiry", pid=4343)
        for tid in (no_lock, no_expiry):
            _age_progress(conn, tid, 7200)
        conn.execute("UPDATE tasks SET claim_lock = NULL WHERE id = ?", (no_lock,))
        conn.execute("UPDATE tasks SET claim_expires = NULL WHERE id = ?", (no_expiry,))
        conn.commit()

        assert kb.detect_no_progress_running(conn, no_progress_timeout_seconds=60) == []

        assert _row(conn, no_lock)["claim_lock"] is None
        assert _row(conn, no_expiry)["claim_expires"] is None
        for tid in (no_lock, no_expiry):
            assert _events(conn, tid, "no_progress_deferred") == []


def test_only_running_tasks_are_candidates(kanban_home, monkeypatch):
    _forbid_signals(monkeypatch)
    with kb.connect() as conn:
        tid = _running_task(conn)
        _age_progress(conn, tid, 7200)
        kb.block_task(conn, tid, reason="need creds")
        assert kb.detect_no_progress_running(conn, no_progress_timeout_seconds=60) == []
        assert _events(conn, tid, "no_progress_deferred") == []


def test_stale_liveness_is_reported_truthfully(kanban_home, monkeypatch):
    """The receipt records liveness separately from progress so an operator
    can tell a reasoning loop (fresh) from a wedged process (stale)."""
    _forbid_signals(monkeypatch)
    with kb.connect() as conn:
        tid = _running_task(conn)
        _age_progress(conn, tid, 7200, heartbeat_age=5000)
        assert kb.detect_no_progress_running(conn, no_progress_timeout_seconds=60) == [tid]
        payload = _events(conn, tid, "no_progress_deferred")[0]
        assert payload["liveness"] == "stale"
        assert payload["heartbeat_age_seconds"] >= 5000

        never = _running_task(conn, title="never", pid=4343)
        conn.execute(
            "UPDATE tasks SET last_heartbeat_at = NULL, last_progress_at = ? WHERE id = ?",
            (int(time.time()) - 7200, never),
        )
        conn.commit()
        assert kb.detect_no_progress_running(conn, no_progress_timeout_seconds=60) == [never]
        assert _events(conn, never, "no_progress_deferred")[0]["liveness"] == "never"


# ---------------------------------------------------------------------------
# Dispatcher wiring
# ---------------------------------------------------------------------------

def test_dispatch_tick_runs_the_detector_after_the_existing_passes(
    kanban_home, monkeypatch,
):
    """The detector is classification AFTER the crash / stale / max-runtime
    passes, so each recovery path keeps its own case."""
    _forbid_signals(monkeypatch)
    order: list[str] = []

    def _wrap(name):
        real = getattr(kb, name)

        def _inner(*a, **kw):
            order.append(name)
            return real(*a, **kw)
        return _inner

    for name in (
        "release_stale_claims", "detect_stale_running", "detect_crashed_workers",
        "enforce_max_runtime", "detect_no_progress_running", "recompute_ready",
    ):
        monkeypatch.setattr(kb, name, _wrap(name))

    with kb.connect() as conn:
        tid = _running_task(conn)
        _age_progress(conn, tid, 7200)
        result = kb.dispatch_once(
            conn, spawn_fn=lambda *a, **kw: None, no_progress_timeout_seconds=2700,
        )
    assert result.no_progress_deferred == [tid]
    assert order.index("detect_no_progress_running") > order.index("detect_crashed_workers")
    assert order.index("detect_no_progress_running") > order.index("detect_stale_running")
    assert order.index("detect_no_progress_running") > order.index("enforce_max_runtime")
    assert order.index("detect_no_progress_running") < order.index("recompute_ready")


def test_dispatch_tick_defaults_to_the_validated_default(kanban_home, monkeypatch):
    """On by default, with a window wide enough for legitimate long model
    calls; the default is the same object every entry point resolves to."""
    assert kb.DEFAULT_NO_PROGRESS_TIMEOUT_SECONDS == 45 * 60
    _forbid_signals(monkeypatch)
    with kb.connect() as conn:
        tid = _running_task(conn)
        _age_progress(conn, tid, kb.DEFAULT_NO_PROGRESS_TIMEOUT_SECONDS + 600)
        result = kb.dispatch_once(conn, spawn_fn=lambda *a, **kw: None)
        assert result.no_progress_deferred == [tid]
        assert kb.get_task(conn, tid).status == "running"


def test_dispatch_tick_below_the_default_window_is_quiet(kanban_home, monkeypatch):
    _forbid_signals(monkeypatch)
    with kb.connect() as conn:
        tid = _running_task(conn)
        _age_progress(conn, tid, kb.DEFAULT_NO_PROGRESS_TIMEOUT_SECONDS - 600)
        result = kb.dispatch_once(conn, spawn_fn=lambda *a, **kw: None)
        assert result.no_progress_deferred == []


def test_dispatch_result_carries_the_deferred_list_by_default():
    result = kb.DispatchResult()
    assert result.no_progress_deferred == []


def test_dispatch_tick_hook_classifies_a_deferred_only_tick_as_ok(monkeypatch):
    """A tick whose only action was a deferred receipt did something (a
    durable write); it must not be reported as idle."""
    from hermes_cli import lifecycle

    calls: list[dict] = []
    monkeypatch.setattr(kb, "_kanban_observer_consumed", lambda _name: True)
    monkeypatch.setattr(
        lifecycle, "invoke_hook", lambda name, **kw: calls.append({"name": name, **kw}),
    )

    kb._fire_dispatch_tick_hook(kb.DispatchResult(), board="b", dry_run=False)
    kb._fire_dispatch_tick_hook(
        kb.DispatchResult(no_progress_deferred=["t_1"]), board="b", dry_run=False,
    )
    outcomes = [c["outcome"] for c in calls if c["name"] == "on_kanban_dispatch_tick"]
    assert outcomes == ["idle", "ok"]
    assert calls[-1]["result"].no_progress_deferred == ["t_1"]


def test_dispatch_tick_hook_e2e_deferred_only_tick_is_not_idle(kanban_home, monkeypatch):
    """Through ``dispatch_once``: the only thing that happened this tick was
    a deferred receipt, and every dispatcher-health subscriber must be told
    the tick acted — this is precisely the incident being watched for."""
    from hermes_cli import lifecycle

    seen: list[dict] = []
    monkeypatch.setattr(kb, "_kanban_observer_consumed", lambda _name: True)
    monkeypatch.setattr(
        lifecycle, "invoke_hook", lambda name, **kw: seen.append({"hook": name, **kw}),
    )
    _forbid_signals(monkeypatch)
    with kb.connect() as conn:
        tid = _running_task(conn)
        _age_progress(conn, tid, 7200)
        result = kb.dispatch_once(
            conn, spawn_fn=lambda *a, **kw: None, max_spawn=0,
            no_progress_timeout_seconds=2700,
        )
    assert result.no_progress_deferred == [tid]
    ticks = [s for s in seen if s.get("hook") == "on_kanban_dispatch_tick"]
    assert ticks and ticks[-1]["outcome"] == "ok"


# ---------------------------------------------------------------------------
# Configuration: one validated default, zero disables, every entry point
# ---------------------------------------------------------------------------

def test_resolve_no_progress_timeout_seconds_contract(caplog):
    d = kb.DEFAULT_NO_PROGRESS_TIMEOUT_SECONDS
    assert kb.MIN_NO_PROGRESS_TIMEOUT_SECONDS == 60
    assert kb.resolve_no_progress_timeout_seconds(None) == d
    assert kb.resolve_no_progress_timeout_seconds(0) == 0          # explicit off
    assert kb.resolve_no_progress_timeout_seconds("0") == 0
    assert kb.resolve_no_progress_timeout_seconds(60) == 60         # the minimum
    assert kb.resolve_no_progress_timeout_seconds(900) == 900
    assert kb.resolve_no_progress_timeout_seconds("900") == 900     # YAML string
    assert kb.resolve_no_progress_timeout_seconds(900.0) == 900
    # Every rejection falls back to the DEFAULT, never to 0: a typo must not
    # silently switch the guard off — and each one is logged at WARNING.
    with caplog.at_level("WARNING"):
        for bad in (True, False, -5, "-5", 1, 30, 45, 59, "banana", [], {},
                    float("inf"), float("-inf"), float("nan")):
            caplog.clear()
            assert kb.resolve_no_progress_timeout_seconds(bad) == d, bad
            assert "no_progress_timeout_seconds" in caplog.text, bad


def test_config_default_and_example_expose_the_timeout(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from hermes_cli.config import DEFAULT_CONFIG, load_config

    assert DEFAULT_CONFIG["kanban"]["no_progress_timeout_seconds"] == (
        kb.DEFAULT_NO_PROGRESS_TIMEOUT_SECONDS
    )
    assert load_config()["kanban"]["no_progress_timeout_seconds"] == (
        kb.DEFAULT_NO_PROGRESS_TIMEOUT_SECONDS
    )

    repo_root = Path(__file__).resolve().parents[2]
    example = (repo_root / "cli-config.yaml.example").read_text(encoding="utf-8")
    assert (
        f"no_progress_timeout_seconds: {kb.DEFAULT_NO_PROGRESS_TIMEOUT_SECONDS}"
        in example
    )


def test_user_config_override_propagates(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (home / "config.yaml").write_text(
        "kanban:\n  no_progress_timeout_seconds: 600\n", encoding="utf-8",
    )
    from hermes_cli.config import load_config

    assert kb.resolve_no_progress_timeout_seconds(
        load_config()["kanban"]["no_progress_timeout_seconds"]
    ) == 600


def test_configured_setting_reader_fails_open(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"kanban": {"no_progress_timeout_seconds": 600}},
    )
    assert kb.configured_kanban_setting("no_progress_timeout_seconds") == 600

    def _boom():
        raise RuntimeError("config unreadable")

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", _boom)
    assert kb.configured_kanban_setting("no_progress_timeout_seconds") is None
    assert kb.resolve_no_progress_timeout_seconds(
        kb.configured_kanban_setting("no_progress_timeout_seconds")
    ) == kb.DEFAULT_NO_PROGRESS_TIMEOUT_SECONDS


@pytest.mark.parametrize(
    "configured, expected",
    [(600, 600), (0, 0), ("banana", "default"), (None, "default")],
)
def test_standalone_daemon_passes_the_resolved_timeout(monkeypatch, configured, expected):
    """The deprecated ``--force`` daemon is still a dispatch entry point; it
    must resolve the bound exactly like the gateway and the CLI."""
    if expected == "default":
        expected = kb.DEFAULT_NO_PROGRESS_TIMEOUT_SECONDS
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"kanban": {"no_progress_timeout_seconds": configured}},
    )
    monkeypatch.setattr(kb, "resolve_max_in_progress", lambda _c: None)
    monkeypatch.setattr(kb, "connect", lambda *a, **kw: type(
        "_Conn", (), {"close": lambda self: None},
    )())
    captured: dict = {}
    stop = threading.Event()

    def _fake_dispatch_once(conn, **kwargs):
        captured.update(kwargs)
        stop.set()
        return kb.DispatchResult()

    monkeypatch.setattr(kb, "dispatch_once", _fake_dispatch_once)
    kb.run_daemon(interval=0.01, stop_event=stop)
    assert captured.get("no_progress_timeout_seconds") == expected


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def test_diagnostics_surface_a_deferred_no_progress_receipt(kanban_home, monkeypatch):
    from hermes_cli import kanban_diagnostics as kd

    _forbid_signals(monkeypatch)
    with kb.connect() as conn:
        tid = _running_task(conn)
        _age_progress(conn, tid, 7200)
        assert kb.detect_no_progress_running(conn, no_progress_timeout_seconds=2700) == [tid]
        task = kb.get_task(conn, tid)
        events = kb.list_events(conn, tid)
        runs = kb.list_runs(conn, tid)

    assert "no_progress_deferred" in kd.DIAGNOSTIC_KINDS
    diags = [
        d for d in kd.compute_task_diagnostics(task, events, runs)
        if d.kind == "no_progress_deferred"
    ]
    assert len(diags) == 1
    diag = diags[0]
    assert diag.severity == "warning"
    assert diag.run_id == task.current_run_id
    assert "no observable progress" in diag.title.lower()
    assert diag.data["progress_age_seconds"] >= 7200
    assert diag.data["timeout_seconds"] == 2700
    assert any(a.kind == "reclaim" for a in diag.actions)
    assert any(
        a.kind == "cli_hint" and f"hermes kanban log {tid}" in a.payload.get("command", "")
        for a in diag.actions
    )


def test_diagnostics_drop_the_receipt_once_progress_resumes(kanban_home, monkeypatch):
    from hermes_cli import kanban_diagnostics as kd

    _forbid_signals(monkeypatch)
    with kb.connect() as conn:
        tid = _running_task(conn)
        _age_progress(conn, tid, 7200)
        assert kb.detect_no_progress_running(conn, no_progress_timeout_seconds=60) == [tid]
        _record_owned_progress(conn, tid)
        task = kb.get_task(conn, tid)
        events = kb.list_events(conn, tid)
        runs = kb.list_runs(conn, tid)
    assert [
        d for d in kd.compute_task_diagnostics(task, events, runs)
        if d.kind == "no_progress_deferred"
    ] == []


def test_diagnostics_ignore_receipts_from_a_previous_attempt(kanban_home, monkeypatch):
    from hermes_cli import kanban_diagnostics as kd

    _forbid_signals(monkeypatch)
    with kb.connect() as conn:
        tid = _running_task(conn)
        _age_progress(conn, tid, 7200)
        assert kb.detect_no_progress_running(conn, no_progress_timeout_seconds=60) == [tid]
        _stub_operator_reclaim(monkeypatch)
        assert kb.reclaim_task(conn, tid, reason="operator") is True
        kb.claim_task(conn, tid, claimer=f"{_host()}:worker")
        task = kb.get_task(conn, tid)
        events = kb.list_events(conn, tid)
        runs = kb.list_runs(conn, tid)
    assert [
        d for d in kd.compute_task_diagnostics(task, events, runs)
        if d.kind == "no_progress_deferred"
    ] == []


def test_diagnostics_read_raw_rows_as_the_dashboard_passes_them(kanban_home, monkeypatch):
    """The dashboard feeds the rule engine raw ``sqlite3.Row`` objects whose
    ``payload`` is still a JSON string; the rule must read the receipt
    either way."""
    from hermes_cli import kanban_diagnostics as kd

    _forbid_signals(monkeypatch)
    with kb.connect() as conn:
        tid = _running_task(conn)
        _age_progress(conn, tid, 7200)
        assert kb.detect_no_progress_running(conn, no_progress_timeout_seconds=2700) == [tid]
        task_row = conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone()
        event_rows = conn.execute(
            "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (tid,),
        ).fetchall()
        run_rows = conn.execute(
            "SELECT * FROM task_runs WHERE task_id = ? ORDER BY id", (tid,),
        ).fetchall()
    diags = [
        d for d in kd.compute_task_diagnostics(task_row, event_rows, run_rows)
        if d.kind == "no_progress_deferred"
    ]
    assert len(diags) == 1
    assert diags[0].data["progress_age_seconds"] >= 7200
    assert diags[0].data["timeout_seconds"] == 2700
    assert diags[0].to_dict()["run_id"] == task_row["current_run_id"]
