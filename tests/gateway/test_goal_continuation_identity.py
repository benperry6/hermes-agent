"""A queued /goal continuation belongs to the goal that judged it.

A continuation the judge queued for goal A must never start once A is gone: cleared, replaced by
goal B (with or without a /goal clear first), or set again with the same text, even when the row
is already committed to the session authority's durable FIFO. It names A by its persistent
``GoalState.created_at`` (as a heartbeat admission names its ``HeartbeatState``), so a new goal
with the same text is not mistaken for it, and a criterion added to A (/subgoal) keeps it.

A continuation drained from the adapter slot joins the FIFO as the same identified wake, so it is
still checked when it waits behind a follow-up.
"""

from __future__ import annotations

import asyncio

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent

MARKER = "[Continuing toward your standing goal]"


async def _telegram_gateway(tmp_path, monkeypatch, instance_id=None):
    import gateway.run as gr
    from gateway.config import GatewayConfig
    from gateway.platforms.base import SendResult
    from gateway.session_authority import initialize_session_authority
    from gateway.session_ingress_context import register_transport_home
    from hermes_cli import goals
    from persistence_assembly import open_persistence
    from plugins.platforms.telegram.adapter import TelegramAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "42")
    monkeypatch.setattr(gr, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    runner = gr.GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    authority = None  # without one: the classic in-process path
    if instance_id is not None:
        authority = await initialize_session_authority(runner, profile_id="default", instance_id=instance_id)
    db = open_persistence(runner.session_store.persistence.scope)
    monkeypatch.setattr(goals, "_get_session_db", lambda: db)
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fixture", typing_indicator=False))
    adapter.gateway_runner, runner.adapters = runner, {Platform.TELEGRAM: adapter}
    runner._wire_adapter_handlers(adapter)
    register_transport_home(runner, None, tmp_path)
    sent = []

    async def send(chat_id, content, reply_to=None, metadata=None):
        sent.append((chat_id, content))
        return SendResult(success=True, message_id=str(len(sent)))

    monkeypatch.setattr(adapter, "send", send)
    return runner, authority, adapter, db, sent


def _reply(text):
    return {"final_response": text, "messages": [], "tools": [], "history_offset": 0, "last_prompt_tokens": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["clear_then_set", "replace", "same_text_again", "storage_reset",
                                    "storage_reset_before_admission", "subgoal_added"])
async def test_durable_continuation_never_starts_for_a_goal_that_is_gone(tmp_path, monkeypatch, change):
    """Goal A's continuation is committed to the durable FIFO (a delegation result started the judged
    turn); before it runs, the real /goal handlers change the goal. Only a continuation of the goal
    still standing reaches the agent."""
    from hermes_cli import goals
    from tests.waiting import async_wait_until

    runner, authority, adapter, db, _sent = await _telegram_gateway(tmp_path, monkeypatch, "goal-identity")
    calls = []

    async def respond(**kw):
        calls.append(kw["message"])
        return _reply(f"reply {len(calls)}")

    monkeypatch.setattr(runner, "_run_agent", respond)
    verdicts = iter([("continue", "one read remains", False, None, False)])
    monkeypatch.setattr(goals, "judge_goal", lambda *a, **k: next(verdicts, ("done", "finished", False, None, False)))
    source = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")
    route = runner._session_key_for_source(source)
    real_judge = runner._post_turn_goal_continuation
    armed, judged = [], []

    async def command(text):
        handler = runner._handle_subgoal_command if text.startswith("/subgoal") else runner._handle_goal_command
        return await handler(MessageEvent(text=text, source=source, message_id=f"cmd-{text}"))

    async def judge_then_change(**kw):
        await real_judge(**kw)
        if not armed or judged:
            return
        judged.append(True)
        # The continuation is queued (wherever this build queues it) and has not started.
        if change == "storage_reset_before_admission":
            return
        if change == "storage_reset":  # a policy reset rotates the route without /new
            runner.session_store.reset_session(route)
            return
        for text in {"clear_then_set": ["/goal clear", "/goal ship release B"],
                     "replace": ["/goal ship release B"],
                     "same_text_again": ["/goal clear", "/goal ship release A"],
                     "subgoal_added": ["/subgoal also tag the release"]}[change]:
            await command(text)

    monkeypatch.setattr(runner, "_post_turn_goal_continuation", judge_then_change)
    if change == "storage_reset_before_admission":
        from gateway import session_automation
        real_wake = session_automation.admit_route_wake

        async def reset_then_wake(*args, **kwargs):
            runner.session_store.reset_session(route)  # between the judgment and the commit
            return await real_wake(*args, **kwargs)

        monkeypatch.setattr(session_automation, "admit_route_wake", reset_then_wake)
    try:
        await adapter.handle_message(MessageEvent(text="seed the route", source=source, message_id="1"))
        await async_wait_until(lambda: len(calls) == 1 and not adapter._background_tasks, what="human turn delivered")
        sid = runner.session_store.peek_session_id(route)
        goals.GoalManager(sid).set("ship release A")
        live = authority.sessions[sid]
        armed.append(True)  # the next judgment is goal A's
        assert await runner._inject_watch_notification("[IMPORTANT: background result arrived]", {
            "type": "async_delegation", "delegation_id": "deleg_identity", "status": "completed",
            "parent_session_id": sid, "session_key": route, "platform": "telegram",
            "chat_type": "dm", "chat_id": "42", "user_id": "42", "output": "read-only result",
        }) is True
        await live.task
        # A later message drains whatever is still parked for this chat.
        await adapter.handle_message(MessageEvent(text="next question", source=source, message_id="2"))
        await async_wait_until(lambda: "next question" in calls and not adapter._background_tasks
                               and (live.task is None or live.task.done()), what="the later message ran")

        assert judged, "the judged turn never ran"
        rows = authority.persistence.admissions.list(session_id=authority.logical_owner(sid), pending_only=False)
        assert all(row.status == "terminal" for row in rows), [(r.request_id[:40], r.status) for r in rows]
        continued = [c for c in calls if c.startswith(MARKER)]
        if change == "subgoal_added":
            assert len(continued) == 1 and "ship release A" in continued[0], calls
        else:
            assert continued == [], f"a continuation of the goal that is gone reached the agent: {calls}"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_real_new_still_cancels_a_queued_continuation(tmp_path, monkeypatch):
    """The user's /new, through the adapter, after the continuation was committed: it never runs,
    and the fresh session's next message is answered normally."""
    from hermes_cli import goals
    from tests.waiting import async_wait_until

    runner, authority, adapter, db, _sent = await _telegram_gateway(tmp_path, monkeypatch, "goal-new")
    calls = []

    async def respond(**kw):
        calls.append((kw["session_id"], kw["message"]))
        return _reply(f"reply {len(calls)}")

    monkeypatch.setattr(runner, "_run_agent", respond)
    monkeypatch.setattr(goals, "judge_goal", lambda *a, **k: ("continue", "one read remains", False, None, False))
    source = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")
    route = runner._session_key_for_source(source)
    real_judge = runner._post_turn_goal_continuation
    armed, reset = [], []

    async def judge_then_new(**kw):
        await real_judge(**kw)
        if armed and not reset:
            reset.append(True)
            await adapter.handle_message(MessageEvent(text="/new", source=source, message_id="new"))
            await async_wait_until(lambda: runner.session_store.peek_session_id(route) != sid,
                                   what="the route was reset")

    monkeypatch.setattr(runner, "_post_turn_goal_continuation", judge_then_new)
    try:
        await adapter.handle_message(MessageEvent(text="seed the route", source=source, message_id="1"))
        await async_wait_until(lambda: len(calls) == 1 and not adapter._background_tasks, what="human turn delivered")
        sid = runner.session_store.peek_session_id(route)
        goals.GoalManager(sid).set("ship release A")
        live = authority.sessions[sid]
        armed.append(True)
        assert await runner._inject_watch_notification("[IMPORTANT: background result arrived]", {
            "type": "async_delegation", "delegation_id": "deleg_new", "status": "completed",
            "parent_session_id": sid, "session_key": route, "platform": "telegram",
            "chat_type": "dm", "chat_id": "42", "user_id": "42", "output": "read-only result",
        }) is True
        await live.task
        await async_wait_until(lambda: not adapter._background_tasks, what="the /new reply was sent")
        fresh = runner.session_store.peek_session_id(route)
        await adapter.handle_message(MessageEvent(text="fresh question", source=source, message_id="2"))
        await async_wait_until(lambda: ("fresh question" in [m for _, m in calls]) and not adapter._background_tasks,
                               what="the fresh session answered")

        assert reset and fresh != sid
        assert [m for _, m in calls if m.startswith(MARKER)] == [], calls
        assert (fresh, "fresh question") in calls
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("with_authority", [True, False])
async def test_adapter_slot_continuation_never_runs_for_a_replaced_goal(tmp_path, monkeypatch, with_authority):
    """A human turn's continuation waits in the adapter slot (its task drains it after the reply).
    A goal set meanwhile (as /goal <text> does) does not clear that slot: the continuation of the
    replaced goal must be dropped when it drains, and the new goal's next turn still judged."""
    from hermes_cli import goals
    from tests.waiting import async_wait_until

    runner, _authority, adapter, db, _sent = await _telegram_gateway(
        tmp_path, monkeypatch, "goal-slot" if with_authority else None)
    calls = []

    async def respond(**kw):
        calls.append(kw["message"])
        return _reply(f"reply {len(calls)}")

    monkeypatch.setattr(runner, "_run_agent", respond)
    verdicts = iter([("continue", "more work remains", False, None, False)])
    judged = []

    def judge(*a, **k):
        judged.append(True)
        return next(verdicts, ("done", "finished", False, None, False))

    monkeypatch.setattr(goals, "judge_goal", judge)
    source = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")
    route = runner._session_key_for_source(source)
    session_a = runner.session_store.get_or_create_session(source).session_id
    goals.GoalManager(session_a).set("ship release A")
    real_enqueue, enqueued = runner._enqueue_fifo, []

    def enqueue_then_replace(session_key, queued_event, fifo_adapter):
        real_enqueue(session_key, queued_event, fifo_adapter)
        enqueued.append(queued_event.text)
        goals.GoalManager(session_a).set("ship release B")

    monkeypatch.setattr(runner, "_enqueue_fifo", enqueue_then_replace)
    try:
        await adapter.handle_message(MessageEvent(text="start on the release", source=source, message_id="1"))
        await async_wait_until(lambda: enqueued and route not in adapter._pending_messages
                               and not adapter._background_tasks, what="the continuation slot drained")
        await adapter.handle_message(MessageEvent(text="fresh question", source=source, message_id="2"))
        await async_wait_until(lambda: "fresh question" in calls and not adapter._background_tasks and len(judged) == 2,
                               what="the new goal's turn ran and was judged")

        assert enqueued and enqueued[0].startswith(MARKER) and "ship release A" in enqueued[0]
        assert [c for c in calls if c.startswith(MARKER)] == [], calls
        assert goals.GoalManager(session_a).state.goal == "ship release B"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_new_goal_judged_while_the_old_continuation_waits_still_continues(tmp_path, monkeypatch):
    """Goal A's continuation is still queued when goal B is set and B's turn is judged: A's waiting
    continuation does not stand in for B's (it is dropped when it would start), so B continues."""
    from gateway import session_automation
    from hermes_cli import goals

    runner, authority, adapter, db, _sent = await _telegram_gateway(tmp_path, monkeypatch, "goal-dedupe")
    calls = []

    async def respond(**kw):
        calls.append(kw["message"])
        return _reply(f"reply {len(calls)}")

    monkeypatch.setattr(runner, "_run_agent", respond)
    verdicts = iter([("continue", "A needs a read", False, None, False), ("continue", "B needs a read", False, None, False)])
    monkeypatch.setattr(goals, "judge_goal", lambda *a, **k: next(verdicts, ("done", "finished", False, None, False)))
    source = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")
    route = runner._session_key_for_source(source)
    real_wake, wakes = session_automation.admit_route_wake, []

    async def wake_then_new_goal(*args, **kwargs):
        accepted = await real_wake(*args, **kwargs)
        wakes.append(args[3])
        if len(wakes) == 1:
            await runner._handle_goal_command(MessageEvent(text="/goal ship release B", source=source, message_id="g"))
            entry = await runner.async_session_store.get_or_create_session(source, touch_activity=False)
            await runner._post_turn_goal_continuation(session_entry=entry, source=source, final_response="answered",
                                                      authority_owned_wake=True)
        return accepted

    monkeypatch.setattr(session_automation, "admit_route_wake", wake_then_new_goal)
    try:
        from tests.waiting import async_wait_until
        await adapter.handle_message(MessageEvent(text="seed the route", source=source, message_id="1"))
        await async_wait_until(lambda: len(calls) == 1 and not adapter._background_tasks, what="human turn delivered")
        sid = runner.session_store.peek_session_id(route)
        goals.GoalManager(sid).set("ship release A")
        live = authority.sessions[sid]
        assert await runner._inject_watch_notification("[IMPORTANT: background result arrived]", {
            "type": "async_delegation", "delegation_id": "deleg_dedupe", "status": "completed",
            "parent_session_id": sid, "session_key": route, "platform": "telegram",
            "chat_type": "dm", "chat_id": "42", "user_id": "42", "output": "read-only result",
        }) is True
        await live.task

        assert len(wakes) == 2 and "ship release A" in wakes[0] and "ship release B" in wakes[1], wakes
        assert [c.split("\n")[1] for c in calls if c.startswith(MARKER)] == ["Goal: ship release B"], calls
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["cleared_during_follow_up", "judged_during_follow_up"])
async def test_drained_continuation_behind_a_follow_up_is_still_the_goals(tmp_path, monkeypatch, case):
    """A message sent during a goal turn is queued ahead of that turn's continuation, which waits in the
    adapter slot until the turn's task drains it into the FIFO. While the follow-up runs, a /goal clear
    must still stop the continuation, and the follow-up's own judgment must not queue a second one."""
    from hermes_cli import goals
    from tests.waiting import async_wait_until

    runner, authority, adapter, db, _sent = await _telegram_gateway(tmp_path, monkeypatch, "goal-follow-up")
    calls, gates = [], {"work": asyncio.Event(), "meanwhile": asyncio.Event()}

    async def respond(**kw):
        message = kw["message"]
        calls.append(message)
        if message in gates:
            await gates[message].wait()
        return _reply(f"answered {len(calls)}")

    monkeypatch.setattr(runner, "_run_agent", respond)
    verdicts = iter([("continue", "one step remains", False, None, False)] * 2)
    monkeypatch.setattr(goals, "judge_goal", lambda *a, **k: next(verdicts, ("done", "all done", False, None, False)))
    source = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")
    route = runner._session_key_for_source(source)
    sid = runner.session_store.get_or_create_session(source).session_id
    goals.GoalManager(sid).set("finish the report")
    try:
        await adapter.handle_message(MessageEvent(text="work", source=source, message_id="1"))
        await async_wait_until(lambda: calls == ["work"], what="the goal turn started")
        await adapter.handle_message(MessageEvent(text="meanwhile", source=source, message_id="2"))
        gates["work"].set()
        await async_wait_until(lambda: calls == ["work", "meanwhile"] and route not in adapter._pending_messages,
                               what="the follow-up runs and the continuation left the slot")
        if case == "cleared_during_follow_up":
            await runner._handle_goal_command(MessageEvent(text="/goal clear", source=source, message_id="c"))
        gates["meanwhile"].set()
        live = authority.sessions[authority.logical_owner(sid)]
        await async_wait_until(lambda: not adapter._background_tasks and (live.task is None or live.task.done())
                               and all(r.status == "terminal" for r in authority.persistence.admissions.list(
                                   session_id=authority.logical_owner(sid), pending_only=False)),
                               what="the route settled")

        continued = [c for c in calls if c.startswith(MARKER)]
        assert len(continued) == (0 if case == "cleared_during_follow_up" else 1), calls
    finally:
        db.close()


def test_waiting_continuation_counts_only_for_the_judged_goal():
    """A slot continuation of an earlier goal is dropped when it would start: it must not stop the
    new goal's judgment from queuing its own."""
    from types import SimpleNamespace

    from gateway.run_busy import GatewayBusySessionMixin

    key = "agent:main:telegram:dm:42"
    slot = {key: SimpleNamespace(text=f"{MARKER}\nGoal: ship", internal=False,
                                 metadata={"gateway_session_id": "s1", "goal_created_at": 100.5})}
    stub = SimpleNamespace(
        _session_key_for_source=lambda source: key,
        _delivery_adapter_for=lambda source: SimpleNamespace(_pending_messages=slot),
        _overflow_queue=lambda session_key: [],
        _is_goal_continuation_event=GatewayBusySessionMixin._is_goal_continuation_event,
        _goal_continuation_generation=GatewayBusySessionMixin._goal_continuation_generation,
    )
    waiting = GatewayBusySessionMixin._goal_continuation_waiting
    assert waiting(stub, object(), "s1", 100.5) is True
    assert waiting(stub, object(), "s1", 200.25) is False


def test_in_band_follow_up_check_names_the_goal(tmp_path, monkeypatch):
    """``_run_agent_queued_followup`` drops a drained continuation through ``_stale_goal_continuation``:
    a continuation of the goal now standing runs, one of a goal set again since does not, and any
    other input is never treated as one."""
    from types import SimpleNamespace

    from gateway.run_busy import GatewayBusySessionMixin
    from hermes_cli import goals

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    goals._DB_CACHE.clear()
    try:
        manager = goals.GoalManager("sess-inband")
        first = manager.set("ship release A").created_at
        runner = SimpleNamespace(
            _is_goal_continuation_event=GatewayBusySessionMixin._is_goal_continuation_event,
            _goal_continuation_generation=GatewayBusySessionMixin._goal_continuation_generation,
            _goal_still_active_for_session=lambda sid, gen: GatewayBusySessionMixin._goal_still_active_for_session(
                runner, sid, gen),
        )
        stale = GatewayBusySessionMixin._stale_goal_continuation

        def queued(text, generation):
            return SimpleNamespace(text=text, internal=False, metadata={"goal_created_at": generation})

        continuation = f"{MARKER}\nGoal: ship release A"
        assert stale(runner, queued(continuation, first), "sess-inband") is False
        manager.clear()
        second = goals.GoalManager("sess-inband").set("ship release A").created_at
        assert second != first
        assert stale(runner, queued(continuation, first), "sess-inband") is True
        assert stale(runner, queued(continuation, second), "sess-inband") is False
        assert stale(runner, queued("a person's /queue item", first), "sess-inband") is False
    finally:
        goals._DB_CACHE.clear()
