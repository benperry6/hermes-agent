"""PR103453: one submission, separate HTTP clients, native task ownership.

No provider or production transport is called; real adapter/base callbacks and
HTTP server run with an inert runner. Existing timeout/cancel policy is retained.
"""
import asyncio
import json
import threading
import time
import urllib.request
from unittest.mock import patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.event import ProcessingOutcome
from plugins.platforms.a2a.adapter import A2AAdapter
from plugins.platforms.a2a import protocol, security


@pytest.fixture
def service(monkeypatch):
    import inspect
    from pathlib import Path
    assert Path(inspect.getfile(A2AAdapter)).resolve() == Path(__file__).parents[2] / "plugins/platforms/a2a/adapter.py", inspect.getfile(A2AAdapter)
    adapter = A2AAdapter(PlatformConfig(enabled=True, extra={"port": 0}))
    monkeypatch.setattr(adapter, "_wire_plugin_handlers", lambda native=None: None)
    released = threading.Event()
    admitted = threading.Event()
    calls = []

    async def inert_runner(event):
        calls.append(event)
        admitted.set()
        while not released.is_set():
            await asyncio.sleep(0.01)
        return "final:" + event.message_id

    adapter._message_handler = inert_runner
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    assert asyncio.run_coroutine_threadsafe(adapter.connect(), loop).result(10)
    assert adapter._httpd is not None
    url = f"http://127.0.0.1:{adapter._httpd.server_address[1]}/"

    def request(method, params):
        req = urllib.request.Request(url, data=json.dumps({"jsonrpc": "2.0", "id": "inert", "method": method, "params": params}).encode(), headers={"Content-Type": "application/json"})
        # Each call opens and closes a distinct real HTTP connection.
        with urllib.request.urlopen(req, timeout=2) as response:
            return json.load(response)

    def finish(task_id):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            task = request("tasks/get", {"id": task_id})["result"]
            if task["status"]["state"] in protocol.TERMINAL_STATES or task["status"]["state"] == protocol.STATE_INPUT_REQUIRED:
                return task
            time.sleep(0.1)
        raise AssertionError("native completion was not retrievable")

    yield adapter, loop, released, admitted, calls, request, finish
    released.set()
    asyncio.run_coroutine_threadsafe(adapter.disconnect(), loop).result(10)
    async def settle():
        pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    asyncio.run_coroutine_threadsafe(settle(), loop).result(10)
    loop.call_soon_threadsafe(loop.stop)
    thread.join(10)
    loop.close()


@pytest.mark.parametrize("outcome", ["success", "empty", "input", "failure", "cancel", "disconnect", "same-context", "long-input", "cancel-retry"])
def test_detached_submit_get_preserves_native_outcome_once(service, monkeypatch, outcome):
    adapter, loop, release, admitted, calls, request, finish = service
    context = "detached-inert-context"
    if outcome == "long-input":
        from types import SimpleNamespace
        clock = [100.0]
        monkeypatch.setattr(protocol, "time", SimpleNamespace(time=lambda: clock[0]))
        async def long_reasoning(event):
            calls.append(event)
            admitted.set()
            while not release.is_set():
                await asyncio.sleep(0.01)
            clock[0] += 7200
            return protocol.INPUT_REQUIRED_MARKER + " Choose the inert option"
        adapter.set_message_handler(long_reasoning)
    params = {"configuration": {"returnImmediately": True}, "message": protocol.text_message(protocol.ROLE_USER, "inert", context_id=context)}
    persisted, audited = [], []
    native_persist, native_audit = protocol.persist_message, security.audit
    def persist(*args):
        persisted.append(args)
        native_persist(*args)
    def audit(*args):
        audited.append(args)
        native_audit(*args)
    monkeypatch.setattr(protocol, "persist_message", persist)
    monkeypatch.setattr(security, "audit", audit)
    first = request("message/send", params)["result"]
    assert first["status"]["state"] == protocol.STATE_WORKING
    task_id = first["id"]
    assert admitted.wait(2)
    assert request("tasks/get", {"id": task_id})["result"]["id"] == task_id
    second = None
    if outcome == "same-context":
        second = request("message/send", params)["result"]
        assert second["id"] != task_id
    elif outcome in ("cancel", "cancel-retry"):
        assert request("tasks/cancel", {"id": task_id})["result"]["status"]["state"] == protocol.STATE_CANCELED
        if outcome == "cancel-retry":
            limit = time.monotonic() + 5
            while task_id in adapter._pending and time.monotonic() < limit:
                time.sleep(0.01)
            assert task_id not in adapter._pending
            assert adapter._active_sessions  # EXISTING native worker guard
            second = request("message/send", params)["result"]
    elif outcome == "disconnect":
        # Resolve the native shutdown callback, keeping the inert HTTP listener
        # available solely to inspect the in-memory task (no production restart).
        adapter._resolve_task(task_id, protocol.STATE_FAILED, "[agent shutting down]")
    elif outcome not in ("success", "long-input"):
        text = {"empty": "", "input": protocol.INPUT_REQUIRED_MARKER + " clarify", "failure": "[native inactivity failure]"}[outcome]
        state = protocol.STATE_FAILED if outcome == "failure" else protocol.STATE_COMPLETED
        adapter._resolve_task(task_id, state, text)
    release.set()
    final = finish(task_id)
    expected = {"cancel": protocol.STATE_CANCELED, "cancel-retry": protocol.STATE_CANCELED, "disconnect": protocol.STATE_FAILED, "failure": protocol.STATE_FAILED, "input": protocol.STATE_INPUT_REQUIRED, "long-input": protocol.STATE_INPUT_REQUIRED}.get(outcome, protocol.STATE_COMPLETED)
    assert final["status"]["state"] == expected
    if outcome == "long-input":
        assert adapter.tasks.fail_orphans(3600) == []
        again = request("tasks/get", {"id": task_id})["result"]
        assert again["status"]["state"] == protocol.STATE_INPUT_REQUIRED
        assert again["status"]["message"]["parts"][0]["text"] == "Choose the inert option"
    if second:
        final2 = finish(second["id"])
        assert final2["status"]["state"] == protocol.STATE_REJECTED
        assert second["id"] not in adapter._active_tasks
        assert "in flight" in final2["status"]["message"]["parts"][0]["text"]
    # Native late callbacks cannot overwrite a terminal task or settle a sibling.
    assert not adapter._resolve_task(task_id, protocol.STATE_COMPLETED, "late")
    for _ in range(3):
        again = request("tasks/get", {"id": task_id})["result"]
        assert again["id"] == final["id"] and again["contextId"] == final["contextId"]
        assert again["status"]["state"] == final["status"]["state"]
        assert [a["parts"] for a in again.get("artifacts", [])] == [a["parts"] for a in final.get("artifacts", [])]
        assert (again["status"].get("message") or {}).get("parts") == (final["status"].get("message") or {}).get("parts")
    deadline = time.monotonic() + 3
    while task_id in adapter._active_tasks and time.monotonic() < deadline:
        time.sleep(0.01)
    assert task_id not in adapter._active_tasks
    assert sum(row[1] == "agent" and row[3] == task_id for row in persisted) == 1
    assert sum(row[0] == "outbound" and row[2] == task_id for row in audited) == 1
    assert len(calls) == 1


def test_detached_new_send_allows_native_heal_after_done_owner(service):
    """Prepared nonregression: real native unwind leaves a swapped guard.

    Retrieving the previous ID is not resuming that task: the next message in
    this context gets a new ID and must reach handle_message's existing heal.
    """
    adapter, loop, release, admitted, calls, request, finish = service
    params = {"configuration": {"returnImmediately": True}, "message": protocol.text_message(
        protocol.ROLE_USER, "inert", context_id="retained-guard-context")}
    first = request("message/send", params)["result"]
    assert admitted.wait(2)

    async def retain_swapped_guard():
        key = adapter._event_session_key(calls[0])
        owner = adapter._session_tasks[key]
        assert not owner.done()
        replacement = asyncio.Event()
        adapter._active_sessions[key] = replacement
        return key, owner, replacement

    key, owner, replacement = asyncio.run_coroutine_threadsafe(retain_swapped_guard(), loop).result(2)
    release.set()
    assert finish(first["id"])["status"]["state"] == protocol.STATE_COMPLETED

    async def after_native_unwind():
        await asyncio.wait_for(asyncio.shield(owner), 2)
        assert owner.done()
        assert adapter._active_sessions[key] is replacement
        assert adapter._session_tasks[key] is owner
        assert adapter._session_task_is_stale(key)

    asyncio.run_coroutine_threadsafe(after_native_unwind(), loop).result(3)
    deadline = time.monotonic() + 3
    while first["id"] in adapter._pending and time.monotonic() < deadline:
        time.sleep(0.01)
    assert first["id"] not in adapter._pending
    release.clear()
    admitted.clear()
    second = request("message/send", params)["result"]
    assert second["id"] != first["id"]
    assert admitted.wait(2), "stale guard wrongly rejected a new detached message"
    assert len(calls) == 2
    assert not owner is adapter._session_tasks[key]
    assert adapter._active_sessions[key] is not replacement
    assert not adapter._session_tasks[key].done()
    release.set()
    final = finish(second["id"])
    assert final["status"]["state"] == protocol.STATE_COMPLETED
    assert final["artifacts"][0]["parts"][0]["text"] == "final:" + second["id"]
    assert request("tasks/get", {"id": first["id"]})["result"]["status"]["state"] == protocol.STATE_COMPLETED


@pytest.mark.parametrize("case", ["start-failure", "future-failure", "dispatch-failure", "sync-dispatch-failure", "sync", "nonlocal", "no-handler", "mismatch", "unresolved", "draining"])
def test_detached_failure_and_existing_blocking_paths(service, monkeypatch, case):
    adapter, loop, release, admitted, calls, request, finish = service
    params = {"configuration": {"returnImmediately": True}, "message": protocol.text_message(protocol.ROLE_USER, "inert", context_id="other-context")}
    if case in ("no-handler", "mismatch", "unresolved", "draining"):
        native_handle = adapter.handle_message
        if case == "draining":
            from tests.gateway.test_a2a_runner_delivery import _make_runner
            from gateway.config import Platform
            runner = _make_runner()
            runner.adapters[Platform("a2a")] = adapter
            runner._external_drain_active = True
            async def forbidden_agent(*args, **kwargs):
                raise AssertionError("draining must never start an agent")
            monkeypatch.setattr(runner, "_run_agent", forbidden_agent)
            adapter.set_message_handler(runner._handle_message)
        else:
            async def native_rejection(event):
                if case == "no-handler":
                    adapter._message_handler = None
                elif case == "mismatch":
                    event.metadata = {"gateway_session_key": "not-the-derived-session"}
                else:
                    event.source.profile_route_rejected = True
                await native_handle(event)
            monkeypatch.setattr(adapter, "handle_message", native_rejection)
        result = request("message/send", params)["result"]
        final = finish(result["id"])
        if case == "draining":
            # Real runner returns a native maintenance reply, never an agent turn.
            assert "draining" in final["artifacts"][0]["parts"][0]["text"]
        else:
            assert final["status"]["state"] == protocol.STATE_REJECTED
        assert not calls
        assert result["id"] not in adapter._active_tasks
    elif case in ("dispatch-failure", "sync-dispatch-failure"):
        params["configuration"]["returnImmediately"] = case == "dispatch-failure"
        async def refuse(event):
            raise RuntimeError("inert async dispatch failure")
        monkeypatch.setattr(adapter, "handle_message", refuse)
        result = request("message/send", params)["result"]
        assert finish(result["id"])["status"]["state"] == protocol.STATE_FAILED
        assert result["id"] not in adapter._active_tasks
    elif case == "sync":
        params.pop("configuration")
        release.set()
        assert request("message/send", params)["result"]["status"]["state"] == protocol.STATE_COMPLETED
    elif case == "nonlocal":
        agent = {"local": False, "slug": "sibling", "tenant": "sibling", "timeout": 3600}
        monkeypatch.setattr(adapter, "_forward_to_profile", lambda *a: ("native sibling", protocol.STATE_COMPLETED))
        result = adapter._rpc_message_send("sibling", params, "fixture-peer", agent=agent)["result"]
        assert result["status"]["state"] == protocol.STATE_COMPLETED
        assert result["artifacts"][0]["parts"][0]["text"] == "native sibling"
    elif case == "start-failure":
        # Only the exact finalizer spawn is refused; native server threads exist.
        native_start = threading.Thread.start
        def start(thread):
            if thread.name.startswith("a2a-finalize-"):
                raise RuntimeError("inert finalizer start refused")
            return native_start(thread)
        monkeypatch.setattr(threading.Thread, "start", start)
        result = request("message/send", params)["result"]
        assert result["status"]["state"] == protocol.STATE_FAILED
        assert result["id"] not in adapter._active_tasks
        assert request("tasks/get", {"id": result["id"]})["result"]["status"]["state"] == protocol.STATE_FAILED
    else:
        result = request("message/send", params)["result"]
        adapter._pending[result["id"]][1].set_exception(RuntimeError("inert future failure"))
        assert finish(result["id"])["status"]["state"] == protocol.STATE_FAILED
        assert result["id"] not in adapter._active_tasks
