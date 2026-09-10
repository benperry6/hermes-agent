"""Deterministic registration/dispatch inversion through real A2A admission."""
import asyncio
import threading

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.a2a.adapter import A2AAdapter
from plugins.platforms.a2a import protocol


@pytest.mark.parametrize("b_failed", [True, False])
def test_admission_inversion_preserves_task_ownership(monkeypatch, b_failed):
    adapter = A2AAdapter(PlatformConfig(enabled=True))
    registered = threading.Event()
    release = threading.Event()
    b_finished = asyncio.Event()
    ids = {}
    original_add = adapter._add_pending

    def add(task_id, context_id):
        future = original_add(task_id, context_id)
        if not ids:
            ids["A"] = task_id
            registered.set()
            assert release.wait(10)
        else:
            ids["B"] = task_id
        return future

    adapter._add_pending = add

    async def runner(event):
        name = "A" if event.message_id == ids["A"] else "B"
        failed = b_failed if name == "B" else not b_failed
        if failed:
            event.processing_error = {"code": "fixture_failure"}
        return "response-" + name

    async def dispatch(event):
        await adapter._process_message_background(event, "fixture-session")
        if event.message_id == ids.get("B"):
            b_finished.set()

    adapter.handle_message = dispatch
    adapter._message_handler = runner

    async def run():
        adapter._loop = asyncio.get_running_loop()
        params = {"message": protocol.text_message(protocol.ROLE_USER, "benign", context_id="ctx-inversion")}
        a = asyncio.create_task(asyncio.to_thread(adapter._prepare_task, params, "fixture-peer"))
        try:
            assert await asyncio.to_thread(registered.wait, 10)
            _, b = await asyncio.to_thread(adapter._prepare_task, params, "fixture-peer")
            await asyncio.wait_for(b_finished.wait(), 10)
            assert not adapter._pending[ids["A"]][1].done(), "B settled A before A dispatch"
            expected_b = protocol.STATE_FAILED if b_failed else protocol.STATE_COMPLETED
            assert b["future"].result(timeout=0) == (expected_b, "response-B")
            await adapter.send("ctx-inversion", "late-B", metadata={
                "notify": True, "_processing_message_id": ids["B"]})
            assert not adapter._pending[ids["A"]][1].done()
            assert b["future"].result(timeout=0) == (expected_b, "response-B")
        finally:
            release.set()
            _, pending_a = await a
        await asyncio.wait_for(asyncio.wrap_future(pending_a["future"]), 10)
        expected_a = protocol.STATE_COMPLETED if b_failed else protocol.STATE_FAILED
        assert pending_a["future"].result(timeout=0) == (expected_a, "response-A")
    asyncio.run(run())
