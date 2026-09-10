"""Real gateway handler to A2A delivery; only model runner is substituted."""
import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionStore, SessionSource, build_session_key
from plugins.platforms.a2a.adapter import A2AAdapter
from plugins.platforms.a2a import protocol
from tests.gateway.test_max_concurrent_sessions import _make_runner, _silence_global_gateway_hooks


@pytest.mark.parametrize("failed", [False, True])
def test_real_gateway_runner_result_reaches_a2a_delivery(monkeypatch, tmp_path, failed):
    _silence_global_gateway_hooks(monkeypatch)
    runner = _make_runner()
    runner._session_db = None
    runner._set_session_env = lambda context: None
    runner.session_store = SessionStore(tmp_path / "sessions", runner.config)
    adapter = A2AAdapter(PlatformConfig(enabled=True))
    runner.adapters[adapter.platform] = adapter
    event = MessageEvent(text="benign delivery proof", message_type=MessageType.TEXT,
                         message_id="task-0123456789abcdef",
                         source=SessionSource(platform=adapter.platform, chat_id="ctx-proof",
                                              chat_type="dm", user_id="fixture-peer"))
    result = {"final_response": "proof task-0123456789abcdef", "messages": [],
              "tools": [], "history_offset": 0, "last_prompt_tokens": 0,
              "failed": failed, "completed": not failed}
    if failed:
        result.update(failure_reason="content_policy_blocked", failure_retryable=False,
                      error="fixture safety refusal")
    runner._run_agent = AsyncMock(return_value=result)
    runner._is_user_authorized = lambda source: True
    adapter._message_handler = runner._handle_message

    async def run():
        deliveries = []
        original = adapter._resolve_locked
        def resolve(task_id, state, text):
            resolved = original(task_id, state, text)
            if resolved:
                deliveries.append((task_id, state, text))
            return resolved
        adapter._resolve_locked = resolve
        future = adapter._add_pending(event.message_id, event.source.chat_id)
        await adapter._process_message_background(event, build_session_key(event.source))
        runner._run_agent.assert_awaited_once()
        assert (event.processing_error is not None) == failed
        expected = protocol.STATE_FAILED if failed else protocol.STATE_COMPLETED
        assert future.result(timeout=0)[0] == expected
        assert deliveries[0][0] == event.message_id
        assert deliveries[0][1] == expected
        assert "proof task-0123456789abcdef" in future.result(timeout=0)[1]
        assert len(deliveries) == 1
    asyncio.run(run())
