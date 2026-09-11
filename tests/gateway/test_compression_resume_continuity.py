"""Offline continuity contracts: real loop/result funnel, SQLite and slash handlers.

Only preflight's timeout trigger and summary synthesis are simulated. No test
claims that a model understands VA1: the contract is lossless provider-bound
context, with independent background work excluded from resume resolution.
"""
import copy
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from agent import conversation_loop
from agent.turn_context import _fail_closed_after_preflight_timeout
from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner, _build_gateway_agent_history
from gateway.run_inbound import GatewayInboundMixin
from gateway.run_turn_runner import TurnRunner
from gateway.session import AsyncSessionStore, SessionSource, SessionStore
from gateway.turn_context import TurnContext
from hermes_state import AsyncSessionDB
from run_agent import AIAgent


MANDATE = "Synthetic VA1 only: reconcile approved drafts; approval before send; exclude inventory313."
QUOTE = "Synthetic reply43734: preserve VA1 scope and approval gates."


@pytest.fixture
def lane(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    # Pin the synthetic route's window: no remote /models metadata probe.
    (home / "config.yaml").write_text(
        "model:\n  default: test-model\n  provider: openai\n"
        "  base_url: http://127.0.0.1:1/v1\n  context_length: 256000\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", home / "state.db")
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
    # AIAgent probes localhost for Ollama even with an explicit context window.
    # That unrelated metadata transport is offline here; compression stays native.
    monkeypatch.setattr("agent.model_metadata.detect_local_server_type", lambda *a, **kw: "unknown")
    network = []

    def forbidden_connect(*args, **kwargs):
        import traceback
        network.append("".join(traceback.format_stack(limit=35)))
        raise AssertionError("offline contract attempted network")

    monkeypatch.setattr(socket.socket, "connect", forbidden_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden_connect)
    store = SessionStore(home / "sessions", GatewayConfig())
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)
    runner._session_db = AsyncSessionDB(store._db)
    runner._provider_routing = {}
    runner._running_agents = {}
    runner._voice_mode = {}
    runner._resolve_session_agent_runtime = Mock(return_value=("test-model", {
        "api_key": "offline-fixture", "base_url": "http://127.0.0.1:1/v1", "provider": "openai",
    }))
    runner._resolve_session_reasoning_config = Mock(return_value=None)
    runner._resolve_session_service_tier = Mock(return_value=None)
    runner._resolve_turn_agent_config = Mock(return_value={})
    for name in ("_evict_cached_agent", "_clear_conversation_scope", "_sync_telegram_topic_binding",
                 "_release_running_agent_state"):
        setattr(runner, name, Mock())
    runner._refresh_agent_cache_message_count = AsyncMock()
    runner._consume_pending_native_image_paths = Mock(return_value=[])
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="synthetic-room", user_id="synthetic-user",
                           thread_id="synthetic-topic", chat_type="group")
    entry = store.get_or_create_session(source)
    for message in [
        {"role": "user", "content": "Old inventory313 request " * 200},
        {"role": "assistant", "content": "Old inventory313 completed " * 200},
        {"role": "user", "content": "Close that old task."},
        {"role": "assistant", "content": "Closed; await the new scoped mandate."},
    ]:
        store.append_to_transcript(entry.session_id, message)
    sent = []

    async def send(chat_id, content, **kwargs):
        sent.append(dict(chat_id=chat_id, content=content, **kwargs))
        return SimpleNamespace(success=True, message_id=str(len(sent)))

    runner.adapters = {Platform.TELEGRAM: SimpleNamespace(send=send)}
    env = SimpleNamespace(store=store, db=store._db, runner=runner, source=source, entry=entry,
                          sent=sent, network=network, home=home)
    try:
        yield env
        assert not network, "\n".join(network)
    finally:
        store._db.close()


def timeout_projection(env, monkeypatch, empty=False):
    """Real turn-loop exception handler and run_sync projection, no provider call."""
    history = env.store.load_transcript(env.entry.session_id)
    agent = SimpleNamespace(_persist_disabled=True, model="test-model", _last_compression_timed_out=True)

    def timed_out(*args, **kwargs):
        _fail_closed_after_preflight_timeout(agent, 238978)

    monkeypatch.setattr(conversation_loop, "build_turn_context", timed_out)
    def run(message, **kwargs):
        raw = conversation_loop._run_conversation_turn(agent, message, **kwargs)
        assert raw["api_calls"] == 0
        if empty:
            raw["final_response"] = ""
        return raw

    agent.run_conversation = run
    return project_turn(env, monkeypatch, agent, MANDATE, history)


def project_turn(env, monkeypatch, agent, message, history):
    # Keep native history projection, message preparation, approval wrapper,
    # conversation call and result funnel. Only agent lookup/stream setup doubles.
    ctx = TurnContext(source=env.source, session_id=env.entry.session_id,
                      session_key=env.entry.session_key, message=message,
                      history=history, user_config={})
    ctx.agent_holder[0] = agent
    turn = TurnRunner(env.runner, ctx)
    for name, value in {
        "_combined_ephemeral_prompt": None,
        "_setup_stream_consumer": (None, None, None, False),
        "_resolve_turn_agent": (agent, False),
        "_wire_turn_agent_callbacks": None,
        "_finish_stream_consumer": None,
        "_sync_session_after_run": (False, env.entry.session_id, len(history)),
    }.items():
        monkeypatch.setattr(turn, name, Mock(return_value=value))
    monkeypatch.setattr(turn, "_append_auto_media_tags", lambda text, *_: text)
    return turn.run_sync()


def incoming(env):
    event = MessageEvent(text=MANDATE, source=env.source, message_id="44511",
                         reply_to_message_id="43734", reply_to_text=QUOTE)
    text = GatewayInboundMixin._prepend_inbound_reply_context(event, env.source, MANDATE)
    prepared = SimpleNamespace(history=env.store.load_transcript(env.entry.session_id),
        persist_user_message=text, message_text=text, persist_user_timestamp=1234.5,
        persist_user_display_kind=None, persistence_owner="synthetic-44511-owner",
        persistence_session_id=env.entry.session_id)
    return event, prepared


async def finish(env, result, event, prepared):
    runner, entry = env.runner, env.entry
    flags = runner._hmwa_classify_turn_failure(result, prepared.history, entry)
    response, target = await runner._hmwa_compression_exhaustion_reset(
        result, result.get("final_response"), entry, entry.session_key, env.source)
    await runner._hmwa_persist_turn_transcript(event=event, source=env.source,
        session_entry=target, session_key=entry.session_key, agent_result=result,
        agent_messages=result.get("messages", []), prepared=prepared, response=response,
        agent_failed_early=flags[0], hidden_reasoning_incomplete=flags[1],
        is_context_overflow_failure=flags[2])
    return target


@pytest.mark.asyncio
@pytest.mark.parametrize("case,reset,added", [
    ("timeout", False, 2), ("timeout-empty", False, 2),
    ("exhausted", True, 0), ("midloop-timeout", True, 0),
    ("deferred", False, 2), ("deferred-precedence", False, 0),
    ("transient", False, 2), ("overflow", False, 0), ("normal", False, 2),
])
async def test_failure_contract_preserves_only_recoverable_input(lane, monkeypatch, case, reset, added):
    event, prepared = incoming(lane)
    parent = lane.entry.session_id
    before = lane.store.load_transcript(parent)
    results = {
        "exhausted": dict(failed=True, compression_exhausted=True, turn_exit_reason="context_compression_exhausted"),
        "midloop-timeout": dict(failed=True, compression_exhausted=True, turn_exit_reason="context_compression_timeout", api_calls=0),
        "deferred": dict(failed=True, compression_deferred=True, agent_persisted=False),
        "deferred-precedence": dict(failed=True, compression_deferred=True, compression_exhausted=True),
        "transient": dict(failed=True, error="429", agent_persisted=False),
        "overflow": dict(failed=True, error="context length exceeded"),
        "normal": dict(failed=False, agent_persisted=False, final_response="Synthetic completed response"),
    }
    result = (timeout_projection(lane, monkeypatch, empty=case.endswith("empty"))
              if case.startswith("timeout") else results[case])
    target = await finish(lane, result, event, prepared)
    assert (target.session_id != parent) is reset
    assert len(lane.store.load_transcript(parent)) == len(before) + added
    if reset:
        assert lane.store.load_transcript(target.session_id) == []
    if case.startswith("timeout"):
        assert result["turn_exit_reason"] == "context_compression_timeout"
        assert result["agent_persisted"] is False
        assert result["compression_exhausted"] is False
        pending, closed = lane.store.load_transcript(parent)[-2:]
        # Upstream closes a failed user turn to prevent accidental replay/merging.
        assert pending["role"] == "user" and closed["role"] == "assistant"
        assert pending["content"] == prepared.persist_user_message
        assert pending["timestamp"] == prepared.persist_user_timestamp
        assert pending["message_id"] == event.message_id
        assert pending["display_metadata"]["reply_to_message_id"] == event.reply_to_message_id
        assert pending["display_metadata"]["reply_to_text"] == QUOTE
        assert pending["display_metadata"]["gateway_input_owner"] == prepared.persistence_owner
        for _ in range(100):
            await finish(lane, result, event, prepared)
        event.message_id = None
        await finish(lane, result, event, prepared)
        assert lane.store.load_transcript(parent) == before + [pending, closed]


@pytest.mark.asyncio
@pytest.mark.parametrize("in_place,modern_background", [(True, False), (False, True)])
async def test_timeout_compress_resume_parent_continue(lane, monkeypatch, in_place, modern_background, record_property):
    event, prepared = incoming(lane)
    parent, key = lane.entry.session_id, lane.entry.session_key
    result = timeout_projection(lane, monkeypatch)
    assert (await finish(lane, result, event, prepared)).session_id == parent
    pending, closed = lane.store.load_transcript(parent)[-2:]
    assert pending["role"] == "user" and closed["role"] == "assistant"
    assert pending["content"] == prepared.persist_user_message

    # The handler constructs a real AIAgent. Replace only the context engine's
    # synthesis call; native lease, archive/publication and slash persistence run.
    syntheses = []
    real_build = lane.runner._build_manual_compression_agent

    async def build(*args, **kwargs):
        agent = await real_build(*args, **kwargs)
        assert isinstance(agent, AIAgent)
        agent.compression_in_place = in_place
        agent._compression_feasibility_checked = True
        agent.skip_context_files = True

        def synthesize(messages, **kwargs):
            syntheses.append(copy.deepcopy(messages))
            assert any(prepared.persist_user_message in str(m.get("content", "")) for m in messages)
            agent.context_compressor.compression_count += 1
            return [{"role": "user", "content": "Synthetic summary: previous task closed."},
                    {"role": "assistant", "content": "Synthetic summary acknowledged."},
                    copy.deepcopy(pending), copy.deepcopy(closed)]

        monkeypatch.setattr(agent.context_compressor, "compress", synthesize)
        return agent

    monkeypatch.setattr(lane.runner, "_build_manual_compression_agent", build)
    command = MessageEvent(text="/compress VA1", source=lane.source)
    reply = await lane.runner._handle_compress_command(command)
    await lane.runner._deliver_platform_notice(lane.source, reply)
    assert len(syntheses) == 1, reply
    tip = lane.store.get_or_create_session(lane.source).session_id
    assert (tip == parent) is in_place, reply
    retained = next(m for m in lane.store.load_transcript(tip)
                    if m.get("content") == prepared.persist_user_message)
    assert retained["message_id"] == event.message_id
    assert retained["timestamp"] == prepared.persist_user_timestamp
    assert retained["display_metadata"]["reply_to_text"] == QUOTE
    assert any("Old inventory313" in str(m.get("content", ""))
               for m in lane.db.get_messages(parent, include_compacted=True))

    bg = "bg_184523_ca0d94"
    lane.db.create_session(bg, source="telegram", parent_session_id=tip,
        model_config={"_background_from": tip} if modern_background else {})
    lane.db.append_message(bg, role="user", content="Synthetic background: resume inventory313 instead")
    # Switch away, then execute the native resolver/authorization/switch handler
    # against the OLD parent ID, not an already-resolved tip supplied by the test.
    other = lane.store.reset_session(key)
    assert lane.store.load_transcript(other.session_id) == []
    resume_reply = await lane.runner._handle_resume_command(
        MessageEvent(text="/resume " + parent, source=lane.source))
    await lane.runner._deliver_platform_notice(lane.source, resume_reply)
    resumed = lane.store.get_or_create_session(lane.source)
    assert resumed.session_id == tip, resume_reply
    assert resumed.session_id != bg

    # Restart-style readback and the real provider-history projection. Continue
    # is a fresh user admission; synthesis/answer generation remains offline.
    reloaded = SessionStore(lane.home / "sessions", GatewayConfig())
    try:
        history = reloaded.load_transcript(tip)
        provider_history, _ = _build_gateway_agent_history(history)
        assert any(prepared.persist_user_message in str(m.get("content", "")) for m in provider_history)
        assert not any("Synthetic background:" in str(m.get("content", "")) for m in provider_history)
        assert not any("Old inventory313" in str(m.get("content", "")) for m in provider_history)
        continuation = MessageEvent(text="Continue", source=lane.source, message_id="44512")
        next_prepared = SimpleNamespace(**vars(prepared))
        next_prepared.history = history
        next_prepared.persist_user_message = next_prepared.message_text = "Continue"
        next_prepared.persistence_owner = "synthetic-continue-owner"
        lane.entry = resumed
        captured = []

        def capture_model_turn(message, **kwargs):
            captured.append((message, copy.deepcopy(kwargs["conversation_history"])))
            return dict(failed=False, agent_persisted=False,
                final_response="Synthetic continuation accepted; no action executed.")

        model = SimpleNamespace(model="test-model", run_conversation=capture_model_turn)
        continued = project_turn(lane, monkeypatch, model, continuation.text, history)
        assert captured[0][0] == "Continue"
        context = "\n".join(str(m.get("content", "")) for m in captured[0][1])
        assert prepared.persist_user_message in context and QUOTE in context
        assert "Synthetic background:" not in context and "Old inventory313" not in context
        await finish(lane, continued, continuation, next_prepared)
        final_history, _ = _build_gateway_agent_history(reloaded.load_transcript(tip))
        text = "\n".join(str(m.get("content", "")) for m in final_history)
        assert prepared.persist_user_message in text and QUOTE in text and "Continue" in text
        assert "Synthetic background:" not in text and "Old inventory313" not in text
        assert lane.sent[0]["content"] == reply and lane.sent[1]["content"] == resume_reply
        assert all(s["chat_id"] == lane.source.chat_id for s in lane.sent)
        record_property("compression_mode", "in_place" if in_place else "rotation")
        record_property("background_shape", "explicit_marker" if modern_background else "legacy_native_id")
        record_property("native_path", "preflight-result -> run_sync -> SQLite -> /compress -> /resume OLD parent -> Continue")
        record_property("model_boundary", "deterministic summary + captured Continue; no LLM semantics tested")
        record_property("captured_deliveries", len(lane.sent))
    finally:
        reloaded._db.close()
