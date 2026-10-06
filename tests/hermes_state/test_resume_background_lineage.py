"""Proposal contract tests: real SQLite/agent/gateway, synthetic data and no provider calls."""
import asyncio
import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from hermes_state import SessionDB, AsyncSessionDB


class ResumeBackgroundContracts(unittest.TestCase):
    @contextlib.contextmanager
    def database(self):
        with tempfile.TemporaryDirectory() as home:
            db = SessionDB(Path(home) / "state.db")
            try:
                yield db
            finally:
                db.close()

    def add(self, db, sid, parent=None, *, reason=None, config=None, key="room", started=100):
        db.create_session(sid, source="telegram", parent_session_id=parent,
                          model_config=config or {}, session_key=key)
        db.append_message(sid, role="user", content="Synthetic " + sid)
        db._conn.execute("UPDATE sessions SET started_at=?, last_activity_at=?, ended_at=?, end_reason=? WHERE id=?",
                         (started, started, started + 1 if reason else None, reason, sid))
        db._conn.commit()

    def test_independent_edges_preserve_continuations(self):
        # The original diagnostic used symbolic bg_old/bg_newer_sibling IDs.
        # Legacy recognition here uses the exact production grammar, not a broad prefix.
        for shape in ("legacy_background", "compression_then_background", "compression_background_sibling"):
            for modern in (False, True):
                for key in (None, "room"):
                    with self.subTest(shape=shape, modern=modern, key=key), self.database() as db:
                        self.add(db, "root", reason="session_reset" if shape == "legacy_background" else "compression")
                        target = "root"
                        if shape != "legacy_background":
                            self.add(db, "tip", "root", started=200)
                            target = "tip"
                        parent = "tip" if shape == "compression_then_background" else "root"
                        bg = "arbitrary-modern-job" if modern else "bg_184523_ca0d94"
                        self.add(db, bg, parent, key=key, started=300,
                                 reason=None if shape == "compression_background_sibling" else "agent_close",
                                 config={"_background_from": parent} if modern else {})
                        self.assertEqual(db.get_compression_tip("root"), target)
                        self.assertEqual(db.resolve_resume_session_id("root"), target)
                        self.assertEqual(db.resolve_resume_session_id(bg), bg)
                        self.assertTrue(db.is_explicit_fork_child(bg))
                        if shape == "compression_background_sibling":
                            self.assertEqual(db.find_live_compression_child("root")["id"], "tip")
                        # Real compression publication must preserve the job's own continuation.
                        watermark = db.get_active_message_watermark(bg)
                        db.publish_compression_child(
                            parent_session_id=bg, child_session_id="job-tip", source="telegram",
                            messages=[{"role": "user", "content": "Synthetic compacted job"}],
                            watermark=watermark, require_compression_lease=False)
                        self.assertEqual(db.resolve_resume_session_id(bg), "job-tip")
                        self.assertEqual(db.resolve_resume_session_id("root"), target)
                        self.assertFalse(db.is_explicit_fork_child("job-tip"))

        for marker in ("_delegate_from", "_branched_from", "_background_from", "_reset_from"):
            with self.subTest(shape="inherited_marker_compression", marker=marker), self.database() as db:
                cfg = {marker: "external-parent"}
                self.add(db, "bg_explicit", reason="compression", config=cfg)
                self.add(db, "bg_compressed_tip", "bg_explicit", reason="compression", config=cfg, started=200)
                self.add(db, "double-tip", "bg_compressed_tip", config=cfg, started=300)
                self.assertEqual(db.get_compression_tip("bg_explicit"), "double-tip")
                self.assertEqual(db.resolve_resume_session_id("bg_explicit"), "double-tip")

        for sid in ("legacy_continuation", "bg_old", "bg_newer_sibling", "bg_184523_CA0D94",
                    "bg_244523_ca0d94", "bg_186023_ca0d94", "bg_184560_ca0d94",
                    "bg_184523_ca0d94_extra", "bg_184523_ca0d9", "bgX184523_ca0d94"):
            with self.subTest(shape="non_matching_legacy_control", sid=sid), self.database() as db:
                self.add(db, "root")
                self.add(db, sid, "root", key=None, started=200)
                self.assertEqual(db.resolve_resume_session_id("root"), sid)
                self.assertFalse(db.is_explicit_fork_child(sid))

        for marker in ("_delegate_from", "_branched_from", "_background_from", "_reset_from"):
            with self.subTest(shape="inherited_marker_legacy_walk", marker=marker), self.database() as db:
                self.add(db, "root", config={marker: "external-parent"})
                self.add(db, "continuation", "root", config={marker: "external-parent"}, started=200)
                self.assertEqual(db.resolve_resume_session_id("root"), "continuation")

        for key in (None, "room"):
            with self.subTest(shape="orphan_recovery_background_only", key=key), self.database() as db:
                self.add(db, "root", reason="compression")
                self.add(db, "bg_184523_ca0d94", "root", key=key, started=200)
                self.assertIsNone(db.find_live_compression_child("root"))
                self.assertTrue(db.reopen_orphaned_compression_session("root"))
                self.assertEqual(db.resolve_resume_session_id("root"), "root")

        with self.subTest(shape="legacy_reset_and_tool_controls"), self.database() as db:
            self.add(db, "reset", reason="session_reset")
            self.add(db, "new-conversation", "reset", started=200)
            self.assertEqual(db.resolve_resume_session_id("reset"), "reset")
            self.add(db, "root", reason="compression")
            db.create_session("tool-child", source="tool", parent_session_id="root")
            db.append_message("tool-child", role="user", content="Synthetic tool")
            self.assertEqual(db.get_compression_tip("root"), "root")
            self.assertEqual(db.resolve_resume_session_id("root"), "root")
            self.assertEqual(db.resolve_resume_session_id("missing"), "missing")

        for marker in ("_delegate_from", "_branched_from", "_background_from", "_reset_from"):
            with self.subTest(shape="direct_marker_control", marker=marker), self.database() as db:
                self.add(db, "root", reason="compression")
                self.add(db, "fork", "root", config={marker: "root"}, started=200)
                self.assertEqual(db.get_compression_tip("root"), "root")
                self.assertEqual(db.resolve_resume_session_id("root"), "root")

    def test_gateway_background_creation_and_resume(self):
        from gateway.config import GatewayConfig, Platform
        from gateway.session import SessionStore, AsyncSessionStore
        from gateway.platforms.event import MessageEvent
        from tests.gateway.test_background_command import _make_runner, _make_event
        from run_agent import AIAgent
        with tempfile.TemporaryDirectory() as home:
            with patch("hermes_state.DEFAULT_DB_PATH", Path(home) / "state.db"):
                store = SessionStore(sessions_dir=Path(home) / "sessions", config=GatewayConfig())
            db = store._db
            runner = _make_runner()
            runner.session_store = store
            runner._async_session_store = AsyncSessionStore(store)
            runner._session_db = AsyncSessionDB(db)
            event = _make_event("/bg synthetic request")
            parent = store.get_or_create_session(event.source)
            store.append_to_transcript(parent.session_id, {"role": "user", "content": "Parent context"})
            adapter = MagicMock()
            adapter.send = AsyncMock()
            adapter.extract_media.side_effect = lambda text: ([], text)
            adapter.extract_images.side_effect = lambda text: ([], text)
            runner.adapters[Platform.TELEGRAM] = adapter
            runner._resolve_session_agent_runtime = MagicMock(return_value=("test/model", {
                "api_key": "synthetic-not-a-secret", "base_url": "http://127.0.0.1:1/v1", "provider": "openai"}))
            runner._resolve_turn_toolsets = MagicMock(return_value=([], []))
            runner._resolve_turn_agent_config = lambda prompt, model, runtime: {"model": model, "runtime": runtime}
            runner._refresh_fallback_model = lambda: None
            runner._get_system_prompt_for_channel = lambda *args, **kwargs: None
            seen = []

            def fake_model_turn(agent, *args, **kwargs):
                # Real AIAgent construction and real lazy persistence ran before this seam.
                row = db.get_session(agent.session_id)
                seen.append(row)
                db.append_message(agent.session_id, role="assistant", content="Synthetic job result")
                return {"final_response": "Synthetic job result", "messages": []}

            async def scenario():
                with patch("gateway.run._load_gateway_config", return_value={}), \
                     patch.object(AIAgent, "run_conversation", fake_model_turn):
                    await runner._handle_background_command(event)
                    await asyncio.gather(*runner._background_tasks)
                self.assertEqual(len(seen), 1)
                child = seen[0]
                cfg = child["model_config"]
                if isinstance(cfg, str):
                    cfg = json.loads(cfg)
                self.assertEqual(cfg.get("_background_from"), parent.session_id)
                self.assertIsNone(child["session_key"])
                self.assertEqual(db.resolve_resume_session_id(parent.session_id), parent.session_id)
                # Full native handler -> authorization -> switch on a real temporary store.
                store.switch_session(parent.session_key, child["id"])
                reply = await runner._handle_resume_command(MessageEvent(
                    text="/resume " + parent.session_id, source=event.source))
                self.assertTrue(reply)
                self.assertEqual(store.get_or_create_session(event.source).session_id, parent.session_id)
                self.assertEqual(db.resolve_resume_session_id(parent.session_id), parent.session_id)
                # Explicit job resume remains available even after adoption changes its routing key.
                await runner._handle_resume_command(MessageEvent(text="/resume " + child["id"], source=event.source))
                self.assertEqual(store.get_or_create_session(event.source).session_id, child["id"])

            try:
                asyncio.run(scenario())
            finally:
                db.close()
