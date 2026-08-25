"""Behavior contracts for the external-delivery receipt side channel."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import tools.terminal_tool as terminal_tool

from agent.external_delivery import (
    TERMINAL_RESULT_METADATA_KEY,
    capture_terminal_result_metadata,
    consume_external_delivery_receipts,
    external_delivery_receipt_path,
    inject_external_delivery_receipt_env,
    validate_external_delivery_receipt,
)


def _tool_call(call_id: str = "call-1", name: str = "terminal") -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name),
    )


def _receipt() -> dict:
    return {
        "protocol": "hermes.external_delivery",
        "version": 1,
        "status": "complete",
        "target": "telegram:-1001:8372",
        "message_ids": ["m-1"],
        "content_sha256": "b" * 64,
    }


def _tool_result(call_id: str = "call-1", exit_code: int = 0) -> dict:
    return {
        "role": "tool",
        "name": "terminal",
        "tool_call_id": call_id,
        "content": json.dumps(
            {
                "output": "done",
                "exit_code": exit_code,
                "error": None if exit_code == 0 else "failed",
            }
        ),
    }


def test_receipt_paths_are_isolated_by_tool_call(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    first = external_delivery_receipt_path("session", "turn", "call-1")
    second = external_delivery_receipt_path("session", "turn", "call-2")
    assert first != second
    assert first.parent == second.parent == tmp_path / "runtime" / "external-delivery"


def test_valid_receipt_requires_matching_successful_terminal_result(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = external_delivery_receipt_path("session", "turn", "call-1")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_receipt()), encoding="utf-8")

    receipts = consume_external_delivery_receipts(
        session_id="session",
        turn_id="turn",
        tool_calls=[_tool_call()],
        messages=[_tool_result()],
    )

    assert receipts == [{**_receipt(), "tool_call_id": "call-1"}]
    assert not path.exists()


def test_failed_terminal_result_invalidates_and_consumes_receipt(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = external_delivery_receipt_path("session", "turn", "call-1")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_receipt()), encoding="utf-8")

    receipts = consume_external_delivery_receipts(
        session_id="session",
        turn_id="turn",
        tool_calls=[_tool_call()],
        messages=[
            _tool_result(exit_code=0),
            _tool_result(exit_code=1),
        ],
    )

    assert receipts == []
    assert not path.exists()


def test_captured_terminal_status_survives_persisted_output(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = external_delivery_receipt_path("session", "turn", "call-1")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_receipt()), encoding="utf-8")
    raw_result = json.dumps(
        {"output": "x" * 200_000, "exit_code": 0, "error": None}
    )
    persisted_message = {
        "role": "tool",
        "name": "terminal",
        "tool_call_id": "call-1",
        "content": "<persisted-output>large terminal result</persisted-output>",
        TERMINAL_RESULT_METADATA_KEY: capture_terminal_result_metadata(
            "terminal",
            raw_result,
        ),
    }

    receipts = consume_external_delivery_receipts(
        session_id="session",
        turn_id="turn",
        tool_calls=[_tool_call()],
        messages=[persisted_message],
    )

    assert receipts == [{**_receipt(), "tool_call_id": "call-1"}]
    assert not path.exists()


def test_malformed_and_non_terminal_receipts_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = external_delivery_receipt_path("session", "turn", "call-1")
    path.parent.mkdir(parents=True)
    path.write_text('{"protocol":"hermes.external_delivery"}', encoding="utf-8")

    receipts = consume_external_delivery_receipts(
        session_id="session",
        turn_id="turn",
        tool_calls=[_tool_call(name="web_search")],
        messages=[_tool_result()],
    )

    assert receipts == []
    assert not path.exists()
    assert validate_external_delivery_receipt({**_receipt(), "extra": True}) is None
    assert validate_external_delivery_receipt(
        {**_receipt(), "content_sha256": "INVALID"}
    ) is None


def test_receipt_environment_export_covers_the_full_command(tmp_path):
    path = tmp_path / "receipt with spaces.json"
    command = inject_external_delivery_receipt_env(
        "python3 deliver.py && python3 verify.py",
        path,
    )
    assert command.startswith("export HERMES_TURN_RECEIPT_FILE=")
    assert command.endswith("; python3 deliver.py && python3 verify.py")


def test_dispatch_and_foreground_terminal_preserve_receipt_identity(tmp_path, monkeypatch):
    calls = []

    class FakeEnv:
        env = {}

        def execute(self, command, **kwargs):
            calls.append((command, kwargs))
            return {"output": "ok", "returncode": 0}

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(terminal_tool, "_active_environments", {"task": FakeEnv()})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: {"env_type": "local", "cwd": "/default", "timeout": 60},
    )
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda command, env_type, **kwargs: {"approved": True},
    )

    from model_tools import handle_function_call

    with patch("hermes_cli.plugins.has_hook", return_value=False):
        result = json.loads(
            handle_function_call(
                "terminal",
                {"command": "python3 deliver.py"},
                task_id="task",
                session_id="session",
                turn_id="turn",
                tool_call_id="call-delivery",
                skip_pre_tool_call_hook=True,
                skip_tool_request_middleware=True,
                skip_tool_execution_middleware=True,
            )
        )

    assert result["exit_code"] == 0
    executed, kwargs = calls[0]
    assert executed.startswith("export HERMES_TURN_RECEIPT_FILE=")
    exported = Path(executed.split("=", 1)[1].split("; ", 1)[0])
    assert exported == external_delivery_receipt_path(
        "session", "turn", "call-delivery"
    )
    assert kwargs == {"timeout": 60, "cwd": "/default", "bounded_capture": True}


def test_structured_external_delivery_suppresses_transformed_marker():
    from gateway.response_filters import is_intentional_silence_agent_result

    result = {
        "failed": False,
        "completed": True,
        "turn_exit_reason": "external_delivery_complete",
        "delivery_already_sent": True,
        "external_deliveries": [{"protocol": "hermes.external_delivery"}],
    }
    assert is_intentional_silence_agent_result(result, "transformed text")
