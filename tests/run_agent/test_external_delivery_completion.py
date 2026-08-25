"""End-to-end contract for a tool-owned external delivery terminal turn."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _tool_defs() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "description": "terminal tool",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]


def _tool_call_response() -> SimpleNamespace:
    tool_call = SimpleNamespace(
        id="call_delivery",
        type="function",
        function=SimpleNamespace(
            name="terminal",
            arguments=json.dumps({"command": "python3 deliver.py"}),
        ),
    )
    message = SimpleNamespace(content="", tool_calls=[tool_call])
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=message, finish_reason="tool_calls"),
        ],
        model="test/model",
        usage=None,
    )


def _mixed_tool_call_response() -> SimpleNamespace:
    calls = [
        SimpleNamespace(
            id="call_delivery",
            type="function",
            function=SimpleNamespace(
                name="terminal",
                arguments=json.dumps({"command": "python3 deliver.py"}),
            ),
        ),
        SimpleNamespace(
            id="call_other",
            type="function",
            function=SimpleNamespace(
                name="terminal",
                arguments=json.dumps({"command": "python3 inspect.py"}),
            ),
        ),
    ]
    message = SimpleNamespace(content="", tool_calls=calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
        model="test/model",
        usage=None,
    )


def _text_response(content: str) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        model="test/model",
        usage=None,
    )


def _receipt_path(hermes_home: Path, session_id: str, turn_id: str) -> Path:
    identity = hashlib.sha256(
        f"{session_id}\0{turn_id}\0call_delivery".encode("utf-8")
    ).hexdigest()
    return hermes_home / "runtime" / "external-delivery" / f"{identity}.json"


def _make_agent(tmp_path: Path) -> AIAgent:
    hermes_home = tmp_path / ".hermes"
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            session_id="external-delivery-test",
            api_key="test-key",
            base_url="https://example.invalid/v1",
            provider="openai-compat",
            model="test/model",
            max_iterations=5,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "stable test prompt"
    agent._use_prompt_caching = False
    agent._disable_streaming = True
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent._session_db = None
    agent._session_json_enabled = False
    return agent


def test_successful_external_delivery_stops_before_second_model_call_and_pre_verify(
    tmp_path: Path,
    monkeypatch,
) -> None:
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    agent = _make_agent(tmp_path)
    agent.client.chat.completions.create.side_effect = [
        _tool_call_response(),
        _text_response("THIS SECOND MODEL CALL MUST NOT HAPPEN"),
    ]

    receipt = {
        "protocol": "hermes.external_delivery",
        "version": 1,
        "status": "complete",
        "target": "telegram:-1003905769435:8372",
        "message_ids": ["26477"],
        "content_sha256": "a" * 64,
    }

    def fake_terminal_dispatch(function_name, function_args, task_id, **kwargs):
        assert function_name == "terminal"
        path = _receipt_path(
            hermes_home,
            agent.session_id,
            agent._current_turn_id,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt), encoding="utf-8")
        return json.dumps(
            {
                "output": "delivered\n" + ("x" * 150_000),
                "exit_code": 0,
                "error": None,
            }
        )

    pre_verify = MagicMock(return_value=None)
    with (
        patch("run_agent.handle_function_call", side_effect=fake_terminal_dispatch),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_sync_external_memory_for_turn") as external_memory_sync,
        patch(
            "hermes_cli.plugins.has_hook",
            side_effect=lambda name: name == "pre_verify",
        ),
        patch(
            "hermes_cli.plugins.get_pre_verify_continue_message",
            pre_verify,
        ),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        result = agent.run_conversation("Scanne les inbox et livre les dossiers.")

    assert agent.client.chat.completions.create.call_count == 1
    pre_verify.assert_not_called()
    external_memory_sync.assert_not_called()
    assert result["turn_exit_reason"] == "external_delivery_complete"
    assert result["completed"] is True
    assert result["final_response"] == "NO_REPLY"
    assert result["delivery_already_sent"] is True
    assert result["external_deliveries"] == [
        {**receipt, "tool_call_id": "call_delivery"}
    ]
    assert [message["role"] for message in result["messages"][-3:]] == [
        "assistant",
        "tool",
        "assistant",
    ]
    assert result["messages"][-1]["content"] == "NO_REPLY"
    assert not _receipt_path(
        hermes_home,
        agent.session_id,
        agent._current_turn_id,
    ).exists()


def test_external_delivery_in_mixed_batch_is_consumed_but_model_interprets_all_results(
    tmp_path: Path,
    monkeypatch,
) -> None:
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    agent = _make_agent(tmp_path)
    agent.client.chat.completions.create.side_effect = [
        _mixed_tool_call_response(),
        _text_response("interpreted delivery and inspection results"),
    ]
    receipt = {
        "protocol": "hermes.external_delivery",
        "version": 1,
        "status": "complete",
        "target": "telegram:-1003905769435:8372",
        "message_ids": ["26477"],
        "content_sha256": "d" * 64,
    }

    def fake_terminal_dispatch(function_name, function_args, task_id, **kwargs):
        assert function_name == "terminal"
        if function_args["command"] == "python3 deliver.py":
            path = _receipt_path(
                hermes_home,
                agent.session_id,
                agent._current_turn_id,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(receipt), encoding="utf-8")
            output = "delivered"
        else:
            output = "inspection result"
        return json.dumps({"output": output, "exit_code": 0, "error": None})

    with (
        patch("run_agent.handle_function_call", side_effect=fake_terminal_dispatch),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("hermes_cli.plugins.has_hook", return_value=False),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        result = agent.run_conversation("Deliver and inspect in one batch.")

    assert agent.client.chat.completions.create.call_count == 2
    assert result["final_response"] == "interpreted delivery and inspection results"
    assert result["turn_exit_reason"] == "text_response(finish_reason=stop)"
    assert "delivery_already_sent" not in result
    assert not _receipt_path(
        hermes_home,
        agent.session_id,
        agent._current_turn_id,
    ).exists()


@pytest.mark.parametrize(
    ("receipt", "exit_code"),
    [
        ({"protocol": "hermes.external_delivery"}, 0),
        (
            {
                "protocol": "hermes.external_delivery",
                "version": 1,
                "status": "complete",
                "target": "telegram:-1003905769435:8372",
                "message_ids": ["26477"],
                "content_sha256": "c" * 64,
            },
            1,
        ),
    ],
)
def test_invalid_or_failed_external_delivery_does_not_silence_the_turn(
    tmp_path: Path,
    monkeypatch,
    receipt: dict,
    exit_code: int,
) -> None:
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    agent = _make_agent(tmp_path)
    agent.client.chat.completions.create.side_effect = [
        _tool_call_response(),
        _text_response("normal continuation"),
    ]

    def fake_terminal_dispatch(function_name, function_args, task_id, **kwargs):
        path = _receipt_path(
            hermes_home,
            agent.session_id,
            agent._current_turn_id,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt), encoding="utf-8")
        return json.dumps(
            {
                "output": "delivery attempt",
                "exit_code": exit_code,
                "error": None if exit_code == 0 else "failed",
            }
        )

    with (
        patch("run_agent.handle_function_call", side_effect=fake_terminal_dispatch),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("hermes_cli.plugins.has_hook", return_value=False),
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
    ):
        result = agent.run_conversation("Scanne les inbox et livre les dossiers.")

    assert agent.client.chat.completions.create.call_count == 2
    assert result["final_response"] == "normal continuation"
    assert result["turn_exit_reason"] == "text_response(finish_reason=stop)"
    assert "delivery_already_sent" not in result
    assert not _receipt_path(
        hermes_home,
        agent.session_id,
        agent._current_turn_id,
    ).exists()
