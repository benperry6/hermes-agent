"""Ephemeral control channel for tool-owned external response delivery.

Some deterministic terminal workflows deliver the complete user-visible answer
themselves.  They can atomically write a versioned receipt to the per-call path
in ``HERMES_TURN_RECEIPT_FILE``.  The conversation loop consumes that receipt
only after the full tool batch has completed and only when the matching
terminal result succeeded.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from pathlib import Path
from typing import Any, Iterable

from hermes_constants import get_hermes_home

EXTERNAL_DELIVERY_PROTOCOL = "hermes.external_delivery"
EXTERNAL_DELIVERY_VERSION = 1
EXTERNAL_DELIVERY_ENV = "HERMES_TURN_RECEIPT_FILE"
TERMINAL_RESULT_METADATA_KEY = "_external_delivery_terminal_result"

_RECEIPT_KEYS = frozenset(
    {
        "protocol",
        "version",
        "status",
        "target",
        "message_ids",
        "content_sha256",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def external_delivery_receipt_path(
    session_id: str,
    turn_id: str,
    tool_call_id: str,
) -> Path:
    """Return a profile-safe, collision-resistant path for one tool call."""
    identity = hashlib.sha256(
        f"{session_id}\0{turn_id}\0{tool_call_id}".encode("utf-8")
    ).hexdigest()
    return (
        get_hermes_home()
        / "runtime"
        / "external-delivery"
        / f"{identity}.json"
    )


def prepare_external_delivery_receipt_path(
    session_id: str,
    turn_id: str,
    tool_call_id: str,
) -> Path:
    """Create the receipt directory and remove a stale same-call artifact."""
    path = external_delivery_receipt_path(session_id, turn_id, tool_call_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    return path


def inject_external_delivery_receipt_env(command: str, path: Path) -> str:
    """Export the per-call receipt path for the full foreground shell command."""
    return (
        f"export {EXTERNAL_DELIVERY_ENV}={shlex.quote(str(path))}; "
        f"{command}"
    )


def validate_external_delivery_receipt(payload: Any) -> dict[str, Any] | None:
    """Return a normalized receipt only for the exact version-1 contract."""
    if not isinstance(payload, dict) or frozenset(payload) != _RECEIPT_KEYS:
        return None
    if payload.get("protocol") != EXTERNAL_DELIVERY_PROTOCOL:
        return None
    if payload.get("version") != EXTERNAL_DELIVERY_VERSION:
        return None
    if payload.get("status") != "complete":
        return None

    target = payload.get("target")
    message_ids = payload.get("message_ids")
    content_sha256 = payload.get("content_sha256")
    if not isinstance(target, str) or not target.strip() or len(target) > 512:
        return None
    if (
        not isinstance(message_ids, list)
        or not message_ids
        or any(
            not isinstance(message_id, str)
            or not message_id.strip()
            or len(message_id) > 256
            for message_id in message_ids
        )
    ):
        return None
    if (
        not isinstance(content_sha256, str)
        or not _SHA256_RE.fullmatch(content_sha256)
    ):
        return None

    return {
        "protocol": EXTERNAL_DELIVERY_PROTOCOL,
        "version": EXTERNAL_DELIVERY_VERSION,
        "status": "complete",
        "target": target,
        "message_ids": list(message_ids),
        "content_sha256": content_sha256,
    }


def capture_terminal_result_metadata(
    tool_name: str,
    content: Any,
) -> dict[str, bool] | None:
    """Capture terminal success before output persistence can replace JSON."""
    if tool_name != "terminal" or not isinstance(content, str):
        return None
    try:
        result = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(result, dict):
        return None
    return {
        "success": (
            result.get("exit_code") == 0
            and result.get("error") is None
        )
    }


def _successful_terminal_tool_call_ids(
    tool_calls: Iterable[Any],
    messages: list[dict[str, Any]],
) -> set[str]:
    terminal_ids = {
        str(getattr(tool_call, "id", "") or "")
        for tool_call in tool_calls
        if getattr(getattr(tool_call, "function", None), "name", None) == "terminal"
        and getattr(tool_call, "id", None)
    }
    successful: set[str] = set()
    seen_results: set[str] = set()
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        tool_call_id = str(message.get("tool_call_id") or "")
        if tool_call_id not in terminal_ids or tool_call_id in seen_results:
            continue
        seen_results.add(tool_call_id)
        metadata = message.get(TERMINAL_RESULT_METADATA_KEY)
        if (
            isinstance(metadata, dict)
            and frozenset(metadata) == {"success"}
            and isinstance(metadata.get("success"), bool)
        ):
            if metadata["success"]:
                successful.add(tool_call_id)
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            result = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            continue
        if (
            isinstance(result, dict)
            and result.get("exit_code") == 0
            and result.get("error") is None
        ):
            successful.add(tool_call_id)
    return successful


def consume_external_delivery_receipts(
    *,
    session_id: str,
    turn_id: str,
    tool_calls: Iterable[Any],
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Consume valid receipts for successful terminal calls in this batch.

    Every candidate path is deleted whether parsing succeeds or fails.  A
    receipt paired with a failed/non-terminal tool result can therefore never
    leak into a later turn.
    """
    calls = list(tool_calls)
    successful_ids = _successful_terminal_tool_call_ids(calls, messages)
    receipts: list[dict[str, Any]] = []
    for tool_call in calls:
        tool_call_id = str(getattr(tool_call, "id", "") or "")
        if not tool_call_id:
            continue
        path = external_delivery_receipt_path(session_id, turn_id, tool_call_id)
        if not path.exists():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            receipt = validate_external_delivery_receipt(raw)
            if receipt is not None and tool_call_id in successful_ids:
                receipt["tool_call_id"] = tool_call_id
                receipts.append(receipt)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            pass
        finally:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
    return receipts


__all__ = [
    "EXTERNAL_DELIVERY_ENV",
    "EXTERNAL_DELIVERY_PROTOCOL",
    "EXTERNAL_DELIVERY_VERSION",
    "TERMINAL_RESULT_METADATA_KEY",
    "capture_terminal_result_metadata",
    "consume_external_delivery_receipts",
    "external_delivery_receipt_path",
    "inject_external_delivery_receipt_env",
    "prepare_external_delivery_receipt_path",
    "validate_external_delivery_receipt",
]
