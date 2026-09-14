"""File previews must not cancel a live SessionDB's POSIX locks."""
import asyncio
import os
from pathlib import Path

import pytest


@pytest.mark.linux_only
@pytest.mark.parametrize("route", ["file", "folder", "desktop"])
def test_preview_preserves_live_database_locks(tmp_path, route):
    from hermes_state import SessionDB
    from agent.context_references import parse_context_references, _expand_path_reference
    from hermes_cli.web_routers.files import fs_read_text
    from fastapi import HTTPException

    path = tmp_path / "state.db"
    text = tmp_path / "normal.txt"
    text.write_text("ordinary readable text", encoding="utf-8")
    db = SessionDB(path)
    try:
        db.create_session("preview-test", "cli")
        db.append_message("preview-test", "user", "before preview")
        inode = path.stat().st_ino

        def main_locks():
            return [line.split(": ", 1)[1] for line in Path("/proc/locks").read_text().splitlines()
                    if f":{inode} " in line and f" {os.getpid()} " in line]

        before = main_locks()
        assert before, "SessionDB must have acquired its main-file locks"
        if route == "desktop":
            with pytest.raises(HTTPException) as refusal:
                asyncio.run(fs_read_text(str(path)))
            assert refusal.value.status_code == 409
            assert asyncio.run(fs_read_text(str(text)))["text"] == "ordinary readable text"
        else:
            target = path if route == "file" else tmp_path
            ref = parse_context_references(f"@{route}:{target}")[0]
            warning, block = _expand_path_reference(ref, tmp_path.parent)
            assert warning is None
            assert "binary file" in block if route == "file" else "state.db" in block
            text_ref = parse_context_references(f"@file:{text}")[0]
            warning, block = _expand_path_reference(text_ref, tmp_path.parent)
            assert warning is None and "ordinary readable text" in block
        assert main_locks() == before
        db.append_message("preview-test", "assistant", "after preview")
        assert len(db.get_messages("preview-test")) == 2
    finally:
        db.close()
