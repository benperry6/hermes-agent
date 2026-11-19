"""A native in-place restore must invalidate the Desktop session list."""

from contextlib import closing
import errno
import sqlite3

from hermes_cli import backup_restore
from hermes_cli.backup_restore import _safe_restore_db
from hermes_state import SessionDB
from hermes_state_common import SESSIONS_LIST_REVISION_KEY, SESSIONS_LIST_REVISION_TRIGGERS
from tui_gateway import server


def _store(home, title=None):
    home.mkdir()
    path = home / "state.db"
    with closing(SessionDB(db_path=path)):
        if title is not None:
            with closing(sqlite3.connect(path)) as conn, conn:
                conn.execute(
                    "INSERT INTO sessions (id, source, started_at, title) VALUES ('sample', 'cli', 1, ?)",
                    (title,),
                )
    return path


def _watch(monkeypatch, home):
    monkeypatch.setattr(server, "_watcher_home", lambda: home)
    monkeypatch.setattr(server, "_served_profile_homes", [])
    monkeypatch.setattr(server, "_sessions_db_sig_cache", {})

    def no_space_for_staging(*_args, **_kwargs):
        raise OSError(errno.ENOSPC, "No room for a second full SQLite image")

    # These stores have equal epoch floors. The native page copy still fits.
    monkeypatch.setattr(backup_restore, "_safe_copy_db", no_space_for_staging)


def _revision(path):
    with closing(sqlite3.connect(path)) as conn:
        return server._session_list_revision(conn)


def _restore(source, live, title):
    stat = live.stat()
    original = source.read_bytes()
    assert _safe_restore_db(source, live) is True
    assert (live.stat().st_dev, live.stat().st_ino) == (stat.st_dev, stat.st_ino)
    assert source.read_bytes() == original
    with closing(sqlite3.connect(live)) as conn:
        assert conn.execute("SELECT title FROM sessions").fetchall() == [(title,)]


def test_in_place_restore_invalidates_an_equal_list_revision(tmp_path, monkeypatch):
    live = _store(tmp_path / "live", "Original")
    source = _store(tmp_path / "snapshot", "Restored")
    assert _revision(live) == _revision(source) == ("revision", "1")
    _watch(monkeypatch, live.parent)
    before = server._sessions_sig()

    _restore(source, live, "Restored")

    after = server._sessions_sig()
    assert after != before, f"Restored title on the same inode; watcher still reports {after!r}"


def test_legacy_restore_reopened_before_any_poll_invalidates_the_list(tmp_path, monkeypatch):
    live = _store(tmp_path / "live")
    source = _store(tmp_path / "snapshot", "Legacy title")
    with closing(sqlite3.connect(source)) as conn, conn:
        for name in SESSIONS_LIST_REVISION_TRIGGERS:
            conn.execute(f"DROP TRIGGER {name}")
        conn.execute("DELETE FROM state_meta WHERE key = ?", (SESSIONS_LIST_REVISION_KEY,))
    assert _revision(source) is None
    assert _revision(live) == ("revision", None)
    _watch(monkeypatch, live.parent)
    before = server._sessions_sig()

    _restore(source, live, "Legacy title")
    with closing(SessionDB(db_path=live)):
        assert _revision(live)[0] == "revision"
        # In particular, no digest was observed before the writable reopen.
        after = server._sessions_sig()

    assert after != before, f"Restored legacy rows after writable reopen; watcher still reports {after!r}"
