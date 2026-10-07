"""Published clarification is retained, not orphaned by the age of its work.

Exercise existing RAM budget and cancel/watch transitions without making
INPUT_REQUIRED an irreversible terminal state or adding storage.
"""
import pytest
from plugins.platforms.a2a import protocol


@pytest.mark.parametrize("case", ["published", "unpublished", "publication-race"])
def test_old_work_does_not_erase_a_published_clarification(monkeypatch, case):
    now = [100.0]
    monkeypatch.setattr(protocol.time, "time", lambda: now[0])
    store = protocol.TaskStore()
    store.create("old-task", "same-context", "inert-peer")
    store.set_state("old-task", protocol.STATE_WORKING)
    now[0] += 7200
    if case == "unpublished":
        assert store.fail_orphans(3600) == ["old-task"]
        assert store.get("old-task")["state"] == protocol.STATE_FAILED
        return
    if case == "published":
        store.complete("old-task", protocol.STATE_INPUT_REQUIRED, "Which inert option?")
    else:
        # Sweep selects the old work, then publication wins before completion.
        native_complete = store.complete
        def completing(task_id, state, reply="", **kw):
            if state == protocol.STATE_FAILED:
                native_complete(task_id, protocol.STATE_INPUT_REQUIRED, "Which inert option?")
            return native_complete(task_id, state, reply, **kw)
        monkeypatch.setattr(store, "complete", completing)
    assert store.fail_orphans(3600) == []
    now[0] += 7200
    assert store.fail_orphans(3600) == []
    task = protocol.TaskStore.to_task(store.get("old-task"))
    assert task["id"] == "old-task"
    assert task["status"]["state"] == protocol.STATE_INPUT_REQUIRED
    assert task["status"]["message"]["parts"][0]["text"] == "Which inert option?"
    waiting = store.watch("old-task")
    assert not waiting.done()  # Clarification stays resumable/cancelable.
    assert store.complete("old-task", protocol.STATE_CANCELED, "") is not None
    assert waiting.result()[0] == protocol.STATE_CANCELED
    assert store.complete("old-task", protocol.STATE_COMPLETED, "late") is None


def test_published_clarifications_share_the_existing_bounded_reply_budget():
    store = protocol.TaskStore()
    store.create("still-working", "work", "inert-peer")
    store.set_state("still-working", protocol.STATE_WORKING)
    for n in range(store._MAX_TERMINAL + 1):
        task_id = f"reply-{n}"
        store.create(task_id, "ctx", "inert-peer")
        store.complete(task_id, protocol.STATE_INPUT_REQUIRED, f"question-{n}")
        if n == 0:
            evicted_watch = store.watch(task_id)
            assert not evicted_watch.done()
    assert store.get("reply-0") is None
    assert "reply-0" not in store._watchers
    assert store.get(f"reply-{store._MAX_TERMINAL}")["reply"] == f"question-{store._MAX_TERMINAL}"
    assert len(store.list(page_size=100, with_total=True)[0]) == 100
    assert store.list(with_total=True)[2] == store._MAX_TERMINAL + 1  # working + bounded replies
    assert store.get("still-working")["state"] == protocol.STATE_WORKING
