"""Tests for the extracted GatewayKanbanWatchersMixin (god-file Phase 3).

The kanban watcher loops were lifted out of gateway/run.py into a mixin that
GatewayRunner inherits. These tests confirm the mixin exposes the methods and
that GatewayRunner picks them up via the MRO (behavior-neutral relocation).
"""

from __future__ import annotations

import inspect

from gateway.kanban_watchers import GatewayKanbanWatchersMixin

KANBAN_METHODS = [
    "_kanban_notifier_watcher",
    "_kanban_dispatcher_watcher",
    "_kanban_advance",
    "_kanban_unsub",
    "_kanban_rewind",
    "_deliver_kanban_artifacts",
]


def test_mixin_defines_kanban_methods():
    for m in KANBAN_METHODS:
        assert hasattr(GatewayKanbanWatchersMixin, m), f"mixin missing {m}"




def test_gateway_resolves_no_progress_timeout_through_the_db_parser():
    """The gateway deliberately owns no validity rules of its own — the DB
    layer is the single definition of a valid progress bound. Drive the real
    ``kanban_db`` parser and assert exact resolved values for every branch so
    a drifting second copy (or a swallowed fallback) cannot hide here."""
    from gateway.kanban_watchers import _resolve_no_progress_timeout_seconds
    from hermes_cli import kanban_db as real_kb

    default = real_kb.DEFAULT_NO_PROGRESS_TIMEOUT_SECONDS
    assert default == 45 * 60
    cases = [
        ({}, default),                                        # unset
        ({"no_progress_timeout_seconds": None}, default),     # explicit null
        ({"no_progress_timeout_seconds": 0}, 0),              # disabled
        ({"no_progress_timeout_seconds": 60}, 60),            # the minimum
        ({"no_progress_timeout_seconds": 3600}, 3600),        # ordinary
        ({"no_progress_timeout_seconds": "600"}, 600),        # YAML string
        ({"no_progress_timeout_seconds": True}, default),     # bool is not 1s
        ({"no_progress_timeout_seconds": False}, default),
        ({"no_progress_timeout_seconds": 45}, default),       # units slip
        ({"no_progress_timeout_seconds": 59}, default),
        ({"no_progress_timeout_seconds": -1}, default),
        ({"no_progress_timeout_seconds": "abc"}, default),
        ({"no_progress_timeout_seconds": []}, default),
        ({"no_progress_timeout_seconds": float("inf")}, default),
        ({"no_progress_timeout_seconds": float("nan")}, default),
    ]
    for cfg, expected in cases:
        assert _resolve_no_progress_timeout_seconds(cfg, real_kb) == expected, cfg


def test_gateway_dispatcher_passes_the_no_progress_timeout_to_every_tick():
    """Source-level contract for the watcher loop: the resolved bound is
    threaded into ``dispatch_once`` so a gateway tick cannot behave
    differently from a CLI or daemon tick."""
    from gateway import kanban_watchers

    src = inspect.getsource(kanban_watchers.GatewayKanbanWatchersMixin._kanban_dispatcher_watcher)
    assert "_resolve_no_progress_timeout_seconds(" in src
    assert "no_progress_timeout_seconds=no_progress_timeout_seconds" in src


def test_no_progress_deferred_is_delivered_but_never_wakes_the_creator():
    """A deferred receipt must reach subscribers (dropping it would let a
    held card look healthy), but the task is still running, so it is not a
    terminal outcome and must not synthesise a creator turn the way
    ``crashed`` / ``timed_out`` do."""
    from gateway import kanban_watchers

    src = inspect.getsource(kanban_watchers.GatewayKanbanWatchersMixin._kanban_notifier_watcher)
    terminal = next(line for line in src.splitlines() if "TERMINAL_KINDS = (" in line)
    wake = next(line for line in src.splitlines() if "_WAKE_KINDS = (" in line)
    assert '"no_progress_deferred"' in terminal
    assert '"no_progress_deferred"' not in wake
    assert 'kind == "no_progress_deferred"' in src
