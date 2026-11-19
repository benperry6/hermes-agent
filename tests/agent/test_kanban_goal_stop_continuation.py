"""Native Goal continuation must receive a useful turn before terminal-stop coaching.

Real agent loop, stop gates, finalizer, SQLite history/board and lifecycle tools;
only model/judge transports and unrelated discovery are inert.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway import session_kanban
from hermes_cli import goals, kanban_db as kb
from hermes_cli.kanban_db_connect import connect_closing


@pytest.fixture
def worker(tmp_path, monkeypatch, request):
    params = getattr(request, 'param', {})
    goal_mode = params.get('goal_mode', True)
    import agent.kanban_stop
    import agent.turn_stop_gates
    import agent.turn_finalizer
    import hermes_cli.cli_single_query
    source = Path(__file__).resolve().parents[2]
    for module in (agent.kanban_stop, agent.turn_stop_gates, agent.turn_finalizer,
                   session_kanban, goals, hermes_cli.cli_single_query):
        assert module.__file__ is not None
        assert Path(module.__file__).resolve().is_relative_to(source), module.__file__
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tests.agent.test_run_agent import _make_tool_defs

    for key in tuple(os.environ):
        if key.startswith('HERMES_KANBAN_') or key == 'HERMES_DELEGATED_CHILD_CONTEXT':
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'home'))
    monkeypatch.setenv('TERMINAL_CWD', str(tmp_path))
    monkeypatch.setattr('hermes_cli.plugins.discover_plugins', lambda: None)
    monkeypatch.setattr('hermes_cli.plugins.PluginManager.discover_and_load', lambda self, force=False: None)
    monkeypatch.setattr('model_tools.get_tool_definitions', lambda **kw: _make_tool_defs(
        'read_file', 'write_file', 'kanban_request_review', 'kanban_complete', 'kanban_block'))
    monkeypatch.setattr('model_tools.check_toolset_requirements', lambda **kw: {})
    monkeypatch.setattr('agent.process_bootstrap.OpenAI', MagicMock())
    monkeypatch.setattr(session_kanban, '_watch_submitter', lambda *a: None)
    def no_network(*a, **kw):
        raise AssertionError('network disabled in behavioral fixture')
    monkeypatch.setattr('socket.socket.connect', no_network)
    monkeypatch.chdir(tmp_path)
    kb.init_db()
    with connect_closing() as conn:
        tid = kb.create_task(conn, title='Write a two-section handoff document',
                             body='Both sections must exist before review.', assignee='default',
                             goal_mode=goal_mode, goal_max_turns=3)
        task = kb.claim_task(conn, tid, claimer='test-worker')
    context = dict(task_id=tid, board='default', db=str(kb.kanban_db_path()),
                   run_id=task.current_run_id, claim_lock=task.claim_lock,
                   workspace=str(tmp_path), workspaces_root=str(tmp_path), profile='default',
                   branch=None, tenant=None, goal_mode=goal_mode, goal_max_turns=3,
                   max_runtime_seconds=None, skills=[], context='', goal_text=task.title + '\n' + task.body)
    frame = {'text': 'Write the handoff document.',
             'policy': {'source': 'kanban', 'kanban_json': json.dumps(context)}}
    # Binder owns process-global values; monkeypatch restores them after this private-exec seam.
    for key in ('HERMES_KANBAN_TASK', 'HERMES_KANBAN_BOARD', 'HERMES_KANBAN_DB',
                'HERMES_KANBAN_RUN_ID', 'HERMES_KANBAN_CLAIM_LOCK', 'HERMES_KANBAN_WORKSPACE',
                'HERMES_KANBAN_WORKSPACES_ROOT', 'HERMES_KANBAN_GOAL_MODE', 'HERMES_PROFILE',
                'HERMES_SESSION_SOURCE', 'HERMES_KANBAN_CLAIM_TTL_SECONDS'):
        monkeypatch.setenv(key, '')
    monkeypatch.setenv('HERMES_KANBAN_GOAL_MODE', params.get('inherited_goal', ''))
    session_kanban.bind_worker_context(frame)
    db = SessionDB(db_path=tmp_path / 'home' / 'state.db')
    agent = AIAgent(api_key='inert-test-key', base_url='https://inert.invalid/v1',
                    model='test/model', provider='openrouter', session_id='artifact-session',
                    session_db=db, quiet_mode=True, skip_context_files=True,
                    skip_memory=True, skip_background_review=True, max_iterations=10)
    agent.client = MagicMock()
    agent._cached_system_prompt = 'Write and verify the requested artifact.'
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.tool_delay = 0
    agent._stall_guards = False
    yield SimpleNamespace(agent=agent, frame=frame, context=context, db=db,
                          artifact=tmp_path / 'handoff.md')
    agent.release_clients()
    db.close()


def _run_caller(worker, caller, monkeypatch):
    if caller == 'sdk':
        result = session_kanban.run_worker_turns(worker.agent, worker.frame, [])
        assert result['completed'] and not result.get('failed')
        return
    import cli
    from hermes_cli.cli_single_query import _run_quiet_single_query, _run_single_query_mode

    surface = SimpleNamespace(agent=worker.agent, session_id=worker.agent.session_id,
        conversation_history=[], _last_turn_result=None, _single_query_mode=False,
        _claim_active_session=lambda *a, **kw: True,
        console=SimpleNamespace(print=lambda *a, **kw: None),
        _show_security_advisories=lambda: None, _print_exit_summary=lambda **kw: None)

    def chat(prompt, **kw):
        result = surface.agent.run_conversation(prompt, conversation_history=surface.conversation_history)
        surface.conversation_history = result['messages']
        surface._last_turn_result = result
        return result['final_response']

    surface.chat = chat
    monkeypatch.setattr(cli, '_should_seed_interactive', lambda *a, **kw: False)
    monkeypatch.setattr(cli, '_collect_query_images', lambda query, image: (query, []))
    monkeypatch.setattr(cli, '_collect_kanban_task_images', lambda images: [])
    monkeypatch.setattr(cli, '_finalize_single_query', lambda surface: None)
    monkeypatch.setattr('hermes_cli.quiet_single_query.continue_quiet_notify_completions', lambda *a, **kw: None)
    with pytest.raises(SystemExit) as exited:
        if caller == 'cli-quiet':
            _run_quiet_single_query(surface, worker.frame['text'])
        else:
            _run_single_query_mode(surface, worker.frame['text'], None, False, True)
    assert exited.value.code == 0


@pytest.mark.parametrize('caller', ['sdk', 'cli-quiet', 'cli-chat'])
def test_goal_receives_partial_then_continues_same_session(worker, monkeypatch, caller):
    from tests.agent.test_run_agent import _mock_response, _mock_tool_call

    agent, artifact = worker.agent, worker.artifact
    partial = 'Section one is written and verified; section two is not present.'
    second = 'Both sections are written and verified. Ready for review.'
    requests = []
    requests_at_continuation = []
    judged = []

    def judge(goal, response=None, last_response=None, **kw):
        response = response if response is not None else last_response
        judged.append(response)
        return ('continue' if response == partial else 'done', 'check both sections', False, None, False)

    monkeypatch.setattr(goals, 'judge_goal', judge)
    monkeypatch.setattr('tools.kanban_tools.judge_goal', judge)
    monkeypatch.setattr('agent.auxiliary_client.get_text_auxiliary_client', lambda *a, **kw: (object(), 'inert-judge'))
    script = [
        _mock_response(content='', finish_reason='tool_calls', tool_calls=[_mock_tool_call(
            'write_file', json.dumps({'path': str(artifact), 'content': '# First\nVerified first section.\n'}))]),
        _mock_response(content=partial),
        _mock_response(content='', finish_reason='tool_calls', tool_calls=[_mock_tool_call(
            'read_file', json.dumps({'path': str(artifact)}))]),
        _mock_response(content='', finish_reason='tool_calls', tool_calls=[_mock_tool_call(
            'write_file', json.dumps({'path': str(artifact), 'content': '# First\nVerified first section.\n# Second\nVerified second section.\n'}))]),
        _mock_response(content=second),
        _mock_response(content='', finish_reason='tool_calls', tool_calls=[_mock_tool_call(
            'kanban_request_review', json.dumps({'summary': 'Both document sections written and verified.',
                                                'artifacts': [str(artifact)]}))]),
        _mock_response(content='Both sections are preserved for review.'),
    ]

    def transport(**kw):
        requests.append((agent.session_id, copy.deepcopy(kw['messages'])))
        if len(requests) == 3:
            requests_at_continuation.append((list(judged), artifact.read_text()))
        assert script, 'unexpected extra model entry'
        return script.pop(0)

    agent.client.chat.completions.create.side_effect = transport
    _run_caller(worker, caller, monkeypatch)
    assert requests_at_continuation == [([partial], '# First\nVerified first section.\n')], (
        'premature terminal nudge bypassed native Goal judgment', requests[2][1][-1])
    assert requests[2][1][-1]['content'] == goals.KANBAN_GOAL_CONTINUATION_TEMPLATE.format(
        reason='check both sections')
    assert any(m.get('content') == partial for m in requests[2][1])
    assert any(m.get('content') == second for m in requests[5][1])
    assert judged[:2] == [partial, second]
    assert len(requests) == 7 and not script
    assert {sid for sid, _ in requests} == {'artifact-session'}
    assert getattr(agent, '_kanban_stop_nudges', 0) == 0
    history = worker.db.get_messages_as_conversation(agent.session_id)
    assert any(m.get('content') == partial for m in history)
    assert '# Second' in artifact.read_text()
    with connect_closing() as conn:
        assert kb.get_task(conn, worker.context['task_id']).status == 'review'
        run = kb.latest_run(conn, worker.context['task_id'])
        assert run is not None and run.status == 'review'
    if caller == 'sdk':
        assert session_kanban.worker_exit_code(worker.context['db'], worker.context) == 0


@pytest.mark.parametrize(('tool', 'args', 'status'), [
    ('kanban_complete', {'summary': 'Both sections verified.'}, 'done'),
    ('kanban_request_review', {'summary': 'Both sections ready for review.'}, 'review'),
    ('kanban_block', {'reason': 'Only the customer can provide the source document.',
                      'kind': 'needs_input'}, 'blocked'),
])
def test_first_turn_handoff_is_not_reopened(worker, monkeypatch, tool, args, status):
    from tests.agent.test_run_agent import _mock_response, _mock_tool_call

    def unexpected_judge(*args, **kwargs):
        pytest.fail('The outer loop judged a run that had already settled')

    monkeypatch.setattr(goals, 'judge_goal', unexpected_judge)
    monkeypatch.setattr('tools.kanban_tools.judge_goal',
                        lambda *a, **kw: ('done', 'handoff accepted', False, None, False))
    monkeypatch.setattr('agent.auxiliary_client.get_text_auxiliary_client',
                        lambda *a, **kw: (object(), 'inert-judge'))
    transport = worker.agent.client.chat.completions.create
    transport.side_effect = [
        _mock_response(content='', finish_reason='tool_calls',
                       tool_calls=[_mock_tool_call(tool, json.dumps(args))]),
        _mock_response(content='Handed off.'),
    ]

    result = session_kanban.run_worker_turns(worker.agent, worker.frame, [])

    assert result['final_response'] == 'Handed off.'
    assert transport.call_count == 2
    with connect_closing() as conn:
        task_id = worker.context['task_id']
        task = kb.get_task(conn, task_id)
        assert task.status == status and task.current_run_id is None
        assert conn.execute('SELECT id FROM task_runs WHERE task_id=?',
                            (task_id,)).fetchall()[0][0] == worker.context['run_id']
        assert conn.execute('SELECT COUNT(*) FROM task_runs WHERE task_id=?',
                            (task_id,)).fetchone()[0] == 1


@pytest.mark.parametrize('worker', [{'goal_mode': False, 'inherited_goal': '1'}], indirect=True)
def test_false_assignment_overwrites_inherited_goal_and_keeps_terminal_guard(worker, monkeypatch):
    from agent.turn_stop_gates import apply_stop_gates

    assert worker.context['goal_mode'] is False
    assert os.environ['HERMES_KANBAN_TASK'] == worker.context['task_id']
    assert os.environ['HERMES_KANBAN_GOAL_MODE'] == '0'
    messages = [{'role': 'user', 'content': 'Write the document.'}]
    verdict = apply_stop_gates(worker.agent, {'role': 'assistant', 'content': 'First section written.'},
        final_response='First section written.', messages=messages, conversation_history=[],
        pending_verification_response=None, pending_verification_response_previewed=False)
    assert verdict.continue_turn
    assert messages[-1]['_kanban_stop_synthetic'] is True
    assert worker.context['task_id'] in messages[-1]['content']


@pytest.mark.parametrize(('verdict', 'budget', 'entries', 'reason'), [
    ('blocked', 3, 1, 'unachievable'),
    ('done', 3, 2, 'never'),
    ('continue', 1, 1, 'budget'),
])
def test_goal_controller_still_blocks_without_terminal_handoff(
        worker, monkeypatch, verdict, budget, entries, reason):
    from tests.agent.test_run_agent import _mock_response

    worker.context['goal_max_turns'] = budget
    worker.frame['policy']['kanban_json'] = json.dumps(worker.context)
    monkeypatch.setattr(goals, 'judge_goal', lambda *a, **kw: (verdict, 'criteria missing', False, None, False))
    worker.agent.client.chat.completions.create.side_effect = lambda **kw: _mock_response(
        content='The first section is available; the second section is missing.')
    result = session_kanban.run_worker_turns(worker.agent, worker.frame, [])
    assert result['completed'] and not result.get('failed')
    assert worker.agent.client.chat.completions.create.call_count == entries
    assert getattr(worker.agent, '_kanban_stop_nudges', 0) == 0
    with connect_closing() as conn:
        task = kb.get_task(conn, worker.context['task_id'])
        assert task is not None and task.status == 'blocked'
        event = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='blocked' ORDER BY id DESC LIMIT 1",
                             (worker.context['task_id'],)).fetchone()
        assert event is not None and reason in json.loads(event[0])['reason'].lower()


def test_initial_iteration_exhaustion_cannot_enter_goal_controller(worker, monkeypatch):
    from tests.agent.test_run_agent import _mock_response, _mock_tool_call

    worker.agent.max_iterations = 1
    monkeypatch.setattr(goals, 'judge_goal', lambda *a, **kw: pytest.fail('unfinished initial turn entered Goal'))
    calls = []
    def transport(**kw):
        calls.append(copy.deepcopy(kw['messages']))
        if len(calls) == 1:
            return _mock_response(content='', finish_reason='tool_calls', tool_calls=[_mock_tool_call(
                'write_file', json.dumps({'path': str(worker.artifact), 'content': 'First section only.\n'}))])
        return _mock_response(content='Only the first section was written before the iteration limit.')
    worker.agent.client.chat.completions.create.side_effect = transport
    result = session_kanban.run_worker_turns(worker.agent, worker.frame, [])
    assert result['completed'] is False
    assert 'max_iterations' in result['turn_exit_reason'] or 'budget' in result['turn_exit_reason']
    assert session_kanban.worker_exit_code(worker.context['db'], worker.context) == 1
    assert getattr(worker.agent, '_kanban_stop_nudges', 0) == 0


def _stop_agent():
    return SimpleNamespace(session_id='policy-session', _turn_file_mutation_paths=set(),
        _interim_content_was_streamed=lambda text: False,
        _emit_diagnostic_status=lambda text: None,
        _emit_interim_assistant_message=lambda msg: None,
        _flush_messages_to_session_db=lambda *a: None)


def _apply_stop(agent):
    from agent.turn_stop_gates import apply_stop_gates
    messages = [{'role': 'user', 'content': 'Write the document.'}]
    verdict = apply_stop_gates(agent, {'role': 'assistant', 'content': 'First section written.'},
        final_response='First section written.', messages=messages, conversation_history=[],
        pending_verification_response=None, pending_verification_response_previewed=False)
    return verdict, messages


@pytest.mark.parametrize('mode', [None, '0', 'true', '1 ', 'yes'])
def test_ordinary_modes_keep_real_stop_guard(monkeypatch, mode):
    monkeypatch.setenv('HERMES_KANBAN_TASK', 'artifact-task')
    monkeypatch.delenv('HERMES_KANBAN_STOP_NUDGE', raising=False)
    if mode is None:
        monkeypatch.delenv('HERMES_KANBAN_GOAL_MODE', raising=False)
    else:
        monkeypatch.setenv('HERMES_KANBAN_GOAL_MODE', mode)
    verdict, messages = _apply_stop(_stop_agent())
    assert verdict.continue_turn
    assert messages[-1]['_kanban_stop_synthetic'] is True
    assert messages[-2]['finish_reason'] == 'kanban_terminal_required'


@pytest.mark.parametrize('kind', ['delegate', 'cron', 'descendant', 'no-task'])
def test_nonowned_context_never_acquires_worker_stop_policy(monkeypatch, kind):
    from contextlib import nullcontext
    from agent.delegation_context import (delegated_child_context, non_dispatcher_owned_context,
                                           owned_kanban_task, delegated_child_subprocess_env)
    monkeypatch.setenv('HERMES_KANBAN_TASK', 'artifact-task')
    monkeypatch.setenv('HERMES_KANBAN_GOAL_MODE', '0')
    if kind == 'descendant':
        monkeypatch.setenv('HERMES_DELEGATED_CHILD_CONTEXT', str(Path.cwd()))
    if kind == 'no-task':
        monkeypatch.delenv('HERMES_KANBAN_TASK')
        monkeypatch.setenv('HERMES_KANBAN_GOAL_MODE', '1')
    scope = {'delegate': delegated_child_context, 'cron': non_dispatcher_owned_context}.get(kind, nullcontext)
    with scope():
        assert owned_kanban_task() == ''
        verdict, messages = _apply_stop(_stop_agent())
        assert not verdict.continue_turn
        assert len(messages) == 1
        if kind == 'delegate':
            env = delegated_child_subprocess_env(dict(os.environ, HERMES_KANBAN_GOAL_MODE='1'))
            assert 'HERMES_KANBAN_GOAL_MODE' not in env
            assert 'HERMES_KANBAN_TASK' not in env
            assert env['HERMES_DELEGATED_CHILD_CONTEXT']


def test_goal_exemption_does_not_skip_other_stop_gates(monkeypatch):
    monkeypatch.setenv('HERMES_KANBAN_TASK', 'artifact-task')
    monkeypatch.setenv('HERMES_KANBAN_GOAL_MODE', '1')
    monkeypatch.setattr('agent.verification_stop.verify_on_stop_enabled', lambda: True)
    monkeypatch.setattr('agent.verification_stop.build_verify_on_stop_nudge', lambda **kw: 'Verify the document.')
    verdict, messages = _apply_stop(_stop_agent())
    assert verdict.continue_turn
    assert messages[-1]['_verification_stop_synthetic'] is True
    assert messages[-2]['finish_reason'] == 'verification_required'
