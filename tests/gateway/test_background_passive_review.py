import pytest
from unittest.mock import AsyncMock, MagicMock, patch

@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    import hermes_state
    monkeypatch.setattr(hermes_state, 'DEFAULT_DB_PATH', tmp_path / 'state.db')
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    s = SessionStore(sessions_dir=tmp_path / 'sessions', config=GatewayConfig())
    s._db.create_session(session_id='parent', source='telegram')
    yield s
    s._db.close()

def test_real_compressor_omitting_result_keeps_protected_tail_after_reopen(store):
    from agent.context_compressor import ContextCompressor
    from gateway.session_transcript import background_context_receipt, background_context_for_compaction
    from gateway.session import SessionStore
    from gateway.config import GatewayConfig
    history=[]
    for i in range(40):
        history += [{'role':'user','content':f'old request {i} '+('details '*600)},
                    {'role':'assistant','content':f'old answer {i} '+('analysis '*600)}]
    history.append(background_context_receipt('bg_review', 'result', 'MUST_KEEP_EXACT_482'))
    for m in history:
        store.append_to_transcript('parent', m)
    projected=background_context_for_compaction(store.load_transcript('parent'))
    compressor=ContextCompressor('test/model', config_context_length=8192, protect_first_n=1,
                                 protect_last_n=3, quiet_mode=True)
    compressor.tail_token_budget=1000
    with patch.object(compressor, '_generate_summary', return_value='Summary deliberately omits every background result.') as summary:
        compressed=compressor.compress(projected, current_tokens=50000, force=True)
    assert summary.called
    assert any('MUST_KEEP_EXACT_482' in str(m.get('content')) for m in compressed)
    store._db.publish_compression_child(parent_session_id='parent',child_session_id='child',source='telegram',
                                      messages=compressed,require_compression_lease=False)
    store._db.close()
    reopened=SessionStore(sessions_dir=store.sessions_dir,config=GatewayConfig())
    try:
        reloaded = reopened.load_transcript('parent')
        assert any('MUST_KEEP_EXACT_482' in str(m.get('content')) for m in reloaded)
        from gateway.session_transcript import pending_background_context
        assert pending_background_context(reloaded) is None
    finally:
        reopened._db.close()

@pytest.mark.parametrize('tail', ['user', 'tool_open', 'tool_complete'])
def test_native_sequence_repair_of_projected_tail(tail):
    from gateway.session_transcript import background_context_receipt, background_context_for_compaction
    from gateway.run import _build_gateway_agent_history
    from agent.agent_runtime_helpers import repair_message_sequence
    history=[{'role':'user','content':'ACTUAL_TASK'}]
    if tail != 'user':
        history.append({'role':'assistant','content':None,'tool_calls':[{'id':'tc','type':'function','function':{'name':'terminal','arguments':'{}'}}]})
    if tail == 'tool_complete':
        history.append({'role':'tool','tool_call_id':'tc','content':'TOOL_VALUE'})
    history.append(background_context_receipt('bg_review','result','PAST_RESULT'))
    projected=background_context_for_compaction(history)
    repair_message_sequence(None,projected)
    roles=[m['role'] for m in projected]
    assert not any(a==b and a in {'user','assistant'} for a,b in zip(roles,roles[1:]))
    assert any('PAST_RESULT' in str(m.get('content')) for m in projected)
    if tail == 'tool_complete':
        assert roles == ['user','assistant','tool','user']
        assert projected[1]['tool_calls'][0]['id']=='tc'
    if tail == 'tool_open':
        assert not any(m.get('tool_calls') for m in projected)
    # Child path calls _build BEFORE appending receipt; it removes open tool tails already.
    child,_ = _build_gateway_agent_history(history)
    if tail == 'tool_open':
        assert [m['role'] for m in child] == ['user', 'assistant', 'tool']
        assert child[-1]['tool_call_id'] == 'tc'
        assert 'UNKNOWN' in str(child[-1]['content'])

@pytest.mark.asyncio
async def test_native_db_write_failure_does_not_block_telegram_delivery(store):
    from tests.gateway.test_background_command import _make_runner, _make_event
    from gateway.session import AsyncSessionStore
    from gateway.config import Platform
    runner=_make_runner();runner.session_store=store;runner._async_session_store=AsyncSessionStore(store)
    runner._run_in_executor_with_context=AsyncMock(return_value={'final_response':'RESULT_ON_TELEGRAM','messages':[]})
    runner._resolve_session_agent_runtime=MagicMock(return_value=('test/model',{'api_key':'fake'}))
    adapter=MagicMock();adapter.send=AsyncMock()
    adapter.extract_media.side_effect=lambda t:([],t);adapter.extract_images.side_effect=lambda t:([],t)
    runner.adapters[Platform.TELEGRAM]=adapter
    with patch('gateway.run._load_gateway_config',return_value={}), patch.object(store,'_append_transcript_message',side_effect=OSError('simulated disk outage')):
        await runner._run_background_task_inner('TASK',_make_event().source,'bg_review',parent_session_id='parent',parent_session_key='telegram:x',parent_conversation_history=[])
    adapter.send.assert_awaited_once()
    assert 'RESULT_ON_TELEGRAM' in adapter.send.call_args.kwargs['content']
    assert len(store._dirty_transcripts['parent'])==1
    # On recovery native append drains the receipt before later writes.
    store.append_to_transcript('parent',{'role':'user','content':'next real message'})
    assert any('RESULT_ON_TELEGRAM' in str(m.get('content')) for m in store.load_transcript('parent'))


def test_core_projection_preserves_unrelated_system_role():
    from agent.conversation_compression import _project_passive_gateway_receipts
    from gateway.session_transcript import background_context_receipt
    messages=[{'role':'system','content':'EXISTING_SYSTEM_CONTEXT'},
              {'role':'user','content':'REAL_INSTRUCTION'},
              background_context_receipt('bg','result','RESULT')]
    projected=_project_passive_gateway_receipts(messages)
    assert messages[0] in projected


def test_native_image_reply_turn_preserves_image_in_durable_carrier(store, tmp_path):
    import base64
    from types import SimpleNamespace
    from gateway.run_turn_runner import TurnRunner
    from gateway.session_transcript import background_context_receipt
    path=tmp_path/'one.png'
    path.write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a9N8AAAAASUVORK5CYII='))
    store.append_to_transcript('parent',background_context_receipt('bg','result','RESULT'))
    ctx=SimpleNamespace(mute_notification_reply=False, title_user_message=None, session_key='telegram:test',session_id='parent',history=store.load_transcript('parent'),
        source=SimpleNamespace(user_id='test-user',user_name='Test',is_bot=False),
        message='[Replying to: Previous message reference]\nDescribe this image',current_user_text='Describe this image',reply_to_text='Previous message reference',
        internal_context={},persist_user_display_kind=None, persist_user_display_metadata=None,moa_config=None,inbound_message_id='31')
    runner=MagicMock();runner._consume_pending_native_image_paths.return_value=[str(path)]
    turn=TurnRunner(runner,ctx);turn._approval_notify_sync=lambda *a,**k:None
    class Agent:
        def run_conversation(self,message,*,current_user_text=None,reply_to_text=None,**kwargs):
            assert current_user_text==ctx.current_user_text
            assert reply_to_text==ctx.reply_to_text
            assert any(p.get('type')=='image_url' for p in message)
            store.append_to_transcript('parent',{'role':'user','content':kwargs['persist_user_message'],
                'display_kind':kwargs['persist_user_display_kind'],'display_metadata':kwargs['persist_user_display_metadata']})
    turn._run_conversation_with_approval(Agent(),[],None,None,None)
    row=next(m for m in store.load_transcript('parent') if m['role']=='user')
    assert isinstance(row['content'],list)
    assert any(p.get('type')=='image_url' for p in row['content'])
    assert any('Previous message reference' in str(p.get('text', '')) for p in row['content'])
    from gateway.session import SessionStore
    from gateway.config import GatewayConfig
    store._db.close()
    reopened = SessionStore(sessions_dir=store.sessions_dir, config=GatewayConfig())
    try:
        replay = next(m for m in reopened.load_transcript('parent') if m['role'] == 'user')
        assert replay['content'] == row['content']
    finally:
        reopened._db.close()


from tests.agent.test_api_content_sidecar import wire_env, _chat_requests
from types import SimpleNamespace

@pytest.mark.parametrize('mode',['observed','explicit'])
def test_native_override_image_live_persist_reopen(wire_env,tmp_path,mode):
    """Background receipts preserve native image behavior, including durable path hints.

    Native storage intentionally replaces image blocks by [screenshot]; a text override
    is ignored for multimodal content. Compare the actual baseline instead of changing it.
    """
    from gateway.run_turn_runner import TurnRunner
    from gateway.run import _build_gateway_agent_history
    from gateway.session_transcript import background_context_receipt
    from agent.turn_context import build_api_messages
    from hermes_state import SessionDB
    from PIL import Image
    make_agent,handler,db,sid=wire_env
    image=tmp_path/'sample.png'
    Image.new('RGB',(2,2),(255,0,0)).save(image)
    agents=[]
    session_ids=[]
    try:
        for background in (False,True):
            agent=make_agent()
            agents.append(agent)
            current_sid=sid+('-bg' if background else '-baseline')
            agent.session_id=current_sid
            session_ids.append(current_sid)
            r=background_context_receipt('bg_native','result','RESULT_NATIVE_749')
            ctx=SimpleNamespace(mute_notification_reply=False, title_user_message=None, session_key='telegram:native',session_id=current_sid,history=[r] if background else [],
                source=SimpleNamespace(user_id='test-user',user_name='Test',is_bot=False),
                message='[Replying to: REPLY_547]\nDescribe this image',current_user_text='Describe this image',
                reply_to_text='REPLY_547',internal_context={},persist_user_display_kind=None, persist_user_display_metadata=None,moa_config=None,inbound_message_id='31')
            runner=MagicMock();runner._consume_pending_native_image_paths.return_value=[str(image)]
            turn=TurnRunner(runner,ctx);turn._approval_notify_sync=lambda *a,**k:None
            with patch('agent.image_routing._lookup_supports_vision', return_value=True):
                result=turn._run_conversation_with_approval(agent,[], 'OBSERVED_GROUP_191' if mode=='observed' else None,
                                                          'INTENDED_OVERRIDE_835' if mode=='explicit' else None,None)
            assert not result.get('error'),result
            live_user=next(m for m in _chat_requests(handler)[-1]['messages'] if m['role']=='user')
            assert isinstance(live_user['content'],list)
            assert any(part.get('type')=='image_url' for part in live_user['content'])
            agent._end_session_on_close=False
            agent.close()
        path=db.db_path
        db.close()
        reopened=SessionDB(db_path=path)
        try:
            user_rows=[]
            for index,current_sid in enumerate(session_ids):
                durable=reopened.get_messages_as_conversation(current_sid)
                user=next(m for m in durable if m['role']=='user')
                user_rows.append(user)
                text=user['content']
                assert isinstance(text,str)
                assert '[screenshot]' in text
                assert str(image) in text
                assert 'REPLY_547' in text
                assert 'Describe this image' in text
                assert 'INTENDED_OVERRIDE_835' not in text  # native multimodal precedence
                assert ('OBSERVED_GROUP_191' in text) == (mode=='observed')
                if index:
                    assert 'RESULT_NATIVE_749' in text
                    assert user['display_metadata']['background_event_ids']==['bg_native:result']
                replay,_=_build_gateway_agent_history(durable)
                wire,_=build_api_messages(agents[index],replay,current_turn_user_idx=None,ext_prefetch_cache=None,
                                         plugin_user_context=None,moa_config=None,active_system_prompt='')
                replay_user=next(m for m in wire if m['role']=='user')
                assert str(image) in replay_user['content']
                assert '[screenshot]' in replay_user['content']
            assert ('RESULT_NATIVE_749' in user_rows[0]['content']) is False
        finally:
            reopened.close()
    finally:
        for agent in agents:
            agent._end_session_on_close=False
            agent.close()
