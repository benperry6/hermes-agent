"""A/B/A publication homes through actual HTTP; inference alone is inert."""
import asyncio
import json
import threading
import time
import urllib.request
from pathlib import Path

from gateway.config import PlatformConfig
from gateway.run import _profile_runtime_scope
from hermes_constants import get_hermes_home
from plugins.platforms.a2a import adapter as native_adapter, protocol
from plugins.platforms.a2a.adapter import A2AAdapter


def test_detached_http_publication_keeps_owning_profile_a_b_a(tmp_path, monkeypatch):
    monkeypatch.setattr(A2AAdapter, "_wire_plugin_handlers", lambda self, native=None: None)
    homes = {name: tmp_path / name for name in ("A", "B")}
    for home in homes.values():
        home.mkdir()
    observed = []
    persist = protocol.persist_message
    def observe_persist(*args):
        observed.append((str(get_hermes_home()), args[1], args[2]))
        return persist(*args)
    monkeypatch.setattr(protocol, "persist_message", observe_persist)
    loops, threads, adapters = [], [], []
    try:
        for name in ("A", "B", "A"):
            with _profile_runtime_scope(homes[name]):
                adapter = A2AAdapter(PlatformConfig(enabled=True, extra={"port": 0}))
                loop = asyncio.new_event_loop()
                thread = threading.Thread(target=loop.run_forever, daemon=True)
                thread.start()
                asyncio.run_coroutine_threadsafe(adapter.connect(), loop).result(5)
                async def inert(event):
                    return "inert owner:" + str(get_hermes_home())
                adapter.set_message_handler(inert)
            loops.append(loop); threads.append(thread); adapters.append(adapter)
            def rpc(method, params):
                req = urllib.request.Request(
                    "http://127.0.0.1:" + str(adapter._httpd.server_address[1]) + "/",
                    data=json.dumps({"jsonrpc": "2.0", "id": "scope", "method": method, "params": params}).encode(),
                    headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=5) as res:
                    return json.load(res)["result"]
            result = rpc("message/send", {"configuration": {"returnImmediately": True},
                "message": protocol.text_message(protocol.ROLE_USER, "inert scope", context_id="scope-" + name)})
            limit = time.monotonic() + 5
            while time.monotonic() < limit:
                result = rpc("tasks/get", {"id": result["id"]})
                if result["status"]["state"] != protocol.STATE_WORKING:
                    break
                time.sleep(0.01)
            assert result["status"]["state"] == protocol.STATE_COMPLETED
            assert result["status"]["message"]["parts"][0]["text"] == "inert owner:" + str(homes[name])
        assert [row[0] for row in observed] == [str(homes["A"]), str(homes["A"]),
            str(homes["B"]), str(homes["B"]), str(homes["A"]), str(homes["A"])]
        assert [row[1] for row in observed] == ["user", "agent"] * 3
    finally:
        for adapter, loop, thread in zip(adapters, loops, threads):
            asyncio.run_coroutine_threadsafe(adapter.disconnect(), loop).result(5)
            loop.call_soon_threadsafe(loop.stop)
            thread.join(5)
            loop.close()
