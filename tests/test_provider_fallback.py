"""Provider integration tests using only a temporary SQLite store and synthetic clients."""

import json
import sys
import types

import pytest

from hindsight_hermes.local_fallback import LocalFallbackStore


class OfflineClient:
    def __init__(self):
        self.retains = []
        self.local_store = None
        self.saw_local_before_remote = False

    async def aretain_batch(self, **kwargs):
        if self.local_store is not None:
            self.saw_local_before_remote = self.local_store.get_document("session:session-1:session-1") is not None
        self.retains.append(kwargs)
        return types.SimpleNamespace(operations=[])

    async def arecall(self, **kwargs):
        raise ConnectionError("synthetic offline client")

    async def aclose(self):
        pass


class VerifiedOfflineClient(OfflineClient):
    def __init__(self):
        super().__init__()
        self.documents = types.SimpleNamespace(rows={})
        async def get_document(bank_id, document_id):
            return self.documents.rows[(bank_id, document_id)]
        self.documents.get_document = get_document

    async def aretain_batch(self, **kwargs):
        if self.local_store is not None:
            self.saw_local_before_remote = self.local_store.get_document("session:session-1:session-1") is not None
        self.retains.append(kwargs)
        bank_id = kwargs["bank_id"]
        document_id = kwargs.get("document_id", "tool-retain-document")
        content = kwargs["items"][0]["content"]
        self.documents.rows[(bank_id, document_id)] = types.SimpleNamespace(
            id=document_id, bank_id=bank_id, original_text=content
        )
        return types.SimpleNamespace(
            success=True, bank_id=bank_id, operation_id=kwargs.get("operation_id")
        )


def _with_store(provider, tmp_path, monkeypatch, config=None):
    exceptions = types.ModuleType("hindsight_client_api.exceptions")
    exceptions.NotFoundException = type("NotFoundException", (Exception,), {})
    package = types.ModuleType("hindsight_client_api")
    package.exceptions = exceptions
    monkeypatch.setitem(sys.modules, "hindsight_client_api", package)
    monkeypatch.setitem(sys.modules, "hindsight_client_api.exceptions", exceptions)
    instance, client = provider(config or {}, client=OfflineClient())
    instance._fallback_store = LocalFallbackStore(tmp_path / "fallback.sqlite3")
    client.local_store = instance._fallback_store
    return instance, client


def test_automatic_turn_is_durable_locally_before_remote_attempt(provider, tmp_path, monkeypatch):
    client = VerifiedOfflineClient()
    instance, _ = provider({}, client=client)
    instance._fallback_store = LocalFallbackStore(tmp_path / "fallback.sqlite3")
    client.local_store = instance._fallback_store
    instance.sync_turn("remember the cobalt kite", "I will remember it.")
    document = instance._fallback_store.get_document("session:session-1:session-1")
    instance._retain_queue.join()
    instance.shutdown()

    assert document is not None
    assert "cobalt kite" in document.text
    assert len(client.retains) == 1
    assert client.saw_local_before_remote


def test_offline_recall_uses_local_fts_and_marks_memory_untrusted(provider, tmp_path, monkeypatch):
    instance, _client = _with_store(provider, tmp_path, monkeypatch)
    instance._fallback_store.retain("source:synthetic", "The cobalt kite belongs to Ada.")

    result = json.loads(instance.handle_tool_call("hindsight_recall", {"query": "cobalt kite"}))
    instance.shutdown()

    assert "The cobalt kite belongs to Ada." in result["result"]
    assert "untrusted" in result["result"].lower()


def test_turn_outbox_persists_request_item_options_for_restart_replay(provider, tmp_path, monkeypatch):
    instance, _client = _with_store(provider, tmp_path, monkeypatch)
    instance._client = None
    monkeypatch.setattr(instance, "_enqueue_retain", lambda _job: None)

    instance.sync_turn("I prefer cobalt tea", "Noted.")

    operation = instance._fallback_store.next_operation()
    assert operation.remote_options["item_options"]["context"] == instance._retain_context
    assert operation.remote_options["item_options"]["timestamp"] == "unset"
    assert operation.remote_options["item_options"]["tags"] == ["session:session-1"]
    assert operation.remote_options["item_options"]["metadata"]["session_id"] == "session-1"
    instance.shutdown()


def test_auto_retain_policy_still_gates_local_turn_capture(provider, tmp_path, monkeypatch):
    instance, _client = _with_store(provider, tmp_path, monkeypatch, {"auto_retain": False})
    instance.sync_turn("do not retain locally", "not retained")
    document = instance._fallback_store.get_document("session:session-1:session-1")
    instance.shutdown()

    assert document is None


def test_builtin_replace_and_remove_fail_closed_without_unique_mapping(provider, tmp_path, monkeypatch):
    instance, _client = _with_store(provider, tmp_path, monkeypatch)
    instance.on_memory_write("replace", "memory", "new value", metadata={"previous_content": "old value"})
    instance.on_memory_write("remove", "memory", "", metadata={"previous_content": "old value"})

    assert instance._fallback_store.operations() == []
    instance.shutdown()


@pytest.mark.parametrize(
    "remote_texts,local_texts,expected_untrusted",
    [
        (["remote synthetic memory"], [], ["remote synthetic memory"]),
        ([], ["local synthetic memory"], ["local synthetic memory"]),
        (["remote synthetic memory"], ["local synthetic memory"], ["remote synthetic memory", "local synthetic memory"]),
    ],
    ids=["remote-only", "local-only", "mixed"],
)
def test_every_recalled_memory_is_marked_untrusted(
    provider, tmp_path, monkeypatch, remote_texts, local_texts, expected_untrusted
):
    instance, _client = _with_store(provider, tmp_path, monkeypatch)
    if local_texts:
        instance._fallback_store.retain("source:synthetic", "local synthetic memory")
    instance._recall = lambda _query: [
        *(types.SimpleNamespace(text=text) for text in remote_texts),
        *instance._fallback_results("synthetic"),
    ]

    result = instance._tool_recall({"query": "synthetic"})
    instance.shutdown()

    for text in expected_untrusted:
        line = next(line for line in result.splitlines() if text in line)
        assert "[UNTRUSTED" in line
    assert len(result.splitlines()) == len(expected_untrusted)


def test_sync_turn_never_replays_primary_network_inline(provider, tmp_path, monkeypatch):
    instance, client = _with_store(provider, tmp_path, monkeypatch)
    instance._client = client
    replay_calls = []
    monkeypatch.setattr(instance, "_replay_fallback_outbox", lambda: replay_calls.append("network"))
    monkeypatch.setattr(instance, "_enqueue_retain", lambda job: None)

    instance.sync_turn("synthetic user turn", "synthetic assistant reply")

    assert replay_calls == []
    assert instance._fallback_store.next_operation() is not None
    instance.shutdown()


def test_offline_tool_write_has_one_primary_path_and_replays_after_reconnect(provider, tmp_path, monkeypatch):
    client = VerifiedOfflineClient()
    instance, _ = provider({}, client=client)
    instance._fallback_store = LocalFallbackStore(tmp_path / "verified-fallback.sqlite3")
    instance._bank_id = "primary"
    instance._write_bank_ids = ["primary", "shared", "archive"]
    instance._client = None
    queued = []
    monkeypatch.setattr(instance, "_enqueue_retain", queued.append)

    instance._tool_retain({"content": "synthetic offline tool fact"})
    assert client.retains == []
    assert len(queued) == 1

    instance._client = client
    queued[0]()
    # Replaying after reconnect finds the acknowledged FIFO head gone, so the primary
    # document is not sent a second time. Extra banks preserve configured order.
    assert instance._replay_fallback_outbox() is False

    assert [call["bank_id"] for call in client.retains] == ["primary", "shared", "archive"]
    operation = instance._fallback_store.operations()[0]
    assert client.retains[0].get("operation_id") == operation.operation_id
    assert instance._fallback_store.next_operation() is None
    instance.shutdown()


def test_memory_derived_reflection_is_marked_untrusted(provider, monkeypatch):
    instance, _client = provider({}, client=None)
    instance._reflect = lambda _query: "synthetic reflection from memories"

    tool_text = instance._tool_reflect({"query": "synthetic"})
    instance._prefetch_method = "reflect"
    prefetched, _count = instance._do_recall("synthetic")
    instance.shutdown()

    assert "[UNTRUSTED" in tool_text
    assert "[UNTRUSTED" in prefetched
