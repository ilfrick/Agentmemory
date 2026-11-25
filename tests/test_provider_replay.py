import asyncio
import types

import pytest

from hindsight_hermes.local_fallback import LocalFallbackStore


class NotFoundException(Exception):
    status = 404


class FakeDocuments:
    def __init__(self):
        self.rows = {}
        self.get_calls = []
        self.delete_calls = []
        self.fail_retain = False
        self.preserve_remote_state = False
        self.retain_response = types.SimpleNamespace()

    async def get_document(self, bank_id, document_id):
        self.get_calls.append((bank_id, document_id))
        try:
            return self.rows[(bank_id, document_id)]
        except KeyError:
            raise NotFoundException()

    async def delete_document(self, bank_id, document_id):
        self.delete_calls.append((bank_id, document_id))
        self.rows.pop((bank_id, document_id), None)


class ReplayClient:
    def __init__(self):
        self.documents = FakeDocuments()
        self.retain_calls = []
        self.fail_retain = False
        self.retain_result_override = None

    async def aretain_batch(self, **kwargs):
        self.retain_calls.append(kwargs)
        if self.fail_retain:
            raise ConnectionError("synthetic outage")
        bank_id = kwargs["bank_id"]
        document_id = kwargs.get("document_id", "tool-retain-document")
        content = kwargs["items"][0]["content"]
        key = (bank_id, document_id)
        if not self.documents.preserve_remote_state:
            prior = self.documents.rows.get(key)
            if prior is not None and kwargs["items"][0].get("update_mode") == "append":
                content = prior.original_text + "\n" + content
            self.documents.rows[key] = types.SimpleNamespace(
                id=document_id, bank_id=bank_id, original_text=content
            )
        return self.retain_result_override or types.SimpleNamespace(
            success=True, bank_id=bank_id, operation_id=kwargs.get("operation_id")
        )


def _provider(provider, tmp_path):
    instance, client = provider({"temporal_gate_passed": True}, client=ReplayClient())
    instance._fallback_store = LocalFallbackStore(tmp_path / "replay.sqlite3")
    instance._client = client
    def run(operation, **_kwargs):
        return asyncio.run(operation(client))

    instance._run_sync = lambda coro: asyncio.run(coro)
    instance._run_hindsight_operation = run
    return instance, client


def test_online_sync_turn_is_sent_once_not_by_replay_and_direct_job(provider, tmp_path, monkeypatch):
    instance, client = _provider(provider, tmp_path)
    monkeypatch.setattr(instance, "_enqueue_retain", lambda job: job())

    instance.sync_turn("one user turn", "one assistant reply")

    assert len(client.retain_calls) == 1
    instance.shutdown()


def test_online_builtin_write_uses_one_primary_durable_path(provider, tmp_path, monkeypatch):
    instance, client = _provider(provider, tmp_path)
    instance._bank_id = "primary"
    instance._write_bank_ids = ["primary"]
    monkeypatch.setattr(instance, "_enqueue_retain", lambda job: job())

    instance.on_memory_write("add", "memory", "one built-in fact")

    assert [call["bank_id"] for call in client.retain_calls] == ["primary"]
    op = instance._fallback_store.operations()[0]
    assert client.retain_calls[0]["operation_id"] == op.operation_id
    assert instance._fallback_store.next_operation() is None
    instance.shutdown()


def test_online_tool_write_replays_primary_and_writes_each_extra_once(provider, tmp_path, monkeypatch):
    instance, client = _provider(provider, tmp_path)
    instance._bank_id = "primary"
    instance._write_bank_ids = ["primary", "shared", "archive"]
    monkeypatch.setattr(instance, "_enqueue_retain", lambda job: job())

    instance._tool_retain({"content": "one explicit fact"})

    assert [call["bank_id"] for call in client.retain_calls] == ["primary", "shared", "archive"]
    primary_op = instance._fallback_store.operations()[0]
    assert client.retain_calls[0]["operation_id"] == primary_op.operation_id
    assert instance._fallback_store.next_operation() is None
    instance.shutdown()


@pytest.mark.parametrize("entrypoint", ["sync_turn", "builtin", "tool_retain"])
def test_failed_primary_outbox_replay_skips_extra_banks(provider, tmp_path, monkeypatch, entrypoint):
    instance, client = _provider(provider, tmp_path)
    instance._bank_id = "primary"
    instance._write_bank_ids = ["primary", "shared", "archive"]
    monkeypatch.setattr(instance, "_replay_fallback_outbox", lambda *_args: False)
    monkeypatch.setattr(instance, "_enqueue_retain", lambda job: instance._retain_queue.put(job))
    direct_banks = []
    instance._retain_batch = lambda item, bank_id, **kwargs: direct_banks.append(bank_id)

    if entrypoint == "sync_turn":
        instance.sync_turn("user", "assistant")
    elif entrypoint == "builtin":
        instance.on_memory_write("add", "memory", "builtin fact")
    else:
        instance._tool_retain({"content": "explicit fact"})
    instance._retain_queue.get_nowait()()

    assert direct_banks == []
    instance.shutdown()


def test_successful_primary_replay_isolates_extra_bank_failures(provider, tmp_path, monkeypatch):
    instance, client = _provider(provider, tmp_path)
    instance._bank_id = "primary"
    instance._write_bank_ids = ["primary", "broken", "shared"]
    monkeypatch.setattr(instance, "_replay_fallback_outbox", lambda *_args: True)
    monkeypatch.setattr(instance, "_enqueue_retain", lambda job: instance._retain_queue.put(job))
    direct_banks = []

    def retain(item, bank_id, **kwargs):
        direct_banks.append(bank_id)
        if bank_id == "broken":
            raise RuntimeError("synthetic extra-bank failure")

    instance._retain_batch = retain
    instance._tool_retain({"content": "explicit fact"})
    instance._retain_queue.get_nowait()()

    assert direct_banks == ["broken", "shared"]
    instance.shutdown()


def test_later_write_waits_for_its_primary_ack_before_extra_banks(provider, tmp_path, monkeypatch):
    instance, client = _provider(provider, tmp_path)
    instance._bank_id = "primary"
    instance._write_bank_ids = ["primary", "shared"]
    earlier = instance._fallback_store.retain(
        "source:earlier", "earlier exact text", bank_id="primary"
    )
    queued_jobs = []
    monkeypatch.setattr(instance, "_enqueue_retain", queued_jobs.append)
    primary_operation_ids_at_extra_write = []
    statuses_at_extra_write = []

    def write_extra(_item, bank_id, **_kwargs):
        assert bank_id == "shared"
        primary_operation_ids_at_extra_write.append(
            [call.get("operation_id") for call in client.retain_calls]
        )
        statuses_at_extra_write.append(
            [operation.status for operation in instance._fallback_store.operations()]
        )

    instance._retain_batch = write_extra
    instance._tool_retain({"content": "later explicit fact"})
    current = instance._fallback_store.operations()[-1]
    assert len(queued_jobs) == 1

    instance._client = client
    queued_jobs.pop()()

    assert [call["operation_id"] for call in client.retain_calls] == [
        earlier.operation_id,
        current.operation_id,
    ]
    assert primary_operation_ids_at_extra_write == [[earlier.operation_id, current.operation_id]]
    assert statuses_at_extra_write == [["done", "done"]]
    assert instance._fallback_store.next_operation() is None
    instance.shutdown()


def test_replay_claims_only_fifo_head_and_reuses_operation_id_on_retry(provider, tmp_path):
    instance, client = _provider(provider, tmp_path)
    store = instance._fallback_store
    first = store.retain("source:first", "first exact text", bank_id="synthetic-bank")
    second = store.retain("source:second", "second exact text", bank_id="synthetic-bank")
    client.fail_retain = True

    assert instance._replay_fallback_outbox() is False
    assert store.next_operation().operation_id == first.operation_id
    assert store.claim_next("other-owner") is not None  # release permits retry
    store.release(first.operation_id, owner_id="other-owner")
    client.fail_retain = False
    assert instance._replay_fallback_outbox() is True
    assert [call["operation_id"] for call in client.retain_calls] == [first.operation_id, first.operation_id]
    assert client.retain_calls[-1]["document_id"] == first.document_id
    assert store.next_operation().operation_id == second.operation_id
    instance.shutdown()


def test_successful_direct_write_is_not_replayed_after_reconnect(provider, tmp_path, monkeypatch):
    instance, client = _provider(provider, tmp_path)
    instance._client = None
    monkeypatch.setattr(instance, "_enqueue_retain", lambda job: job())

    instance.sync_turn("completed user turn", "completed assistant turn")
    initial_call_count = len(client.retain_calls)
    instance._client = client

    assert instance._replay_fallback_outbox() is False
    assert len(client.retain_calls) == initial_call_count == 1
    instance.shutdown()


def test_two_turn_fallback_replay_uses_full_session_overwrite(provider, tmp_path, monkeypatch):
    instance, client = _provider(provider, tmp_path)
    instance._client = None
    monkeypatch.setattr(instance, "_enqueue_retain", lambda _job: None)

    instance.sync_turn("first user", "first assistant")
    instance.sync_turn("second user", "second assistant")
    operations = instance._fallback_store.operations()
    assert len(operations) == 2

    instance._client = client
    assert instance._replay_fallback_outbox() is True
    assert instance._replay_fallback_outbox() is True

    first, second = operations
    assert client.retain_calls[1]["operation_id"] == second.operation_id
    assert client.retain_calls[1]["items"][0]["content"] == second.expected_remote_text
    assert "update_mode" not in client.retain_calls[1]["items"][0]
    assert client.documents.rows[("hermes", second.document_id)].original_text == second.expected_remote_text
    assert client.retain_calls[1]["document_id"] == first.document_id == second.document_id
    instance.shutdown()


def test_overwrite_fallback_expected_text_is_not_duplicated(provider, tmp_path, monkeypatch):
    instance, _client = _provider(provider, tmp_path)
    instance._document_id = "session-1-overwrite"
    monkeypatch.setattr(instance, "_resolve_retain_target", lambda _fallback: ("session-1-overwrite", None))
    monkeypatch.setattr(instance, "_enqueue_retain", lambda _job: None)

    instance.sync_turn("first user turn", "first assistant reply")
    instance.sync_turn("second user turn", "second assistant reply")

    operations = instance._fallback_store.operations()
    assert len(operations) == 2
    second = operations[1]
    assert second.payload == second.expected_remote_text
    assert second.expected_remote_text.count("first user turn") == 1
    assert second.expected_remote_text.count("second user turn") == 1
    instance.shutdown()


def test_replay_requires_exact_remote_document_bank_id_and_text(provider, tmp_path):
    instance, client = _provider(provider, tmp_path)
    store = instance._fallback_store
    op = store.retain("source:exact", "expected exact text", bank_id="synthetic-bank")
    client.documents.preserve_remote_state = True
    client.documents.rows[("synthetic-bank", op.document_id)] = types.SimpleNamespace(
        bank_id="other-bank", id=op.document_id, original_text="expected exact text"
    )

    assert instance._replay_fallback_outbox() is False
    assert store.next_operation().status == "pending"
    assert store.next_operation().lease_owner is None
    instance.shutdown()


@pytest.mark.parametrize("field,value", [("id", "other-id"), ("original_text", "almost exact text")])
def test_replay_does_not_ack_mismatched_readback(provider, tmp_path, field, value):
    instance, client = _provider(provider, tmp_path)
    store = instance._fallback_store
    op = store.retain("source:mismatch", "expected exact text", bank_id="synthetic-bank")
    client.documents.preserve_remote_state = True
    attrs = {"bank_id": "synthetic-bank", "id": op.document_id, "original_text": "expected exact text"}
    attrs[field] = value
    client.documents.rows[("synthetic-bank", op.document_id)] = types.SimpleNamespace(**attrs)

    assert instance._replay_fallback_outbox() is False
    assert store.next_operation().status == "pending"
    instance.shutdown()


def test_delete_replay_verifies_not_found_before_ack(provider, tmp_path):
    instance, client = _provider(provider, tmp_path)
    store = instance._fallback_store
    retained = store.retain("source:delete", "delete me", bank_id="synthetic-bank")
    store.claim_next("setup")
    store.complete_retain(retained.operation_id, retained.document_id, "delete me", owner_id="setup")
    deletion = store.remove("source:delete", bank_id="synthetic-bank")
    client.documents.rows[("synthetic-bank", deletion.document_id)] = types.SimpleNamespace(
        bank_id="synthetic-bank", id=deletion.document_id, original_text="delete me"
    )

    assert instance._replay_fallback_outbox() is True
    assert client.documents.delete_calls == [("synthetic-bank", deletion.document_id)]
    assert client.documents.get_calls[-1] == ("synthetic-bank", deletion.document_id)
    assert store.next_operation() is None
    instance.shutdown()


def test_replay_reconstructs_stored_item_options(provider, tmp_path):
    instance, client = _provider(provider, tmp_path)
    expected_options = {
        "context": "user preference",
        "metadata": {"source": "synthetic", "turn": "3"},
        "tags": ["offline", "preference"],
        "timestamp": "2026-11-24T11:00:00+01:00",
        "strategy": "test-strategy",
    }
    instance._fallback_store.retain(
        "source:options", "exact replay content", bank_id="synthetic-bank", item_options=expected_options
    )

    assert instance._replay_fallback_outbox() is True
    assert client.retain_calls[0]["items"] == [{"content": "exact replay content", **expected_options}]
    instance.shutdown()


def test_replay_does_not_ack_ambiguous_retain_response(provider, tmp_path):
    instance, client = _provider(provider, tmp_path)
    store = instance._fallback_store
    op = store.retain("source:response", "must verify response", bank_id="synthetic-bank")
    client.retain_result_override = types.SimpleNamespace(
        success=True, bank_id="synthetic-bank", operation_id="different-operation"
    )

    assert instance._replay_fallback_outbox() is False
    pending = store.next_operation()
    assert pending.operation_id == op.operation_id
    assert pending.lease_owner is None
    instance.shutdown()


def test_replay_failure_releases_head_without_acknowledging(provider, tmp_path):
    instance, client = _provider(provider, tmp_path)
    store = instance._fallback_store
    op = store.retain("source:failure", "durable text", bank_id="synthetic-bank")
    client.fail_retain = True

    assert instance._replay_fallback_outbox() is False
    pending = store.next_operation()
    assert pending.operation_id == op.operation_id
    assert pending.status == "pending"
    assert pending.lease_owner is None
    instance.shutdown()


def test_local_capture_commits_without_inline_outbox_replay(provider, tmp_path, monkeypatch):
    instance, client = _provider(provider, tmp_path)
    calls = []
    monkeypatch.setattr(instance, "_replay_fallback_outbox", lambda: calls.append("network"))

    op = instance._persist_fallback("source:auto", "automatically replayed", bank_id="synthetic-bank")

    assert calls == []
    assert client.retain_calls == []
    assert instance._fallback_store.next_operation().operation_id == op.operation_id
    instance.shutdown()
