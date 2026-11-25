import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_fallback import AmbiguousSourceError, LocalFallbackStore


def _finish_retain(store, operation, text, remote_document_id=None, owner="writer"):
    claimed = store.claim_next(owner)
    assert claimed is not None and claimed.operation_id == operation.operation_id
    return store.complete_retain(
        operation.operation_id,
        remote_document_id or operation.document_id,
        text,
        owner_id=owner,
    )


def test_records_survive_reopening_store(tmp_path):
    path = tmp_path / "fallback.sqlite3"
    store = LocalFallbackStore(path)
    assert store._db.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert store._db.execute("PRAGMA synchronous").fetchone()[0] == 2  # FULL
    operation = store.retain("source:conversation-1", "Remember the blue mug.")
    operation_id = operation.operation_id
    store.close()

    reopened = LocalFallbackStore(path)

    assert reopened.get_document("source:conversation-1").text == "Remember the blue mug."
    assert reopened.next_operation().operation_id == operation_id
    reopened.close()


def test_new_database_has_private_file_mode(tmp_path):
    path = tmp_path / "private.sqlite3"
    LocalFallbackStore(path).close()

    assert path.stat().st_mode & 0o777 == 0o600


def test_existing_database_and_wal_sidecars_are_private(tmp_path):
    path = tmp_path / "existing.sqlite3"
    path.touch()
    path.chmod(0o644)
    store = LocalFallbackStore(path)
    store.retain("source:private", "must not be world-readable")

    for db_file in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        assert db_file.exists()
        assert db_file.stat().st_mode & 0o777 == 0o600
    store.close()


def test_fts_retrieves_natural_language_query(tmp_path):
    store = LocalFallbackStore(tmp_path / "fallback.sqlite3")
    store.retain("source:coffee", "The espresso grinder is set to medium-fine.")

    results = store.search("espresso grinder")

    assert [item.source_id for item in results] == ["source:coffee"]
    store.close()


def test_outbox_is_fifo_idempotent_and_keeps_identity_and_payload(tmp_path):
    path = tmp_path / "fallback.sqlite3"
    store = LocalFallbackStore(path)
    first = store.retain("source:first", "first immutable content")
    duplicate = store.retain("source:first", "first immutable content")
    second = store.retain("source:second", "second content")
    store.close()

    reopened = LocalFallbackStore(path)
    assert duplicate.operation_id == first.operation_id
    assert [op.operation_id for op in reopened.operations()] == [first.operation_id, second.operation_id]
    assert reopened.next_operation().operation_id == first.operation_id
    assert reopened.operations()[0].payload == "first immutable content"
    assert reopened.claim_next("owner-a").operation_id == first.operation_id
    assert reopened.claim_next("owner-b") is None
    assert reopened.complete_retain(second.operation_id, second.document_id, "second content", owner_id="owner-a") is False
    assert reopened.complete_retain(first.operation_id, first.document_id, "first immutable content", owner_id="owner-b") is False
    assert reopened.complete_retain(first.operation_id, first.document_id, "first immutable content", owner_id="owner-a") is True
    assert reopened.next_operation().operation_id == second.operation_id
    reopened.close()


def test_retain_requires_exact_text_and_target_document_readback(tmp_path):
    store = LocalFallbackStore(tmp_path / "fallback.sqlite3")
    op = store.retain("source:exact", "line one\nline two ")
    assert store.claim_next("writer").operation_id == op.operation_id

    assert store.complete_retain(op.operation_id, "wrong-document", "line one\nline two ", owner_id="writer") is False
    assert store.complete_retain(op.operation_id, op.document_id, "line one\nline two", owner_id="writer") is False
    assert store.next_operation().status == "pending"
    assert store.complete_retain(op.operation_id, op.document_id, "line one\nline two ", owner_id="writer") is True
    assert store.next_operation() is None
    store.close()


def test_delete_waits_for_verified_remote_absence(tmp_path):
    store = LocalFallbackStore(tmp_path / "fallback.sqlite3")
    initial = store.retain("source:remove", "keep until remote is absent")
    assert _finish_retain(store, initial, "keep until remote is absent") is True
    op = store.remove("source:remove")
    assert store.claim_next("writer").operation_id == op.operation_id

    assert store.complete_delete(op.operation_id, remote_absent=False, owner_id="writer") is False
    assert store.get_document("source:remove") is None
    assert store.search("keep until remote") == []
    assert store.next_operation().operation_id == op.operation_id
    assert store.complete_delete(op.operation_id, remote_absent=True, owner_id="writer") is True
    assert store.get_document("source:remove") is None
    assert store.next_operation() is None
    store.close()


def test_final_delete_purges_local_text_indexes_and_completed_payloads(tmp_path):
    store = LocalFallbackStore(tmp_path / "fallback.sqlite3")
    original = store.retain(
        "source:cancel", "sensitive cancelled ruby harbor", context="private context", tags=["cancelled"]
    )
    assert _finish_retain(store, original, "sensitive cancelled ruby harbor") is True
    delete = store.remove("source:cancel", bank_id="staging-bank")
    repeated = store.remove("source:cancel", bank_id="staging-bank")
    assert repeated.operation_id == delete.operation_id
    assert len(store.operations()) == 2
    assert store.claim_next("writer").operation_id == delete.operation_id

    assert store.complete_delete(delete.operation_id, remote_absent=True, owner_id="writer") is True

    assert store._db.execute("SELECT count(*) FROM documents WHERE document_id=?", (original.document_id,)).fetchone()[0] == 0
    assert store._db.execute("SELECT count(*) FROM source_mappings WHERE document_id=?", (original.document_id,)).fetchone()[0] == 0
    assert store._db.execute("SELECT count(*) FROM document_fts WHERE document_id=?", (original.document_id,)).fetchone()[0] == 0
    rows = store._db.execute(
        "SELECT status,payload,expected_remote_text,remote_options FROM outbox WHERE document_id=?",
        (original.document_id,),
    ).fetchall()
    assert len(rows) == 2
    assert all(row["status"] == "done" for row in rows)
    assert all(row["payload"] is None and row["expected_remote_text"] is None for row in rows)
    assert all(row["remote_options"] == "{}" for row in rows)
    assert store.search("sensitive cancelled ruby harbor") == []
    store.close()


def test_delete_ack_rolls_back_outbox_status_if_document_update_fails(tmp_path):
    store = LocalFallbackStore(tmp_path / "fallback.sqlite3")
    original = store.retain("source:ack-rollback", "rollback ack")
    assert _finish_retain(store, original, "rollback ack") is True
    deletion = store.remove("source:ack-rollback")
    assert store.claim_next("writer").operation_id == deletion.operation_id
    store._db.execute(
        "CREATE TRIGGER fail_delete_ack BEFORE UPDATE OF deleted ON documents "
        "BEGIN SELECT RAISE(ABORT, 'injected ack failure'); END"
    )

    try:
        store.complete_delete(deletion.operation_id, remote_absent=True, owner_id="writer")
    except Exception as exc:
        assert "injected ack failure" in str(exc)
    else:
        raise AssertionError("injected ack failure should propagate")

    row = store._db.execute(
        "SELECT status,lease_owner FROM outbox WHERE operation_id=?", (deletion.operation_id,)
    ).fetchone()
    assert tuple(row) == ("pending", "writer")
    store._db.execute("DROP TRIGGER fail_delete_ack")
    assert store.complete_delete(deletion.operation_id, remote_absent=True, owner_id="writer") is True
    store.close()


def test_add_mapping_rejects_document_already_owned_by_another_source(tmp_path):
    store = LocalFallbackStore(tmp_path / "fallback.sqlite3")
    original = store.retain("source:owner-a", "sole source owner")

    try:
        store.add_mapping("source:owner-b", original.document_id)
    except AmbiguousSourceError:
        pass
    else:
        raise AssertionError("one remote document must not map to multiple sources")

    assert store._db.execute(
        "SELECT count(*) FROM source_mappings WHERE document_id=?", (original.document_id,)
    ).fetchone()[0] == 1
    assert len(store.operations()) == 1
    store.close()


def test_shared_document_mapping_blocks_replace_and_remove_without_mutation(tmp_path):
    store = LocalFallbackStore(tmp_path / "fallback.sqlite3")
    original = store.retain("source:owner-a", "shared mapping keeps this text")
    store._db.execute(
        "INSERT INTO source_mappings(source_id,document_id) VALUES(?,?)",
        ("source:owner-b", original.document_id),
    )
    before_doc = tuple(store._db.execute(
        "SELECT document_id,source_id,text,deleted FROM documents WHERE document_id=?", (original.document_id,)
    ).fetchone())
    before_fts = [tuple(row) for row in store._db.execute(
        "SELECT document_id,source_id,text FROM document_fts WHERE document_id=?", (original.document_id,)
    ).fetchall()]
    before_outbox = [tuple(row) for row in store._db.execute(
        "SELECT operation_id,kind,payload,expected_remote_text FROM outbox ORDER BY sequence"
    ).fetchall()]

    for mutation in (
        lambda: store.retain("source:owner-a", "replacement must not happen"),
        lambda: store.remove("source:owner-a"),
    ):
        try:
            mutation()
        except AmbiguousSourceError:
            pass
        else:
            raise AssertionError("shared remote document must fail closed")

    after_doc = tuple(store._db.execute(
        "SELECT document_id,source_id,text,deleted FROM documents WHERE document_id=?", (original.document_id,)
    ).fetchone())
    after_fts = [tuple(row) for row in store._db.execute(
        "SELECT document_id,source_id,text FROM document_fts WHERE document_id=?", (original.document_id,)
    ).fetchall()]
    after_outbox = [tuple(row) for row in store._db.execute(
        "SELECT operation_id,kind,payload,expected_remote_text FROM outbox ORDER BY sequence"
    ).fetchall()]
    assert after_doc == before_doc
    assert after_fts == before_fts
    assert after_outbox == before_outbox
    store.close()


def test_commit_failure_rolls_back_and_closes_transaction(tmp_path):
    store = LocalFallbackStore(tmp_path / "fallback.sqlite3")
    connection = store._db

    class CommitFails:
        def execute(self, sql, parameters=()):
            if sql.strip().upper() == "COMMIT":
                raise sqlite3.OperationalError("injected commit failure")
            return connection.execute(sql, parameters)

    store._db = CommitFails()
    try:
        try:
            with store._transaction():
                connection.execute(
                    "INSERT INTO documents(document_id,source_id,text) VALUES('rollback-id','source:rollback','rollback text')"
                )
        except sqlite3.OperationalError as exc:
            assert "injected commit failure" in str(exc)
        else:
            raise AssertionError("injected commit failure should propagate")
        assert connection.in_transaction is False
        assert connection.execute(
            "SELECT count(*) FROM documents WHERE document_id='rollback-id'"
        ).fetchone()[0] == 0
    finally:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        store._db = connection
        store.close()


def test_ambiguous_mapping_blocks_replacement_and_removal_without_mutation(tmp_path):
    store = LocalFallbackStore(tmp_path / "fallback.sqlite3")
    initial = store.retain("source:ambiguous", "original")
    store.add_mapping("source:ambiguous", "remote-document-not-in-local-store")

    try:
        store.retain("source:ambiguous", "replacement")
    except AmbiguousSourceError:
        pass
    else:
        raise AssertionError("ambiguous replacement must fail closed")
    try:
        store.remove("source:ambiguous")
    except AmbiguousSourceError:
        pass
    else:
        raise AssertionError("ambiguous removal must fail closed")

    row = store._db.execute("SELECT text FROM documents WHERE document_id=?", (initial.document_id,)).fetchone()
    assert row["text"] == "original"
    assert [op.operation_id for op in store.operations()] == [initial.operation_id]
    store.close()


def test_multi_step_retain_rolls_back_document_index_and_outbox(tmp_path, monkeypatch):
    store = LocalFallbackStore(tmp_path / "fallback.sqlite3")

    def fail_index(*_args):
        raise RuntimeError("injected index failure")

    monkeypatch.setattr(store, "_index_document", fail_index)
    try:
        store.retain("source:rollback", "transaction should roll back")
    except RuntimeError as exc:
        assert str(exc) == "injected index failure"
    else:
        raise AssertionError("injected failure should propagate")

    assert store._db.execute("SELECT count(*) FROM documents").fetchone()[0] == 0
    assert store._db.execute("SELECT count(*) FROM source_mappings").fetchone()[0] == 0
    assert store.operations() == []
    store.close()


def test_two_instances_claim_fifo_head_once_and_expired_lease_recovers(tmp_path):
    path = tmp_path / "fallback.sqlite3"
    first_store = LocalFallbackStore(path)
    second_store = LocalFallbackStore(path)
    first = first_store.retain("source:first", "first")
    second = first_store.retain("source:second", "second")

    claim = first_store.claim_next("owner-one", lease_seconds=30)
    assert claim.operation_id == first.operation_id
    assert second_store.claim_next("owner-two", lease_seconds=5) is None
    assert second_store.complete_retain(first.operation_id, first.document_id, "first", owner_id="owner-two") is False
    first_store._db.execute("UPDATE outbox SET lease_until=0 WHERE operation_id=?", (first.operation_id,))
    recovered = second_store.claim_next("owner-two", lease_seconds=30)
    assert recovered.operation_id == first.operation_id
    assert first_store.release(first.operation_id, owner_id="owner-one") is False
    assert second_store.release(first.operation_id, owner_id="owner-two") is True
    recovered_again = first_store.claim_next("owner-three", lease_seconds=30)
    assert recovered_again.operation_id == first.operation_id
    assert second_store.complete_retain(first.operation_id, first.document_id, "first", owner_id="owner-two") is False
    assert first_store.complete_retain(first.operation_id, first.document_id, "first", owner_id="owner-three") is True
    assert second_store.claim_next("owner-three").operation_id == second.operation_id
    first_store.close()
    second_store.close()


def test_delete_then_retain_preserves_newer_local_document_and_search(tmp_path):
    store = LocalFallbackStore(tmp_path / "fallback.sqlite3")
    old = store.retain("source:revive", "obsolete lavender comet")
    assert _finish_retain(store, old, "obsolete lavender comet") is True
    deletion = store.remove("source:revive")
    assert store.search("lavender comet") == []
    replacement = store.retain("source:revive", "current emerald lighthouse")
    assert replacement.document_id == old.document_id
    assert store.get_document("source:revive").text == "current emerald lighthouse"
    assert [d.text for d in store.search("emerald lighthouse")] == ["current emerald lighthouse"]

    assert store.claim_next("writer").operation_id == deletion.operation_id
    assert store.complete_delete(deletion.operation_id, remote_absent=True, owner_id="writer") is True
    assert store.get_document("source:revive").text == "current emerald lighthouse"
    assert [d.text for d in store.search("emerald lighthouse")] == ["current emerald lighthouse"]
    replacement_row = store._db.execute(
        "SELECT status,payload,expected_remote_text FROM outbox WHERE operation_id=?", (replacement.operation_id,)
    ).fetchone()
    assert tuple(replacement_row) == ("pending", "current emerald lighthouse", "current emerald lighthouse")
    assert store.claim_next("writer").operation_id == replacement.operation_id
    store.close()


def test_retain_separates_remote_delta_expected_text_and_local_mirror(tmp_path):
    path = tmp_path / "append-fallback.sqlite3"
    store = LocalFallbackStore(path)
    operation = store.retain(
        "source:append",
        "cumulative local transcript",
        document_id="remote-session-id",
        remote_payload="second turn delta",
        expected_remote_text="cumulative remote transcript",
        bank_id="synthetic-bank",
        item_options={"update_mode": "append"},
    )
    store.close()

    reopened = LocalFallbackStore(path)
    operation = reopened.next_operation()
    assert operation.document_id == "remote-session-id"
    assert operation.payload == "second turn delta"
    assert operation.expected_remote_text == "cumulative remote transcript"
    assert operation.remote_options["item_options"] == {"update_mode": "append"}
    assert reopened.get_document("source:append").text == "cumulative local transcript"
    reopened.close()


def test_options_and_expected_text_are_immutable_across_reopen(tmp_path):
    path = tmp_path / "fallback.sqlite3"
    options = {
        "bank_id": "bank-staging",
        "context": "user preference",
        "metadata": {"source": "test", "turn": 7},
        "tags": ["preference", "offline"],
        "timestamp": "2026-11-24T11:00:00+01:00",
    }
    store = LocalFallbackStore(path)
    op = store.retain("source:options", "immutable expected text", **options)
    store.close()

    reopened = LocalFallbackStore(path)
    stored = reopened.next_operation()
    assert stored.operation_id == op.operation_id
    assert stored.expected_remote_text == "immutable expected text"
    assert stored.remote_options == options
    assert reopened.retain("source:options", "immutable expected text", **options).operation_id == op.operation_id
    assert len(reopened.operations()) == 1
    reopened.close()
