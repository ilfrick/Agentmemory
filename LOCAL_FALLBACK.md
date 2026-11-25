# Local fallback and temporal policy

This repository contains a Hermes memory-provider fork based on Hindsight. The fallback code is **not** installed or activated in Nicola's live Hermes profile by this change; only the temporal strategy patch is installed there.

## What the fallback stores and does

- Eligible retained session turns are written to a local SQLite store at `$HERMES_HOME/hindsight/local-fallback.sqlite3` before remote replay. This is a durable local copy of the retained conversation text, not merely a temporary outage cache; treat the database and its WAL sidecars as sensitive user data.
- The store uses SQLite WAL/FULL durability, a FIFO outbox, leased operations, stable operation IDs, and restrictive file modes. Remote retain/delete operations are acknowledged only after exact remote readback or verified absence.
- When Hindsight recall fails, the provider can return a small local FTS5 keyword-search result set. It is not semantic/vector search and does not call an LLM. Returned text is labeled as untrusted memory data.
- The existing retain cadence and the provider's exclusions still govern automatic turn capture. In this fork, automatic retention is also fail-closed behind `temporal_gate_passed`; it remains off until that gate is explicitly set after temporal acceptance tests.

## Temporal policy

For items with `timestamp: "unset"`, configure the per-item strategy `retain_unset_timestamp_strategy` and define the same strategy name in the Hindsight bank's `retain_strategies`. Explicit event timestamps keep the general `retain_strategy`. With an unknown event-date anchor, the strategy preserves the wording and leaves both occurrence bounds unset rather than inventing a date.

## Operational boundary

The tests use synthetic fixtures and fake clients. They do not require credentials or read a personal bank. Installing this fork activates a local persistent copy of retained content; review that data-retention scope and choose an appropriate backup/retention policy before activation. This repository deliberately excludes live `config.json`, `.env` files, Hindsight banks, databases, and run reports.
