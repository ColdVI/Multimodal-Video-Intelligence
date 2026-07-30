# Run-versioned ingest, recovery, and migration

Faz 11 keeps the currently active dataset queryable while a new ingest is in
progress. Legacy tables remain intact for compatibility. New ingests use the
additive `ingest_runs`, `dataset_active_runs`, `ingest_chunks`, `run_*` metadata
tables, run-scoped pgvector/ClickHouse tables, and run-scoped Qdrant point IDs.

## State and activation contract

Run states are `created`, `preflight_passed`, `ingesting`, `validating`,
`completed`, `failed`, and `aborted`. Chunk states are `pending`, `writing`,
`committed`, and `failed`.

A chunk retry first verifies that its run is inactive, removes only the same
run/chunk from enabled vector backends, and then marks the ledger `writing`.
The chunk becomes `committed` only when PostgreSQL metadata and every enabled
backend/dimension report the expected count. Destructive cleanup of an active
run is rejected.

Finalize checks all chunks, PostgreSQL segment count, duplicate segment IDs,
and every enabled backend/dimension. A mismatch marks the staging run failed
and does not change `dataset_active_runs`. Success changes run state and the
active pointer in one PostgreSQL transaction. Cross-backend writes happen
before that transaction; they are never presented as a distributed two-phase
commit.

Every search reads one active run snapshot before filtering. The same `run_id`
is reused for filter compilation, vector search, hydration, and diagnostics,
even if another run activates during the request. Legacy datasets without an
active pointer continue through the existing tables.

## Existing volume migration

Plan first; this is read-only:

```bash
PYTHONPATH=service python scripts/migrate_faz11_schema.py --plan \
  --output artifacts/faz11/schema_migration_report.json
```

`--dry-run` is an explicit alias for a reported non-applying plan. After the
operator has taken an environment-appropriate backup/snapshot and reviewed row
counts, apply with:

```bash
PYTHONPATH=service python scripts/migrate_faz11_schema.py --apply \
  --output artifacts/faz11/schema_migration_report.json
```

The migration creates a deterministic legacy run per dataset, copies into new
run-scoped tables, validates metadata/vector counts, and only then activates
the legacy run. It does not DROP, rename, truncate, delete a volume, or mutate
the old tables. Qdrant legacy points cannot be given new run-scoped UUIDs
without source provenance; when Qdrant is enabled the migration reports
manifest-driven re-ingest as required and does not change active pointers.

## Resume and garbage collection

Resume is chunk-scoped; deterministic segment IDs remain independent of run ID,
while physical uniqueness is `(run_id, segment_id)`. Failed writes therefore
cannot duplicate committed rows on retry.

Always inspect GC first:

```bash
PYTHONPATH=service python -m app.ingestion.gc_runs --dry-run \
  --retain-previous-completed 1 --min-age-hours 24
```

Omit `--dry-run` only after review. Active, ingesting, and validating runs are
never candidates. The default retains the active run plus one previous
completed run, and protects all runs younger than 24 hours. Deleted row counts
are reported per backend. Peak vector storage during ingest is approximately
active + staging (about 2x); retaining a previous completed run can temporarily
increase this according to backend/dimension selection and actual compression.
