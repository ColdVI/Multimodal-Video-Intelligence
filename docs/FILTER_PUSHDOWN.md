# Canonical filter registry and native pushdown

The production default is `FILTER_EXECUTION_MODE=pushdown`. For an active Faz
11 run, the API normalizes filter values once and passes the predicates to the
selected backend. It does not fetch the full matching segment-ID corpus into
Python.

Canonical filter types are float, integer, keyword, boolean, and circular
degrees. The run-scoped registry key is `(dataset_id, run_id, field_name)`, so a
staging manifest cannot change active UI bounds or semantics. Only meaningful
fields have physical indexes; the schema does not blindly index every column.

Circular filters explicitly support wrap-around:

```json
{"compass_heading": {"min": 350, "max": 10, "wrap": true}}
```

This compiles to `heading >= 350 OR heading <= 10`. A normal `10..350` interval
uses AND. Values outside `[0,360)` fail closed.

Backend behavior:

- ClickHouse stores the full canonical projection with real nullable columns,
  includes dataset/run predicates in the vector query, and computes candidate
  count with the same native filter.
- Qdrant stores run-scoped UUID5 point IDs and canonical payload, creates
  indexes from the canonical registry, and uses payload Filter/Range clauses.
- pgvector joins the run vector, segment, video, telemetry, and metadata tables
  in one SQL statement. It does not consume a large Python candidate list.
- Adaptive MRL reuses the same native predicate in the base stage. Only the
  bounded `top_n` IDs may enter reranking.

`legacy_candidate_ids` remains for benchmark comparison and active-run exact
equivalence checks. It is capped by `LEGACY_CANDIDATE_LIMIT`; exceeding the cap
fails before embedding/vector search. Datasets that have not yet been migrated
to an active run use an explicitly reported
`legacy_candidate_ids_compatibility` path.

Live exact equivalence command:

```bash
python -m app.search.equivalence \
  --dataset-id kurum_ucuslari --backend clickhouse --dimension 512 \
  --telemetry-filters '{"altitude_m":{"min":10,"max":30}}' \
  --output /workspace/artifacts/faz11/filter_equivalence.json
```

For exact search, ID order must match. ANN validation separately requires every
returned row to satisfy the predicate; recall/agreement is reported rather than
claiming exact ordering.
