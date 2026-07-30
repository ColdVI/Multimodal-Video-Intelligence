from __future__ import annotations

import json
from typing import Iterable

from app.db import postgres
from app.search.filter_schema import FilterField


def replace_fields(dataset_id: str, run_id: str, fields: Iterable[FilterField]) -> None:
    rows = [(
        run_id, dataset_id, field.name, field.source, field.data_type, field.unit,
        json.dumps({"display_name": field.display_name, "filterable": field.filterable, "indexed": field.indexed}),
    ) for field in fields]
    with postgres.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM dataset_active_runs WHERE active_run_id=%s", (run_id,))
        if cur.fetchone():
            raise ValueError("active run filter registry cannot be replaced")
        cur.execute("DELETE FROM telemetry_field_registry WHERE dataset_id=%s AND run_id=%s", (dataset_id, run_id))
        postgres._execute_values(
            cur,
            """INSERT INTO telemetry_field_registry(
                 run_id,dataset_id,field_name,source_name,field_type,unit,semantics
               ) VALUES %s""",
            rows,
        )
        conn.commit()


def fields_for_run(dataset_id: str, run_id: str) -> list[dict]:
    with postgres.connection() as conn:
        _, extras = postgres._driver()
        with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
            cur.execute(
                """SELECT field_name AS name,source_name AS source,field_type AS data_type,unit,
                          semantics->>'display_name' AS display_name,
                          coalesce((semantics->>'filterable')::boolean,false) AS filterable,
                          coalesce((semantics->>'indexed')::boolean,false) AS indexed
                   FROM telemetry_field_registry WHERE dataset_id=%s AND run_id=%s ORDER BY field_name""",
                (dataset_id, run_id),
            )
            return [dict(row) for row in cur.fetchall()]


__all__ = ["fields_for_run", "replace_fields"]
