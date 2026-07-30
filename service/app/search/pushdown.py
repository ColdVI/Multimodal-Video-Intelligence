from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from app.search.filter_schema import CANONICAL_FILTER_FIELDS, FilterField


@dataclass(frozen=True)
class Predicate:
    field: str
    kind: str
    minimum: float | None = None
    maximum: float | None = None
    value: Any = None
    wrap: bool = False


def normalize_filters(
    metadata_filters: Mapping[str, Any] | None,
    telemetry_filters: Mapping[str, Any] | None,
    *,
    schema: Mapping[str, FilterField] = CANONICAL_FILTER_FIELDS,
) -> tuple[Predicate, ...]:
    result = []
    for key, raw in {**(metadata_filters or {}), **(telemetry_filters or {})}.items():
        if raw in (None, "", []):
            continue
        field = schema.get(key)
        if field is None or not field.filterable:
            raise ValueError(f"unknown or non-filterable field: {key}")
        if field.data_type in {"keyword", "boolean"}:
            result.append(Predicate(key, "equals", value=raw))
            continue
        if isinstance(raw, Mapping):
            minimum, maximum = raw.get("min"), raw.get("max")
            wrap = bool(raw.get("wrap", False))
        elif isinstance(raw, (list, tuple)) and len(raw) == 2:
            minimum, maximum, wrap = raw[0], raw[1], False
        else:
            minimum = maximum = raw
            wrap = False
        if field.data_type == "circular_deg":
            if minimum is not None and not 0 <= float(minimum) < 360:
                raise ValueError(f"{key} minimum must be in [0,360)")
            if maximum is not None and not 0 <= float(maximum) < 360:
                raise ValueError(f"{key} maximum must be in [0,360)")
            wrap = wrap or (
                minimum is not None and maximum is not None and float(minimum) > float(maximum)
            )
        result.append(Predicate(
            key, "range", None if minimum is None else float(minimum),
            None if maximum is None else float(maximum), wrap=wrap,
        ))
    return tuple(result)


def sql_predicates(
    predicates: Iterable[Predicate], columns: Mapping[str, str], *, placeholder: str = "%s",
) -> tuple[str, list[Any]]:
    clauses, params = [], []
    for item in predicates:
        column = columns[item.field]
        if item.kind == "equals":
            clauses.append(f"{column}={placeholder}")
            params.append(item.value)
        elif item.wrap and item.minimum is not None and item.maximum is not None:
            clauses.append(f"({column}>={placeholder} OR {column}<={placeholder})")
            params.extend([item.minimum, item.maximum])
        else:
            if item.minimum is not None:
                clauses.append(f"{column}>={placeholder}")
                params.append(item.minimum)
            if item.maximum is not None:
                clauses.append(f"{column}<={placeholder}")
                params.append(item.maximum)
    return (" AND ".join(clauses), params)


def matches(row: Mapping[str, Any], predicates: Iterable[Predicate]) -> bool:
    for item in predicates:
        value = row.get(item.field)
        if value is None:
            return False
        if item.kind == "equals" and value != item.value:
            return False
        if item.kind == "range":
            numeric = float(value)
            if item.wrap and item.minimum is not None and item.maximum is not None:
                if not (numeric >= item.minimum or numeric <= item.maximum):
                    return False
            elif (item.minimum is not None and numeric < item.minimum) or (
                item.maximum is not None and numeric > item.maximum
            ):
                return False
    return True


__all__ = ["Predicate", "matches", "normalize_filters", "sql_predicates"]
