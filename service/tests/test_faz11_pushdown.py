from __future__ import annotations

import pytest

from app.search.pushdown import matches, normalize_filters, sql_predicates


def test_circular_range_wrap_compiles_as_or_and_matches_both_sides():
    predicates = normalize_filters({}, {"compass_heading": {"min": 350, "max": 10, "wrap": True}})
    sql, params = sql_predicates(predicates, {"compass_heading": "t.compass_heading"})
    assert sql == "(t.compass_heading>=%s OR t.compass_heading<=%s)"
    assert params == [350.0, 10.0]
    assert matches({"compass_heading": 355}, predicates)
    assert matches({"compass_heading": 5}, predicates)
    assert not matches({"compass_heading": 180}, predicates)


def test_normal_circular_range_uses_and():
    predicates = normalize_filters({}, {"yaw": {"min": 10, "max": 350}})
    sql, params = sql_predicates(predicates, {"yaw": "t.yaw"})
    assert sql == "t.yaw>=%s AND t.yaw<=%s"
    assert params == [10.0, 350.0]


def test_keyword_boolean_integer_and_float_filters_are_normalized():
    predicates = normalize_filters(
        {"event_category": "traffic"},
        {"is_night": True, "person_count": [2, 8], "altitude_m": {"min": 10, "max": 30}},
    )
    row = {"event_category": "traffic", "is_night": True, "person_count": 4, "altitude_m": 20}
    assert matches(row, predicates)
    assert not matches({**row, "person_count": 9}, predicates)


def test_unknown_filter_and_invalid_heading_fail_closed():
    with pytest.raises(ValueError, match="unknown"):
        normalize_filters({}, {"invented": [0, 1]})
    with pytest.raises(ValueError, match=r"\[0,360\)"):
        normalize_filters({}, {"yaw": {"min": -1, "max": 10}})
