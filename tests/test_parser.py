from search.parser import parse


def test_bus_query_extracts_filter():
    p = parse("otobüsü göster")
    assert ("bus_count", ">=", 1) in p.filters
    assert p.semantic == "otobüsü göster"


def test_compound_query_extracts_both_filters():
    p = parse("otobüs ve yürüyen adam")
    cols = {c for c, _, _ in p.filters}
    assert "bus_count" in cols
    assert "person_count" in cols


def test_motion_only_query_has_no_extra_filter():
    p = parse("yürüyen adamı göster")
    cols = {c for c, _, _ in p.filters}
    assert cols == {"person_count"}


def test_semantic_always_full_query():
    q = "gece otobüs ve kalabalık"
    p = parse(q)
    assert p.semantic == q


def test_case_insensitive():
    p = parse("OTOBÜSÜ GÖSTER")
    assert ("bus_count", ">=", 1) in p.filters
