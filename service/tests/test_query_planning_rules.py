from __future__ import annotations

from app.query_planning.rules import parse


def test_otobus_selects_bus_count_only_not_vehicle_count():
    parsed = parse("otobüs bulunan sahne")
    fields = {c.field for c in parsed.structured_constraints}
    assert fields == {"bus_count"}


def test_bus_english_synonym_matches_too():
    parsed = parse("a scene containing a bus")
    fields = {c.field for c in parsed.structured_constraints}
    assert fields == {"bus_count"}


def test_kamyon_widens_to_vehicle_count_with_diagnostic():
    parsed = parse("kamyon gören sahne")
    fields = {c.field for c in parsed.structured_constraints}
    assert fields == {"vehicle_count"}
    assert parsed.diagnostics.ontology_widened == ("truck->vehicle_count",)


def test_person_and_vehicle_both_present_are_independent_constraints():
    parsed = parse("yürüyen insanlar ve araçlar")
    fields = {c.field for c in parsed.structured_constraints}
    assert fields == {"person_count", "vehicle_count"}


def test_numeric_override_en_az_n():
    parsed = parse("en az 3 araç")
    constraint = next(c for c in parsed.structured_constraints if c.field == "vehicle_count")
    assert constraint.value == 3


def test_numeric_override_at_least_n_english():
    parsed = parse("at least 4 buses")
    constraint = next(c for c in parsed.structured_constraints if c.field == "bus_count")
    assert constraint.value == 4


def test_default_minimum_is_one_without_numeric_qualifier():
    parsed = parse("bir yaya")
    constraint = next(c for c in parsed.structured_constraints if c.field == "person_count")
    assert constraint.value == 1


def test_no_match_produces_no_constraints_and_full_query_residual():
    parsed = parse("boş yol")
    assert parsed.structured_constraints == ()
    assert parsed.semantic_residual == "boş yol"
    assert parsed.residual_policy == "full_query"


def test_all_constraints_are_soft_and_relaxable_with_rules_provenance():
    parsed = parse("gece görüntüsü")
    assert len(parsed.structured_constraints) == 1
    provenance = parsed.structured_constraints[0].provenance
    assert provenance.source == "rules_parser"
    assert provenance.hard is False
    assert provenance.relaxable is True


def test_diagnostics_report_rules_mode_and_latency():
    parsed = parse("otobüs")
    assert parsed.diagnostics.parser_mode == "rules"
    assert parsed.diagnostics.provider == "rules"
    assert parsed.diagnostics.latency_ms is not None
    assert parsed.diagnostics.latency_ms >= 0


def test_output_is_accepted_by_the_real_pushdown_layer():
    from app.query_planning.models import constraints_to_filter_dict
    from app.search.pushdown import normalize_filters

    parsed = parse("otobüs ve yaya")
    filter_dict = constraints_to_filter_dict(parsed.structured_constraints)
    predicates = normalize_filters(filter_dict, None)
    assert len(predicates) == 2
