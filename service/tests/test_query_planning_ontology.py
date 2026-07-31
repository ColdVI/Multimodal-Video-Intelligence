from __future__ import annotations

from app.query_planning.ontology import ONTOLOGY, ontology_widened_notes, resolve_specificity


def test_more_specific_concept_in_same_family_wins_and_elides_the_broader_one():
    """Plan example: 'otobüs' must select bus_count alone, never also add a redundant
    vehicle_count>=1 -- simulates a matcher that (over-eagerly) flagged both concepts."""
    resolution = resolve_specificity([ONTOLOGY["vehicle"], ONTOLOGY["bus"]])
    assert resolution.selected == (ONTOLOGY["bus"],)
    assert resolution.elided == (ONTOLOGY["vehicle"],)
    assert resolution.widened == ()


def test_concepts_in_different_families_never_compete():
    resolution = resolve_specificity([ONTOLOGY["bus"], ONTOLOGY["person"]])
    assert set(resolution.selected) == {ONTOLOGY["bus"], ONTOLOGY["person"]}
    assert resolution.elided == ()


def test_tied_specificity_within_a_family_keeps_both():
    resolution = resolve_specificity([ONTOLOGY["bus"], ONTOLOGY["truck"]])  # both specificity=2
    assert set(resolution.selected) == {ONTOLOGY["bus"], ONTOLOGY["truck"]}
    assert resolution.elided == ()


def test_truck_concept_is_marked_widened_to_vehicle_count():
    resolution = resolve_specificity([ONTOLOGY["truck"]])
    assert resolution.selected == (ONTOLOGY["truck"],)
    assert resolution.widened == (ONTOLOGY["truck"],)
    assert ontology_widened_notes(resolution) == ("truck->vehicle_count",)


def test_no_widening_note_for_concepts_with_their_own_column():
    resolution = resolve_specificity([ONTOLOGY["bus"], ONTOLOGY["person"]])
    assert ontology_widened_notes(resolution) == ()


def test_every_ontology_concept_targets_a_real_canonical_field():
    from app.search.filter_schema import CANONICAL_FILTER_FIELDS

    for concept in ONTOLOGY.values():
        assert concept.canonical_field in CANONICAL_FILTER_FIELDS


def test_empty_match_list_resolves_to_nothing():
    resolution = resolve_specificity([])
    assert resolution.selected == ()
    assert resolution.elided == ()
    assert resolution.widened == ()
