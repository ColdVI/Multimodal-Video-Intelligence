from __future__ import annotations

from app.evaluation.hard_negatives import (
    SegmentCountRecord, attribute_reachability_precheck, build_tier_a_class_proximity_pairs,
    build_tier_a_count_difference_pairs,
)


def test_reachability_precheck_rejects_too_few_frames():
    assert attribute_reachability_precheck(frames_per_window=2, attribute="count_difference") is False


def test_reachability_precheck_accepts_structurally_sound_attributes():
    assert attribute_reachability_precheck(frames_per_window=6, attribute="count_difference") is True


def test_reachability_precheck_rejects_unlisted_fine_detail_attributes_by_default():
    """color/direction are NOT in the structural allow-list -- this function alone must
    never green-light a Tier C attribute; a human still has to look."""
    assert attribute_reachability_precheck(frames_per_window=6, attribute="color") is False
    assert attribute_reachability_precheck(frames_per_window=6, attribute="direction") is False


def test_count_difference_pairs_split_positives_from_zero_count_negatives():
    records = [
        SegmentCountRecord("s1", {"vehicle_count": 2}),
        SegmentCountRecord("s2", {"vehicle_count": 2}),
        SegmentCountRecord("s3", {"vehicle_count": 5}),
        SegmentCountRecord("s4", {"vehicle_count": 0}),
        SegmentCountRecord("s5", {"vehicle_count": 0}),
    ]
    queries = build_tier_a_count_difference_pairs(records, count_field="vehicle_count", id_prefix="hn")
    assert len(queries) == 2  # one per distinct non-zero count value
    by_id = {q.query_id: q for q in queries}
    assert set(by_id["hn_count_2"].positive_segment_ids) == {"s1", "s2"}
    assert set(by_id["hn_count_2"].hard_negative_segment_ids) == {"s4", "s5"}
    assert set(by_id["hn_count_5"].positive_segment_ids) == {"s3"}
    for query in queries:
        assert query.annotation_method == "auto_derived"
        assert query.evaluation == "binding"
        assert query.attribute_reachable is True


def test_count_difference_pairs_empty_when_no_zero_count_negatives_exist():
    records = [SegmentCountRecord("s1", {"vehicle_count": 3})]
    queries = build_tier_a_count_difference_pairs(records, count_field="vehicle_count", id_prefix="hn")
    assert queries == []


def test_class_proximity_pairs_positive_has_both_negative_has_only_secondary():
    records = [
        SegmentCountRecord("s1", {"person_count": 2, "vehicle_count": 1}),  # both -> positive
        SegmentCountRecord("s2", {"person_count": 0, "vehicle_count": 1}),  # vehicle only -> negative
        SegmentCountRecord("s3", {"person_count": 3, "vehicle_count": 0}),  # person only -> neither
    ]
    queries = build_tier_a_class_proximity_pairs(
        records, primary_field="person_count", secondary_field="vehicle_count", id_prefix="hn",
    )
    assert len(queries) == 1
    assert queries[0].positive_segment_ids == ("s1",)
    assert queries[0].hard_negative_segment_ids == ("s2",)


def test_class_proximity_pairs_empty_when_either_side_is_missing():
    records = [SegmentCountRecord("s1", {"person_count": 2, "vehicle_count": 1})]  # only "both", no secondary-only
    queries = build_tier_a_class_proximity_pairs(
        records, primary_field="person_count", secondary_field="vehicle_count", id_prefix="hn",
    )
    assert queries == []


def test_query_as_dict_matches_the_plan_example_shape():
    records = [
        SegmentCountRecord("s1", {"vehicle_count": 1}),
        SegmentCountRecord("s2", {"vehicle_count": 0}),
    ]
    query = build_tier_a_count_difference_pairs(records, count_field="vehicle_count", id_prefix="hn")[0]
    payload = query.as_dict()
    assert set(payload) == {
        "query_id", "query_tr", "query_en", "positive_segment_ids", "hard_negative_segment_ids",
        "tested_attributes", "annotation_method", "attribute_reachable", "evaluation",
    }
