from reports.capera_validation_html import render_capera_report


def _fake_evidence():
    return {
        "protocol": "CapERA T2V retrieval test protocol",
        "source": "test source",
        "dataset_manifest": {"dataset_id": "capera", "item_count": 2864, "query_count": 14320},
        "known_gaps": ["gap one", "gap two", "gap three"],
        "results": {
            "ModelA": {"n_videos": 100, "n_queries": 500, "embedding_dim": 512,
                      "recall_at_1": 0.2, "recall_at_5": 0.5, "recall_at_10": 0.6,
                      "mrr": 0.3, "peak_gpu_memory_mb": 1000.0},
            "ModelB": {"n_videos": 100, "n_queries": 500, "embedding_dim": 768,
                      "recall_at_1": 0.4, "recall_at_5": 0.7, "recall_at_10": 0.8,
                      "mrr": 0.5, "peak_gpu_memory_mb": 2000.0},
        },
    }


def test_render_includes_all_models_and_sorts_by_recall_at_1_descending():
    out = render_capera_report(_fake_evidence())
    assert "ModelA" in out
    assert "ModelB" in out
    assert out.index("ModelB") < out.index("ModelA")  # ModelB (0.4) > ModelA (0.2)


def test_render_includes_known_gaps():
    out = render_capera_report(_fake_evidence())
    assert "gap one" in out
    assert "gap two" in out


def test_render_includes_dataset_manifest():
    out = render_capera_report(_fake_evidence())
    assert "capera" in out
    assert "2864" in out


def test_scope_badge_html_included_when_provided():
    out = render_capera_report(_fake_evidence(), scope_badge_html="<div>MARKER_BADGE</div>")
    assert "MARKER_BADGE" in out


def test_scope_badge_html_absent_by_default_matches_no_badge_output():
    with_default = render_capera_report(_fake_evidence())
    with_explicit_empty = render_capera_report(_fake_evidence(), scope_badge_html="")
    assert with_default == with_explicit_empty
