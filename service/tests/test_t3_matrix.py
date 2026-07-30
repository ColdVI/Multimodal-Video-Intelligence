from app.bench.protocol import result_interpretable


def test_t3_synthetic_ann_recall_is_not_interpretable():
    assert result_interpretable("synthetic", "ann", "ann_recall_vs_exact") is False
    assert result_interpretable("synthetic", "exact", "ann_recall_vs_exact") is True
    assert result_interpretable("synthetic", "exact", "topk_agreement") is True
    assert result_interpretable("cached", "ann", "ann_recall_vs_exact") is True
