from reports.msrvtt_validation_html import render_msrvtt_report


def _fake_evidence(red_flags=None):
    return {
        "protocol": "MSR-VTT 1k-A, standart t2v retrieval",
        "zero_shot_baseline": {
            "name": "CLIP-straight zero-shot (test)",
            "values": {"R@1": 31.2, "R@5": 53.7, "R@10": 64.2, "MedR": 4.0},
        },
        "reference_only_baselines": {
            "CLIP4Clip (fine-tuned)": {"R@1": 44.5, "R@5": 71.4, "R@10": 81.6, "MedR": 2.0},
        },
        "results": {
            "xclip_hf_zeroshot": {
                "measured": {
                    "R@1": 21.5, "R@5": 42.4, "R@10": 52.3, "MedR": 8.5, "MeanR": 75.056,
                    "n": 1000, "video_embed_total_s": 5438.9, "text_embed_total_s": 27.9,
                    "n_videos_embedded": 1000, "n_pairs_evaluated": 1000,
                    "mean_rank_vs_chance": "MeanR=75.1, rastgele sans ~500.5 -> 6.7x daha iyi",
                },
                "red_flags": red_flags or [],
            }
        },
    }


def test_render_includes_model_name_and_measured_values():
    out = render_msrvtt_report(_fake_evidence())
    assert "xclip_hf_zeroshot" in out
    assert "21.5" in out
    assert "6.7x daha iyi" in out


def test_render_shows_no_flags_message_when_clean():
    out = render_msrvtt_report(_fake_evidence(red_flags=[]))
    assert "kırmızı bayrak yok" in out.lower()
    assert "BAYRAKLI" not in out


def test_render_shows_flagged_rows_and_reasons_when_present():
    red_flags = [{
        "baseline": "CLIP-straight zero-shot (test)",
        "flags": ["R@5: olculen=42.4 yayinlanan=53.7 (fark 11.3 puan > 10.0 esigi)"],
    }]
    out = render_msrvtt_report(_fake_evidence(red_flags=red_flags))
    assert "BAYRAKLI" in out
    assert "fark 11.3 puan" in out


def test_render_includes_reference_only_baseline_table():
    out = render_msrvtt_report(_fake_evidence())
    assert "CLIP4Clip (fine-tuned)" in out
    assert "44.5" in out


def test_render_handles_multiple_models():
    evidence = _fake_evidence()
    evidence["results"]["qwen3vl_emb_2048"] = evidence["results"]["xclip_hf_zeroshot"]
    out = render_msrvtt_report(evidence)
    assert out.count("Zero-shot baseline karşılaştırması") == 2
