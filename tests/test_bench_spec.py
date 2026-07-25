from bench.spec import RunSpec


def test_run_id_deterministic_for_same_spec():
    a = RunSpec(model_name="xclip_hf_zeroshot", use_filters=True)
    b = RunSpec(model_name="xclip_hf_zeroshot", use_filters=True)
    assert a.run_id == b.run_id


def test_run_id_differs_for_different_filters():
    a = RunSpec(model_name="xclip_hf_zeroshot", use_filters=True)
    b = RunSpec(model_name="xclip_hf_zeroshot", use_filters=False)
    assert a.run_id != b.run_id


def test_run_id_differs_for_different_hardware_profile():
    a = RunSpec(model_name="xclip_hf_zeroshot", hardware_profile="cpu")
    b = RunSpec(model_name="xclip_hf_zeroshot", hardware_profile="gt1030_cuda")
    assert a.run_id != b.run_id


def test_as_dict_contains_all_fields():
    spec = RunSpec(model_name="siglip2_frameavg", hardware_profile="gt1030_cuda")
    d = spec.as_dict()
    assert d["model_name"] == "siglip2_frameavg"
    assert d["hardware_profile"] == "gt1030_cuda"
    assert d["use_filters"] is True
