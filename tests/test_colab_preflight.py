import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scripts import colab_preflight


def test_check_binary_finds_python_itself():
    result = colab_preflight.check_binary("python")
    assert isinstance(result["found"], bool)


def test_check_binary_reports_missing_for_nonexistent_command():
    result = colab_preflight.check_binary("definitely_not_a_real_binary_xyz123")
    assert result["found"] is False
    assert result["path"] is None


def test_check_port_free_returns_bool():
    # yuksek, nadiren kullanilan port - CI'da genelde bos
    assert isinstance(colab_preflight.check_port_free(58234), bool)


def test_check_os_arch_never_crashes_and_has_expected_keys():
    result = colab_preflight.check_os_arch()
    assert "system" in result and "machine" in result


def test_build_report_never_crashes_and_is_json_serializable():
    """gercek kontrol: build_report() Windows/Linux HERHANGI bir platformda
    cokmemeli - eksik arac/izin durumunda False/unknown DONMELI, exception
    FIRLATMAMALI (Colab preflight'in butun amaci budur)."""
    report = colab_preflight.build_report()
    serialized = json.dumps(report)  # crash etmemeli
    assert "os_arch" in report
    assert "docker" in report
    assert "ports_free" in report


def test_check_url_reachable_handles_invalid_url_gracefully():
    result = colab_preflight.check_url_reachable("https://this-domain-does-not-exist-xyz123.invalid")
    assert result["reachable"] is False
    assert "error" in result
