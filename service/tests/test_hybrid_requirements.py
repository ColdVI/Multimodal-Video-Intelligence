from pathlib import Path
import re

from packaging.version import Version


ROOT = Path(__file__).resolve().parents[2] / "service"


def _pin(path: Path, package: str) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(rf"{re.escape(package)}(?:\[[^]]+\])?==(.+)", line)
        if match:
            return match.group(1)
    raise AssertionError(f"missing exact pin for {package} in {path.name}")


def test_api_and_ui_dependency_sets_keep_incompatible_hub_ranges_separate():
    common = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "gradio==" not in common
    assert "huggingface-hub" not in common

    api_hub = _pin(ROOT / "requirements-real.txt", "huggingface-hub")
    ui_hub = _pin(ROOT / "requirements-ui.txt", "huggingface-hub")
    assert Version("0.34.0") <= Version(api_hub) < Version("1.0")
    assert Version("1.2.0") <= Version(ui_hub) < Version("2.0")
    assert _pin(ROOT / "requirements-ui.txt", "gradio") == "6.20.0"


def test_ui_dockerfile_installs_the_ui_specific_requirements():
    dockerfile = (ROOT / "Dockerfile.ui").read_text(encoding="utf-8")
    assert "requirements-ui.txt" in dockerfile
    assert "pip install --no-cache-dir -r requirements-ui.txt" in dockerfile
