from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE_ROOT = REPO_ROOT / "service"
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.config import Settings  # noqa: E402
from app.ingestion.manifest import load_manifest  # noqa: E402


def _read(*parts: str) -> str:
    return (REPO_ROOT / Path(*parts)).read_text(encoding="utf-8")


def test_env_example_contains_every_settings_key():
    """Every env var Settings.from_env() actually reads must be documented in
    .env.example (or intentionally absent because it's a pure-default DB/
    connection value not part of the institution onboarding surface)."""
    source = _read("service", "app", "config.py")
    documented = set(re.findall(r"^([A-Z][A-Z0-9_]+)=", _read(".env.example"), re.MULTILINE))
    # Keys read via values.get("NAME", ...) / _int/_float/_bool/_csv(values, "NAME", ...)
    referenced = set(re.findall(r'"([A-Z][A-Z0-9_]{2,})"', source))
    # Internal DB connection details are intentionally not institution-facing
    # onboarding keys in .env.example's curated list; everything else that
    # Settings.from_env reads must be documented.
    exempt = {
        "POSTGRES_HOST", "POSTGRES_PORT", "CLICKHOUSE_HOST", "CLICKHOUSE_PORT",
        "QDRANT_URL", "MILVUS_URI", "DEFAULT_TOP_K", "REQUIRE_SECURE_CREDENTIALS",
        "QWEN_TEXT_WARM_LIMIT_S", "QWEN_TEXT_WARM_RUNS",
        "PROJECT_CONFIG_PATH",  # optional legacy CapERA quality-protocol path, not part of the FAZ11 profile
    }
    missing = referenced - documented - exempt
    assert not missing, f".env.example is missing documented keys: {sorted(missing)}"


def test_env_example_keys_are_all_real_settings_fields():
    """The inverse check: .env.example must not advertise a key nothing in the
    codebase reads - either Settings.from_env() (Python), a docker-compose*.yml
    ${NAME} interpolation, or the UI's direct os.environ read (API_URL)."""
    documented = set(re.findall(r"^([A-Z][A-Z0-9_]+)=", _read(".env.example"), re.MULTILINE))
    referenced: set[str] = set()
    referenced |= set(re.findall(r'"([A-Z][A-Z0-9_]{2,})"', _read("service", "app", "config.py")))
    referenced |= set(re.findall(r'"([A-Z][A-Z0-9_]{2,})"', _read("service", "ui", "app.py")))
    for compose_file in ("docker-compose.yml", "docker-compose.gpu.yml", "docker-compose.benchmark.yml", "docker-compose.debug.yml"):
        referenced |= set(re.findall(r"\$\{([A-Z][A-Z0-9_]{2,})[:}]", _read(compose_file)))
    unreal = documented - referenced
    assert not unreal, f".env.example documents keys nothing in the codebase reads: {sorted(unreal)}"


def test_env_example_has_no_real_secrets():
    text = _read(".env.example")
    for line in text.splitlines():
        if line.startswith("POSTGRES_PASSWORD=") or line.startswith("CLICKHOUSE_PASSWORD=") or line.startswith("MODEL_BUNDLE_ROOT="):
            assert "CHANGE_ME" in line, f"placeholder line must still read CHANGE_ME: {line}"


def test_example_manifests_parse():
    for name in ("example_uav.yaml", "example_institution.yaml"):
        manifest = load_manifest(REPO_ROOT / "datasets" / name)
        assert manifest.dataset_id


def test_colab_notebook_is_valid_json_and_nbformat4():
    path = REPO_ROOT / "notebooks" / "08_colab_portable_runner.ipynb"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    assert len(payload["cells"]) > 5
    for cell in payload["cells"]:
        assert cell["cell_type"] in ("markdown", "code")
        assert "id" in cell


def test_colab_notebook_pins_the_same_model_revision_as_env_example():
    notebook_text = (REPO_ROOT / "notebooks" / "08_colab_portable_runner.ipynb").read_text(encoding="utf-8")
    env_text = _read(".env.example")
    revision = re.search(r"QWEN_MODEL_REVISION=(\S+)", env_text).group(1)
    commit = re.search(r"QWEN_SOURCE_COMMIT=(\S+)", env_text).group(1)
    assert revision in notebook_text
    assert commit in notebook_text


def test_runbook_documented_commands_reference_real_scripts_and_flags():
    for doc_name, required_snippets in {
        "USER_GUIDE.md": ["scripts/preflight.py", "scripts/prepare_model_bundle.py", "scripts/migrate_faz11_schema.py", "app.ingestion.ingest", "app.ingestion.gc_runs"],
        "OPERATOR_QUICKSTART.md": ["scripts/preflight.py", "scripts/prepare_model_bundle.py", "scripts/run_faz11_acceptance.py"],
        "TARGET_ENVIRONMENT_ACCEPTANCE.md": ["scripts/run_faz11_acceptance.py"],
        "COLAB_RUNBOOK.md": ["notebooks/08_colab_portable_runner.ipynb"],
    }.items():
        text = _read("docs", doc_name)
        for snippet in required_snippets:
            assert snippet in text, f"docs/{doc_name} should reference {snippet!r}"


def test_documented_scripts_and_referenced_paths_exist():
    for doc_name in (
        "USER_GUIDE.md", "OPERATOR_QUICKSTART.md", "END_USER_GUIDE.md",
        "DATASET_ONBOARDING_GUIDE.md", "COLAB_RUNBOOK.md", "TARGET_ENVIRONMENT_ACCEPTANCE.md",
    ):
        text = _read("docs", doc_name)
        for match in re.findall(r"`(scripts/[a-zA-Z0-9_./-]+\.py)`", text):
            assert (REPO_ROOT / match).is_file(), f"docs/{doc_name} references missing file {match}"
        # datasets/kurum.yaml is intentionally a placeholder the operator creates
        # (`cp datasets/example_uav.yaml datasets/kurum.yaml`) - only the shipped
        # example_*.yaml manifests are expected to already exist in the repo.
        for match in re.findall(r"`(datasets/example_[a-zA-Z0-9_./-]+\.yaml)`", text):
            assert (REPO_ROOT / match).is_file(), f"docs/{doc_name} references missing file {match}"
        for match in re.findall(r"`(notebooks/[a-zA-Z0-9_./-]+\.ipynb)`", text):
            assert (REPO_ROOT / match).is_file(), f"docs/{doc_name} references missing file {match}"


def test_run_faz11_acceptance_cli_flags_match_documentation():
    result = subprocess.run(
        [sys.executable, "scripts/run_faz11_acceptance.py", "--help"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    for flag in ("--dataset", "--env-file", "--model-bundle-root", "--bind-host", "--live", "--output"):
        assert flag in result.stdout


def test_gc_runs_documented_flags_match_real_parser():
    text = _read("docs", "USER_GUIDE.md")
    source = _read("service", "app", "ingestion", "gc_runs.py")
    real_flags = set(re.findall(r'add_argument\("(--[a-z-]+)"', source))
    assert {"--dry-run", "--retain-previous-completed", "--min-age-hours"} <= real_flags
    assert "gc_runs" in text and "--dry-run" in text


def test_preflight_documented_exit_codes_match_real_mapping():
    text = _read("docs", "USER_GUIDE.md")
    source = _read("service", "app", "preflight.py")
    mapping = dict(re.findall(r'\("(\w+)",\s*(\d)\)', source))
    for category, code in mapping.items():
        assert code in text, f"USER_GUIDE.md should mention exit code {code} for category {category}"
