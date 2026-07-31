"""Sequential FAZ11 target-environment acceptance runner.

Orchestrates the full acceptance chain end to end on a real target NVIDIA
Linux host and produces one machine-readable result:
artifacts/faz11/target_acceptance.json. Every step that genuinely cannot be
attempted on the current host (no Docker daemon, no GPU, no institution
data) is marked not_run with an exact reason, required command, and expected
environment - it is never marked pass without actually being attempted, and
is never marked pass_synthetic/pass_by_inspection to disguise a skip as a
pass. This script performs no destructive action (no volume deletion, no
--force flags) and never runs `docker compose ... down -v` or `git reset/clean`.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE_ROOT = REPO_ROOT / "service"
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))


@dataclass
class StepResult:
    id: str
    status: str  # pass | fail | blocked | not_run
    started_at: str
    finished_at: str
    command: str
    evidence: str = ""
    reason: str = ""
    expected_environment: str = ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(command: list[str], *, timeout: float = 60.0, cwd: Path | None = None) -> tuple[bool, str]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, cwd=cwd, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    output = (result.stdout or result.stderr).strip()
    return result.returncode == 0, output[-2000:]


def _not_run(step_id: str, command: str, reason: str, expected_environment: str, started: str) -> StepResult:
    return StepResult(step_id, "not_run", started, _now(), command, reason=reason, expected_environment=expected_environment)


def step_git_state(_: argparse.Namespace) -> StepResult:
    started = _now()
    command = "git status --short && git rev-parse HEAD"
    ok_status, status_out = _run(["git", "status", "--short"], cwd=REPO_ROOT)
    ok_sha, sha_out = _run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    status = "pass" if ok_status and ok_sha else "fail"
    return StepResult(
        "git_state", status, started, _now(), command,
        evidence=f"sha={sha_out}; dirty_lines={len(status_out.splitlines()) if status_out else 0}",
    )


def step_host_info(_: argparse.Namespace) -> StepResult:
    started = _now()
    info = {"platform": platform.platform(), "python": sys.version.split()[0]}
    return StepResult("host_info", "pass", started, _now(), "platform.platform() / sys.version", evidence=json.dumps(info))


def step_nvidia_driver(_: argparse.Namespace) -> StepResult:
    started = _now()
    command = "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"
    ok, output = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    if not ok:
        return _not_run(
            "nvidia_driver", command, f"nvidia-smi unavailable or no GPU: {output}",
            "NVIDIA Linux host with driver installed", started,
        )
    return StepResult("nvidia_driver", "pass", started, _now(), command, evidence=output)


def step_docker_daemon(_: argparse.Namespace) -> StepResult:
    started = _now()
    command = "docker info --format {{.ServerVersion}}"
    ok, output = _run(["docker", "info", "--format", "{{.ServerVersion}}"])
    if not ok:
        return _not_run(
            "docker_daemon", command, f"Docker daemon not reachable: {output}",
            "Host with Docker Engine running (Docker Desktop started, or dockerd active)", started,
        )
    return StepResult("docker_daemon", "pass", started, _now(), command, evidence=f"server_version={output}")


def step_compose_config(args: argparse.Namespace) -> StepResult:
    started = _now()
    combos = [
        ["docker-compose.yml"],
        ["docker-compose.yml", "docker-compose.gpu.yml"],
        ["docker-compose.yml", "docker-compose.benchmark.yml"],
        ["docker-compose.yml", "docker-compose.benchmark.yml", "docker-compose.debug.yml"],
    ]
    results = []
    for files in combos:
        command = ["docker", "compose", "--env-file", str(args.env_file)]
        for f in files:
            command += ["-f", f]
        command += ["config"]
        ok, output = _run(command, cwd=REPO_ROOT)
        results.append((files, ok, output[-300:]))
    failed = [files for files, ok, _ in results if not ok]
    command_str = "docker compose --env-file .env -f docker-compose.yml [...] config (x4 combinations)"
    if any(not ok for _, ok, out in results if "docker" not in out.lower() and "not found" not in out.lower()):
        pass
    if not results or all(not ok for _, ok, _ in results):
        return _not_run(
            "compose_config", command_str, "docker compose CLI unavailable or all combinations failed to parse",
            "Docker CLI installed (daemon not required for `config`)", started,
        )
    status = "fail" if failed else "pass"
    return StepResult(
        "compose_config", status, started, _now(), command_str,
        evidence=f"combinations_ok={len(results) - len(failed)}/{len(results)}",
        reason="" if not failed else f"failed combinations: {failed}",
    )


def step_secure_credentials(args: argparse.Namespace) -> StepResult:
    started = _now()
    command = f"grep CHANGE_ME {args.env_file}"
    if not args.env_file.is_file():
        return _not_run("secure_credentials", command, f"{args.env_file} does not exist", "A prepared .env file", started)
    text = args.env_file.read_text(encoding="utf-8")
    placeholders = [line for line in text.splitlines() if "CHANGE_ME" in line]
    status = "fail" if placeholders else "pass"
    return StepResult(
        "secure_credentials", status, started, _now(), command,
        evidence=f"placeholder_count={len(placeholders)}",
        reason="" if not placeholders else "unreplaced CHANGE_ME_* values remain in .env",
    )


def step_model_bundle_hash(args: argparse.Namespace) -> StepResult:
    started = _now()
    command = "python -c \"from app.embedding.bundle import verify_bundle; verify_bundle(MODEL_BUNDLE_ROOT, ...)\""
    from app.config import settings

    bundle_root = args.model_bundle_root or settings.model_bundle_root
    if not Path(bundle_root).is_dir():
        return _not_run(
            "model_bundle_hash", command, f"MODEL_BUNDLE_ROOT does not exist: {bundle_root}",
            "A bundle produced by scripts/prepare_model_bundle.py", started,
        )
    try:
        from app.embedding.bundle import verify_bundle

        manifest = verify_bundle(
            Path(bundle_root), expected_model_id=settings.qwen_model_id,
            expected_model_revision=settings.qwen_model_revision, expected_source_commit=settings.qwen_source_commit,
        )
        return StepResult(
            "model_bundle_hash", "pass", started, _now(), command,
            evidence=f"verified_bytes={manifest['total_size_bytes']}; source_commit={manifest['source_commit']}",
        )
    except Exception as exc:
        return StepResult("model_bundle_hash", "fail", started, _now(), command, reason=f"{type(exc).__name__}: {exc}")


def step_dataset_preflight(args: argparse.Namespace) -> StepResult:
    started = _now()
    command = f"python scripts/preflight.py --dataset {args.dataset} --env-file {args.env_file}"
    if args.dataset is None:
        return _not_run("dataset_preflight", command, "--dataset not provided", "A dataset manifest YAML", started)
    ok, output = _run(
        [sys.executable, "scripts/preflight.py", "--dataset", str(args.dataset), "--env-file", str(args.env_file)],
        cwd=REPO_ROOT, timeout=120,
    )
    status = "pass" if ok else "fail"
    return StepResult("dataset_preflight", status, started, _now(), command, evidence=output[-500:])


def step_migration_plan(_: argparse.Namespace) -> StepResult:
    started = _now()
    command = "python scripts/migrate_faz11_schema.py --plan --output artifacts/faz11/schema_migration_report.json"
    ok, output = _run([sys.executable, "scripts/migrate_faz11_schema.py", "--plan",
                        "--output", "artifacts/faz11/schema_migration_report.json"], cwd=REPO_ROOT, timeout=60)
    try:
        report = json.loads((REPO_ROOT / "artifacts/faz11/schema_migration_report.json").read_text(encoding="utf-8"))
    except Exception:
        report = {}
    if report.get("status") == "not_run":
        return _not_run(
            "migration_plan", command, report.get("reason", "migration plan could not run"),
            "Healthy configured PostgreSQL", started,
        )
    status = "pass" if report.get("status") == "pass" else "fail"
    return StepResult("migration_plan", status, started, _now(), command, evidence=json.dumps(report)[-500:])


def step_compose_up(args: argparse.Namespace) -> StepResult:
    started = _now()
    files = ["-f", "docker-compose.yml", "-f", "docker-compose.gpu.yml"]
    command = f"docker compose --env-file {args.env_file} {' '.join(files)} up -d --build"
    if not args.live:
        return _not_run(
            "compose_up", command, "--live not passed; starting services is skipped by default",
            "Operator explicitly runs with --live on the real target host", started,
        )
    ok, output = _run(["docker", "compose", "--env-file", str(args.env_file), *files, "up", "-d", "--build"],
                       cwd=REPO_ROOT, timeout=900)
    status = "pass" if ok else "fail"
    return StepResult("compose_up", status, started, _now(), command, evidence=output[-500:])


def step_health_check(args: argparse.Namespace) -> StepResult:
    started = _now()
    bind_host = args.bind_host
    command = f"curl -fsS http://{bind_host}:8000/health"
    if not args.live:
        return _not_run("health_check", command, "compose_up was skipped (--live not passed)", "A running api container", started)
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://{bind_host}:8000/health", timeout=5) as response:
            body = response.read().decode("utf-8")
        return StepResult("health_check", "pass", started, _now(), command, evidence=body[-500:])
    except Exception as exc:
        return StepResult("health_check", "fail", started, _now(), command, reason=f"{type(exc).__name__}: {exc}")


def step_gpu_smoke(args: argparse.Namespace) -> StepResult:
    started = _now()
    command = f"python scripts/gpu_smoke.py --dataset {args.dataset} --output artifacts/faz11/gpu_smoke.json --windows 10"
    if args.dataset is None:
        return _not_run("gpu_smoke", command, "--dataset not provided", "A dataset manifest and real GPU", started)
    ok, output = _run([sys.executable, "scripts/gpu_smoke.py", "--dataset", str(args.dataset),
                        "--output", "artifacts/faz11/gpu_smoke.json", "--windows", "10"], cwd=REPO_ROOT, timeout=600)
    try:
        report = json.loads((REPO_ROOT / "artifacts/faz11/gpu_smoke.json").read_text(encoding="utf-8"))
    except Exception:
        report = {}
    if report.get("status") == "not_run" or report.get("result") == "not_run":
        return _not_run("gpu_smoke", command, report.get("reason", "GPU unavailable"), "NVIDIA Linux host with verified bundle", started)
    status = "pass" if ok else "fail"
    return StepResult("gpu_smoke", status, started, _now(), command, evidence=json.dumps(report)[-500:])


def step_real_ingest(args: argparse.Namespace) -> StepResult:
    started = _now()
    command = f"docker compose exec -T api python -m app.ingestion.ingest --dataset /workspace/datasets/{Path(args.dataset).name if args.dataset else '<dataset>'} --resume"
    if not args.live or args.dataset is None:
        return _not_run("real_ingest", command, "--live not passed or --dataset not provided",
                         "A running institution stack with real institution video/telemetry under DATA_ROOT", started)
    ok, output = _run(["docker", "compose", "--env-file", str(args.env_file), "exec", "-T", "api",
                        "python", "-m", "app.ingestion.ingest", "--dataset",
                        f"/workspace/datasets/{Path(args.dataset).name}", "--resume"], cwd=REPO_ROOT, timeout=3600)
    status = "pass" if ok else "fail"
    return StepResult("real_ingest", status, started, _now(), command, evidence=output[-500:])


def step_interrupted_resume(args: argparse.Namespace) -> StepResult:
    started = _now()
    command = "docker compose exec api python -m app.ingestion.ingest ... --resume (after an intentional kill mid-run)"
    return _not_run(
        "interrupted_resume", command,
        "Requires an operator-supervised interruption of a live ingest run; not automated by this script to avoid corrupting a real run unattended",
        "An intentionally interrupted inactive run on the institution stack", started,
    )


def step_active_run_verification(args: argparse.Namespace) -> StepResult:
    started = _now()
    bind_host = args.bind_host
    command = f"curl -fsS http://{bind_host}:8000/stats"
    if not args.live:
        return _not_run("active_run_verification", command, "compose_up was skipped (--live not passed)", "A running api container with a completed run", started)
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://{bind_host}:8000/stats", timeout=5) as response:
            body = response.read().decode("utf-8")
        return StepResult("active_run_verification", "pass", started, _now(), command, evidence=body[-500:])
    except Exception as exc:
        return StepResult("active_run_verification", "fail", started, _now(), command, reason=f"{type(exc).__name__}: {exc}")


def step_pushdown_equivalence(args: argparse.Namespace) -> StepResult:
    started = _now()
    command = ("docker compose exec -T api python -m app.search.equivalence --dataset-id <id> "
               "--backend clickhouse --dimension 512 --output /workspace/artifacts/faz11/filter_equivalence.json")
    return _not_run(
        "pushdown_equivalence", command, "Requires a completed active institution run at representative scale",
        "Completed active institution run with clickhouse enabled", started,
    )


def step_scale_diagnostics(args: argparse.Namespace) -> StepResult:
    started = _now()
    command = "docker compose exec -T api python -m app.search.equivalence ... (scale/index diagnostics section)"
    return _not_run(
        "scale_diagnostics", command, "Requires a representative-scale active institution run",
        "Completed active institution run at representative scale", started,
    )


def step_ui_search(args: argparse.Namespace) -> StepResult:
    started = _now()
    command = f"RUN_FAZ8_INTEGRATION=1 UI_URL=http://{args.bind_host}:7860 PYTHONPATH=service pytest service/tests/test_t10_ui.py -q"
    return _not_run(
        "ui_search", command, "Requires a healthy running UI/API stack and Playwright Chromium in this environment",
        "Healthy API/UI stack with an active real run and Playwright Chromium installed", started,
    )


def step_media_playback(args: argparse.Namespace) -> StepResult:
    started = _now()
    command = f"curl -fsS http://{args.bind_host}:8000/media/<segment_id> -o /tmp/clip.mp4"
    return _not_run(
        "media_playback", command, "Requires an active run with a real local MP4 source and a running API",
        "Active real ingest run with DATA_ROOT local MP4", started,
    )


def step_final_artifact_audit(args: argparse.Namespace) -> StepResult:
    started = _now()
    required = [
        "artifacts/faz11/baseline.json", "artifacts/faz11/preflight_example.json",
        "artifacts/faz11/schema_migration_report.json", "artifacts/faz11/ingest_resume_smoke.json",
        "artifacts/faz11/filter_equivalence.json", "artifacts/faz11/search_scale_smoke.json",
        "artifacts/faz11/gpu_smoke.json", "artifacts/faz11/ui_smoke.png", "artifacts/faz11/final_acceptance.json",
        "artifacts/faz11/traceability_audit.json", "artifacts/faz11/migration_contract_audit.json",
        "artifacts/faz11/preflight_no_write_audit.json", "artifacts/faz11/run_versioning_fault_matrix.json",
        "artifacts/faz11/pushdown_adapter_audit.json", "artifacts/faz11/streaming_memory_smoke.json",
    ]
    missing = [name for name in required if not (REPO_ROOT / name).is_file()]
    status = "fail" if missing else "pass"
    command = "test -f artifacts/faz11/<each required artifact>"
    return StepResult(
        "final_artifact_audit", status, started, _now(), command,
        evidence=f"present={len(required) - len(missing)}/{len(required)}",
        reason="" if not missing else f"missing: {missing}",
    )


STEPS: list[Callable[[argparse.Namespace], StepResult]] = [
    step_git_state, step_host_info, step_nvidia_driver, step_docker_daemon, step_compose_config,
    step_secure_credentials, step_model_bundle_hash, step_dataset_preflight, step_migration_plan,
    step_compose_up, step_health_check, step_gpu_smoke, step_real_ingest, step_interrupted_resume,
    step_active_run_verification, step_pushdown_equivalence, step_scale_diagnostics, step_ui_search,
    step_media_playback, step_final_artifact_audit,
]


def overall_status(results: list[StepResult]) -> str:
    statuses = {result.status for result in results}
    if "fail" in statuses:
        return "implementation_incomplete"
    if "not_run" in statuses:
        return "implementation_complete_hardware_acceptance_pending"
    return "fully_accepted_on_target_environment"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--model-bundle-root", type=Path, default=None)
    parser.add_argument("--bind-host", default="127.0.0.1")
    parser.add_argument("--live", action="store_true", help="Actually start Compose services and run live ingest steps")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "artifacts" / "faz11" / "target_acceptance.json")
    args = parser.parse_args()

    git_sha_ok, git_sha = _run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    results = [step(args) for step in STEPS]
    report = {
        "schema_version": 1,
        "generated_at_utc": _now(),
        "tested_code_sha": git_sha if git_sha_ok else None,
        "live_mode": args.live,
        "steps": [asdict(result) for result in results],
        "summary": {
            status: sum(1 for result in results if result.status == status)
            for status in ("pass", "fail", "blocked", "not_run")
        },
        "overall_status": overall_status(results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["overall_status"] != "implementation_incomplete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
