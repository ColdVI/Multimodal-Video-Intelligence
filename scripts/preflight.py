from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE_ROOT = REPO_ROOT / "service"
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.config import Settings  # noqa: E402
from app.preflight import exit_code as data_exit_code  # noqa: E402
from app.preflight import run_data_preflight  # noqa: E402


def _read_env(path: Path) -> dict[str, str]:
    values = dict(os.environ)
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values.setdefault(key.strip(), value.strip())
    return values


def _command(command: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    detail = (result.stdout or result.stderr).strip().splitlines()
    return result.returncode == 0, (detail[0] if detail else f"exit={result.returncode}")


def _host_check(check_id: str, category: str, ok: bool, detail: str) -> dict[str, str]:
    return {"check_id": check_id, "category": category, "status": "pass" if ok else "fail", "detail": detail}


def run_host_preflight(dataset: Path, env_file: Path) -> dict[str, Any]:
    env = _read_env(env_file)
    checks: list[dict[str, str]] = []
    docker_ok, docker_detail = _command(["docker", "--version"])
    checks.append(_host_check("docker", "configuration", docker_ok, docker_detail))
    compose_ok, compose_detail = _command(["docker", "compose", "version"])
    checks.append(_host_check("compose_v2", "configuration", compose_ok, compose_detail))
    gpu_ok, gpu_detail = _command(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    checks.append(_host_check("nvidia_driver", "gpu", gpu_ok, gpu_detail))
    toolkit_ok, toolkit_detail = _command(["docker", "info", "--format", "{{json .Runtimes}}"])
    toolkit_ok = toolkit_ok and "nvidia" in toolkit_detail.lower()
    checks.append(_host_check("nvidia_container_toolkit", "gpu", toolkit_ok, toolkit_detail))

    required_env = ("POSTGRES_PASSWORD", "CLICKHOUSE_PASSWORD", "DATA_ROOT", "MODEL_BUNDLE_ROOT", "CUDA_IMAGE_TAG")
    missing = [name for name in required_env if not env.get(name) or env[name].startswith("CHANGE_ME")]
    checks.append(_host_check("required_env", "configuration", not missing, f"missing_or_placeholder={missing}"))
    bind_host = env.get("BIND_HOST", "127.0.0.1")
    checks.append(_host_check("bind_host", "configuration", bool(bind_host), f"BIND_HOST={bind_host}"))

    data_root = Path(env.get("DATA_ROOT", "data")).expanduser()
    if not data_root.is_absolute():
        data_root = (REPO_ROOT / data_root).resolve()
    model_root = Path(env.get("MODEL_BUNDLE_ROOT", "")).expanduser()
    checks.append(_host_check("data_root", "data", data_root.is_dir(), f"path={data_root}"))
    checks.append(_host_check("model_bundle_root", "model", model_root.is_dir(), f"path={model_root}"))
    try:
        usage = shutil.disk_usage(data_root if data_root.exists() else REPO_ROOT)
        checks.append(_host_check("disk_space", "resources", usage.free > 0, f"free_bytes={usage.free}"))
    except OSError as exc:
        checks.append(_host_check("disk_space", "resources", False, f"{type(exc).__name__}: {exc}"))

    compose_command = ["docker", "compose", "--env-file", str(env_file), "config"]
    compose_config_ok, compose_config_detail = _command(compose_command)
    checks.append(_host_check("compose_config", "configuration", compose_config_ok, compose_config_detail))
    cuda_tag = env.get("CUDA_IMAGE_TAG", "12.1.1-runtime-ubuntu22.04")
    cuda_image = f"nvidia/cuda:{cuda_tag}"
    container_gpu_ok, container_gpu_detail = _command([
        "docker", "run", "--rm", "--gpus", "all", cuda_image,
        "nvidia-smi", "--query-gpu=name", "--format=csv,noheader",
    ])
    checks.append(_host_check(
        "cuda_container_gpu", "gpu", container_gpu_ok,
        f"image={cuda_image}; {container_gpu_detail}",
    ))

    try:
        configured = Settings.from_env(env)
        data_report = run_data_preflight(dataset, data_root=data_root, configured=configured)
    except Exception as exc:
        data_report = {
            "status": "fail", "checks": [{
                "check_id": "settings", "category": "configuration", "status": "fail",
                "detail": f"{type(exc).__name__}: {exc}",
            }],
        }
    checks.extend(data_report.get("checks", []))
    status = "fail" if any(item["status"] == "fail" for item in checks) else (
        "not_run" if any(item["status"] == "not_run" for item in checks) else "pass"
    )
    git_ok, git_sha = _command(["git", "rev-parse", "HEAD"])
    return {
        "schema_version": 1,
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha if git_ok else None,
        "environment": {
            "platform": sys.platform,
            "python": sys.version.split()[0],
            "env_file": str(env_file),
            "dataset_manifest": str(dataset),
        },
        "command": [
            "python", "scripts/preflight.py", "--dataset", str(dataset),
            "--env-file", str(env_file),
        ],
        "checks": checks,
        "data_preflight": data_report,
    }


def _combined_exit_code(report: dict[str, Any]) -> int:
    failed = {item["category"] for item in report["checks"] if item["status"] == "fail"}
    for category, code in (
        ("configuration", 2), ("data", 3), ("gpu", 4), ("model", 5), ("resources", 6),
    ):
        if category in failed:
            return code
    return data_exit_code(report.get("data_preflight", {"checks": []}))


def main() -> int:
    parser = argparse.ArgumentParser(description="Faz 11 host and dataset preflight (read-only)")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--not-run-reason", help="mark an acceptance artifact not_run while retaining observed checks")
    parser.add_argument("--required-command")
    parser.add_argument("--expected-environment")
    args = parser.parse_args()
    report = run_host_preflight(args.dataset.resolve(), args.env_file.resolve())
    observed_report = report
    if args.not_run_reason:
        report = {
            **report,
            "status": "not_run",
            "reason": args.not_run_reason,
            "required_command": args.required_command,
            "expected_environment": args.expected_environment,
            "observed_preflight_status": observed_report["status"],
        }
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return _combined_exit_code(observed_report)


if __name__ == "__main__":
    raise SystemExit(main())
