"""Faz 6 Colab handoff - notebook 04'un ilk hucresi bunu cagirir. Ortamin
ClickHouse/Qdrant/pgvector'i GERCEKTEN kurup calistirabilecek durumda olup
olmadigini kontrol eder ve artifacts/research/environment_capability_report.json'a
yazar. Bu bir SESSION/VM tanisi - Drive'a degil, yerel calisan repo'nun
artifacts/research/ dizinine yazilir (buyuk/kalici veri degil).

Sessiz varsayim YAPMAZ: her kontrol GERCEKTEN calistirilir (subprocess/
socket/HTTP HEAD), sonuc bilinmiyorsa 'unknown' yazilir, asla 'muhtemelen
calisir' gibi bir tahmin URETMEZ."""
import json
import pathlib
import platform
import shutil
import socket
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

REPORT_PATH = pathlib.Path("artifacts/research/environment_capability_report.json")
BACKEND_VERSIONS_PATH = pathlib.Path("backend_versions.json")


def _run(cmd: list, timeout: int = 10) -> dict:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {"ok": r.returncode == 0, "returncode": r.returncode,
               "stdout": r.stdout.strip()[:500], "stderr": r.stderr.strip()[:500]}
    except FileNotFoundError:
        return {"ok": False, "error": "komut bulunamadi"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "zaman asimi"}
    except Exception as e:  # noqa: BLE001 - tanisal kontrol, cokmemeli
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def check_os_arch() -> dict:
    return {"system": platform.system(), "release": platform.release(),
           "machine": platform.machine(), "python_version": platform.python_version()}


def check_user() -> dict:
    try:
        import os
        uid = os.getuid() if hasattr(os, "getuid") else None
        return {"uid": uid, "is_root": uid == 0 if uid is not None else "unknown (Windows)"}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def check_binary(name: str) -> dict:
    path = shutil.which(name)
    return {"found": path is not None, "path": path}


def check_apt_get() -> dict:
    if shutil.which("apt-get") is None:
        return {"available": False, "reason": "apt-get PATH'te yok (Linux/Colab disinda beklenir)"}
    return {"available": True, **_run(["apt-get", "--version"])}


def check_docker_daemon() -> dict:
    if shutil.which("docker") is None:
        return {"cli_found": False, "daemon_reachable": False}
    info = _run(["docker", "info"], timeout=15)
    return {"cli_found": True, "daemon_reachable": info["ok"], "detail": info}


def check_port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        result = s.connect_ex(("127.0.0.1", port))
        return result != 0  # != 0 -> baglanti REDDEDILDI -> port bos


def check_ports(ports: list) -> dict:
    return {str(p): check_port_free(p) for p in ports}


def check_ram_mb() -> dict:
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {"total_mb": round(vm.total / 1e6), "available_mb": round(vm.available / 1e6)}
    except ImportError:
        return {"error": "psutil yuklu degil - pip install psutil"}


def check_disk_space(path: str = ".") -> dict:
    usage = shutil.disk_usage(path)
    return {"path": path, "total_gb": round(usage.total / 1e9, 1),
           "free_gb": round(usage.free / 1e9, 1)}


def check_drive_access() -> dict:
    drive_mount_root = pathlib.Path("/content/drive")
    return {"mount_point_exists": drive_mount_root.exists(),
           "colab_detected": _is_colab()}


def _is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def check_url_reachable(url: str, timeout: int = 10) -> dict:
    try:
        import requests
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        return {"reachable": r.status_code < 400, "status_code": r.status_code, "final_url": r.url}
    except Exception as e:  # noqa: BLE001
        return {"reachable": False, "error": f"{type(e).__name__}: {e}"}


def check_postgres_installable() -> dict:
    """apt-cache ile postgresql paketinin apt deposunda gorunur olup
    olmadigini kontrol eder - GERCEKTEN kurmaz (o notebook 04'un isi)."""
    if shutil.which("apt-cache") is None:
        return {"checked": False, "reason": "apt-cache yok (Linux/Colab disinda beklenir)"}
    result = _run(["apt-cache", "policy", "postgresql-16"], timeout=15)
    return {"checked": True, "package_visible": bool(result.get("stdout")), "detail": result}


def build_report() -> dict:
    backend_versions = json.loads(BACKEND_VERSIONS_PATH.read_text(encoding="utf-8")) \
        if BACKEND_VERSIONS_PATH.exists() else {}
    ch_url = backend_versions.get("clickhouse", {}).get("download_url_template", "").format(
        version=backend_versions.get("clickhouse", {}).get("version", ""))
    qd_url = backend_versions.get("qdrant", {}).get("download_url_template", "").format(
        version=backend_versions.get("qdrant", {}).get("version", ""))

    report = {
        "os_arch": check_os_arch(),
        "user": check_user(),
        "apt_get": check_apt_get(),
        "docker": check_docker_daemon(),
        "binaries": {name: check_binary(name) for name in ("curl", "wget", "tar", "git", "make", "gcc")},
        "ports_free": check_ports([8123, 9000, 6333, 6334, 5432]),
        "ram": check_ram_mb(),
        "disk_content": check_disk_space("."),
        "drive": check_drive_access(),
        "github_releases_reachable": check_url_reachable("https://github.com"),
        "clickhouse_binary_url_reachable": check_url_reachable(ch_url) if ch_url else {"skipped": "backend_versions.json yok"},
        "qdrant_binary_url_reachable": check_url_reachable(qd_url) if qd_url else {"skipped": "backend_versions.json yok"},
        "postgres_pgvector_installable": check_postgres_installable(),
    }
    return report


def main():
    report = build_report()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n-> {REPORT_PATH}")
    return report


if __name__ == "__main__":
    main()
