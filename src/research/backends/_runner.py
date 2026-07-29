"""Uc backend sarmalayicisinin (ch.py/qd.py/pv.py) paylastigi tek fonksiyon:
install/start/health/stop/cleanup shell script'lerini cagirir. SESSIZ
FALLBACK YOK (spec madde 9) - script basarisiz donerse `ok=False` ve
gercek stderr/returncode DONER, cagiran taraf bunu 'environment_unavailable'
olarak isaretler, sahte basari URETMEZ."""
import pathlib
import subprocess

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parents[3] / "scripts"


def run_action(script_name: str, action: str, env: dict = None, timeout: int = 600) -> dict:
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        return {"ok": False, "action": action, "error": f"script bulunamadi: {script_path}"}
    try:
        result = subprocess.run(
            ["bash", str(script_path), action],
            capture_output=True, text=True, timeout=timeout, env=env,
        )
        return {
            "ok": result.returncode == 0,
            "action": action,
            "returncode": result.returncode,
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
        }
    except FileNotFoundError:
        return {"ok": False, "action": action, "error": "bash bulunamadi (Windows/Colab-disi ortam)"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "action": action, "error": f"zaman asimi ({timeout}s)"}


__all__ = ["run_action", "SCRIPTS_DIR"]
