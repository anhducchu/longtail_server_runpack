import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_python_module_entrypoint_exposes_the_cli():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(ROOT / "src"), env.get("PYTHONPATH", "")))
    )
    result = subprocess.run(
        [sys.executable, "-m", "ltx", "--help"],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "{preflight,plan,stop,status,retry-failed,run}" in result.stdout
