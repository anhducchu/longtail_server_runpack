from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_main_script_exposes_the_cli():
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "{preflight,plan,stop,status,retry-failed,run}" in result.stdout
