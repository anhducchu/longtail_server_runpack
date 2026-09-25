"""A long phase must not write one log record per tqdm redraw."""
import sys
import types
from pathlib import Path

from runner.worker import run_phase


class FakeState:
    def heartbeat(self, task_id, message=""): pass


def phase(command, tmp_path):
    return types.SimpleNamespace(name="train", command=command, cwd=str(tmp_path), env={}, skip_if_exists=[])


def child(script, tmp_path):
    """Run from a file: an inline -c body would echo into the log header itself."""
    path = Path(tmp_path) / "child.py"
    path.write_text(script)
    return [sys.executable, str(path)]


REDRAWS = (
    "import sys, time\n"
    "for i in range(200):\n"
    "    sys.stdout.write('\\r%d%%|bar| %d/200' % (i // 2, i)); sys.stdout.flush()\n"
    "sys.stdout.write('\\n done\\n')\n"
)


def test_progress_redraws_are_thinned(tmp_path):
    log = tmp_path / "stdout.log"
    code = run_phase(phase(child(REDRAWS, tmp_path), tmp_path), {}, log, FakeState(), "t", None, progress_seconds=30.0)
    assert code == 0
    lines = [l for l in log.read_text().splitlines() if l.strip()]
    bars = [l for l in lines if "bar|" in l]
    # One kept redraw plus the final bar tqdm terminates with a newline.
    assert len(bars) <= 2, lines
    assert any("done" in l for l in lines)
    assert any("suppressed" in l for l in lines)


def test_zero_interval_keeps_every_redraw(tmp_path):
    log = tmp_path / "stdout.log"
    run_phase(phase(child(REDRAWS, tmp_path), tmp_path), {}, log, FakeState(), "t", None, progress_seconds=0.0)
    assert len([l for l in log.read_text().splitlines() if "bar|" in l]) == 200


def test_ordinary_lines_are_never_dropped(tmp_path):
    log = tmp_path / "stdout.log"
    script = "import sys\nfor i in range(50): sys.stdout.write('line %d\\n' % i)\n"
    run_phase(phase(child(script, tmp_path), tmp_path), {}, log, FakeState(), "t", None, progress_seconds=30.0)
    assert log.read_text().count("line ") == 50
