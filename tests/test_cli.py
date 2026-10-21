from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_cli_example(tmp_path):
    run = subprocess.run([sys.executable, "-m", "vertical_dynamics", "--config", "examples/config.json",
                          "--force", "examples/force.csv", "--output", str(tmp_path)],
                         cwd=ROOT, capture_output=True)
    assert run.returncode == 0, run.stderr.decode(errors="replace")
    assert (tmp_path / "plots.png").exists()
    assert (tmp_path / "summary.csv").exists()


def test_cli_missing_file(tmp_path):
    run = subprocess.run([sys.executable, "-m", "vertical_dynamics", "--config", "missing.json",
                          "--force", "examples/force.csv", "--output", str(tmp_path)],
                         cwd=ROOT, capture_output=True)
    assert run.returncode == 1
    assert not (tmp_path / "plots.png").exists()
