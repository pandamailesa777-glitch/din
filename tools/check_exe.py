"""Bounded packaged-app smoke test; no Python installation is used by the child."""
from pathlib import Path
import subprocess
import json

ROOT = Path(__file__).resolve().parents[1]
exe = ROOT / "dist/VerticalDynamics/VerticalDynamics.exe"
output = ROOT / "output/exe_check"
child = subprocess.Popen([str(exe), "--case", str(ROOT / "examples/rectangular_case.json"),
                          "--output", str(output), "--report"], cwd=ROOT)
try:
    code = child.wait(timeout=45)
except subprocess.TimeoutExpired:
    child.kill()
    child.wait()
    raise RuntimeError("Packaged app timed out, possibly due to a bootloader error dialog.")
assert code == 0, f"Packaged app exited with {code}"
for name in ("time_history.csv", "summary.csv", "run_metadata.json", "plots.png", "report.pdf"):
    assert (output / name).stat().st_size > 100, name
assert (output / "report.pdf").read_bytes().startswith(b"%PDF")
metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
assert metadata["parameters_SI"]["mass"] == 10
print("PASS: packaged executable, calculation, CSV/JSON/PNG/PDF exports")
