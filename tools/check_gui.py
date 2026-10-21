"""Run explicitly in a desktop session: python tools/check_gui.py."""
from pathlib import Path
import sys
import tempfile
import time
import tkinter as tk
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vertical_dynamics.gui import Application


def main():
    root = tk.Tk()
    root.withdraw()
    try:
        app = Application(root)
        app.set_config(ROOT / "examples/config.json")
        app.set_force(ROOT / "examples/force.csv")
        app.run()
        deadline = time.monotonic() + 20
        while app.busy and time.monotonic() < deadline:
            root.update()
            time.sleep(0.02)
        assert not app.busy, "Background calculation timed out"
        assert app.result is not None
        assert len(app.table.get_children()) == 13
        assert len(app.canvas.figure.axes) == 5
        with tempfile.TemporaryDirectory() as directory:
            with patch("vertical_dynamics.gui.filedialog.askdirectory", return_value=directory):
                app.export()
            assert len(list(Path(directory).glob("*/summary.csv"))) == 1
            png = Path(directory) / "gui_export.png"
            with patch("vertical_dynamics.gui.filedialog.asksaveasfilename", return_value=str(png)):
                app.save_png()
            assert png.read_bytes().startswith(b"\x89PNG")
        app.set_force(ROOT / "examples/force.csv")
        assert app.result is None and app.canvas is None
        assert str(app.export_button["state"]) == "disabled"
        print("PASS: GUI loading, background solver, 5 plots, 13 rows, CSV/PNG exports, result invalidation")
    finally:
        root.destroy()


if __name__ == "__main__":
    main()
