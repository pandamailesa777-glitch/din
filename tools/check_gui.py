"""Desktop integration smoke: wizard, validation, calculations, JSON and all exports."""
from pathlib import Path
import sys
import tempfile
import time
import tkinter as tk
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vertical_dynamics.gui import Application
from vertical_dynamics.case import load_case


def wait(root, app):
    deadline = time.monotonic() + 30
    while app.busy and time.monotonic() < deadline:
        root.update()
        time.sleep(.02)
    assert not app.busy, "Background calculation timed out"
    assert app.result is not None


def main():
    root = tk.Tk()
    root.withdraw()
    try:
        app = Application(root)
        with patch("vertical_dynamics.gui.messagebox.showerror") as error:
            app.run()
            assert error.called and "Масса" in error.call_args.args[1]
            assert "Постоянная" in error.call_args.args[1]
            assert not app.busy
        app.vars["mass"].set("10")
        app.vars["force"].set("196,2")
        app.vars["duration"].set("0,5")
        app.vars["force_start"].set("0,2")
        app.next_page(1)
        assert app.tabs.index("current") == 1
        app.preview()
        assert app.preview_canvas is not None
        app.next_page(2)
        assert app.tabs.index("current") == 2
        app.run()
        wait(root, app)
        assert abs(app.result.touchdown - 1.90710678118655) < 1e-8
        assert len(app.table.get_children()) == 18
        assert len(app.canvas.figure.axes) == 6
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with patch("vertical_dynamics.gui.filedialog.askdirectory", return_value=directory):
                app.export()
            assert len(list(folder.glob("*/summary.csv"))) == 1
            for extension, action in (("png", app.save_png), ("pdf", app.save_report), ("json", app.save_inputs)):
                path = folder / ("gui_export." + extension)
                with patch("vertical_dynamics.gui.filedialog.asksaveasfilename", return_value=str(path)):
                    action()
                assert path.stat().st_size > 100
            p, load, title = load_case(folder / "gui_export.json")
            assert p.mass == 10 and load.force[0] == 196.2
            app.vars["mass"].set("20")
            assert app.result is None and app.canvas is None
            with patch("vertical_dynamics.gui.filedialog.askopenfilename", return_value=str(folder / "gui_export.json")):
                app.open_case()
            assert app.read_inputs()[0].mass == 10
        app.set_config(ROOT / "examples/config.json")
        app.set_force(ROOT / "examples/force.csv")
        app.method.set("Эйлер (демонстрационный)")
        app.run()
        wait(root, app)
        assert app.result.parameters.method == "Euler"
        assert abs(app.result.touchdown - 1.90710678118655) < .003
        # Edits while the background solver is running must invalidate its result.
        app.run()
        app.vars["mass"].set("30")
        deadline = time.monotonic()+30
        while app.busy and time.monotonic() < deadline:
            root.update()
            time.sleep(.02)
        assert not app.busy and app.result is None
        print("PASS: wizard, required fields, decimal comma, preview, RK45/Euler, 6 plots, 18 rows, JSON/CSV/PNG/PDF, stale result rejection")
    finally:
        root.destroy()


if __name__ == "__main__":
    main()
