"""Small desktop interface. All Tk/Matplotlib operations stay on the UI thread."""
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from .inputs import load_config, load_force
from .plotting import make_figure
from .results import export_csv, format_value, summary
from .solver import simulate


class Application:
    def __init__(self, root):
        self.root = root
        self.parameters = self.load = self.result = None
        self.canvas = self.toolbar = None
        self.queue = Queue()
        self.busy = False
        root.title("Вертикальная динамика — предварительный расчёт")
        root.geometry("1160x900")
        root.minsize(900, 650)
        self.status = tk.StringVar(value="Загрузите конфигурацию JSON и внешнюю нагрузку CSV.")
        self.config_label = tk.StringVar(value="Конфигурация не выбрана")
        self.force_label = tk.StringVar(value="Нагрузка не выбрана")
        outer = ttk.Frame(root, padding=12)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Анализ вертикального движения", font=("Segoe UI", 17, "bold")).pack(anchor="w")
        ttk.Label(outer, text="Ось вверх · жёсткое тело · остановка при первом касании").pack(anchor="w", pady=(2, 10))
        inputs = ttk.Frame(outer)
        inputs.pack(fill="x")
        self.config_button = ttk.Button(inputs, text="1. Конфигурация JSON…", command=self.choose_config)
        self.config_button.grid(row=0, column=0, sticky="w", pady=3)
        ttk.Label(inputs, textvariable=self.config_label).grid(row=0, column=1, sticky="w", padx=12)
        self.force_button = ttk.Button(inputs, text="2. Нагрузка CSV…", command=self.choose_force)
        self.force_button.grid(row=1, column=0, sticky="w", pady=3)
        ttk.Label(inputs, textvariable=self.force_label).grid(row=1, column=1, sticky="w", padx=12)
        actions = ttk.Frame(outer)
        actions.pack(fill="x", pady=10)
        self.run_button = ttk.Button(actions, text="3. Выполнить расчёт", command=self.run, state="disabled")
        self.run_button.pack(side="left")
        self.export_button = ttk.Button(actions, text="Экспорт CSV…", command=self.export, state="disabled")
        self.export_button.pack(side="left", padx=8)
        self.png_button = ttk.Button(actions, text="Сохранить PNG…", command=self.save_png, state="disabled")
        self.png_button.pack(side="left")
        self.progress = ttk.Progressbar(actions, mode="indeterminate", length=150)
        self.progress.pack(side="right")
        self.tabs = ttk.Notebook(outer)
        self.tabs.pack(fill="both", expand=True)
        self.plot_frame = ttk.Frame(self.tabs)
        report_frame = ttk.Frame(self.tabs)
        self.tabs.add(self.plot_frame, text="Графики")
        self.tabs.add(report_frame, text="Итоговые результаты")
        self.placeholder = ttk.Label(self.plot_frame, text="Графики появятся после расчёта.", anchor="center")
        self.placeholder.pack(fill="both", expand=True)
        self.table = ttk.Treeview(report_frame, columns=("parameter", "value", "unit", "note"), show="headings")
        for name, label, width in (("parameter", "Показатель", 405), ("value", "Результат", 130),
                                   ("unit", "Единицы", 65), ("note", "Примечание", 300)):
            self.table.heading(name, text=label)
            self.table.column(name, width=width, minwidth=50)
        scroll = ttk.Scrollbar(report_frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.table.pack(fill="both", expand=True)
        ttk.Label(outer, textvariable=self.status, wraplength=1080).pack(anchor="w", pady=(8, 4))
        ttk.Label(outer, text="Предварительная модель: вращение, деформации и удар не учитываются. "
                  "Не подтверждает безопасность реального оборудования.", wraplength=1080,
                  foreground="#765326").pack(anchor="w")

    def invalidate(self):
        self.result = None
        self.table.delete(*self.table.get_children())
        if self.toolbar:
            self.toolbar.destroy()
            self.toolbar = None
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        self.placeholder.pack(fill="both", expand=True)
        self.export_button.configure(state="disabled")
        self.png_button.configure(state="disabled")
        self.run_button.configure(state="normal" if self.parameters is not None and self.load is not None else "disabled")

    def set_config(self, path):
        parameters = load_config(path)
        self.parameters = parameters
        self.config_label.set(str(Path(path).resolve()))
        self.invalidate()
        self.status.set(f"Конфигурация загружена: m = {parameters.mass:g} кг, g = {parameters.gravity:g} м/с².")

    def set_force(self, path):
        load = load_force(path)
        self.load = load
        self.force_label.set(str(Path(path).resolve()))
        self.invalidate()
        self.status.set(f"Нагрузка загружена: {len(load.time)} точек. Знак плюс — вверх.")

    def choose_config(self):
        path = filedialog.askopenfilename(title="Выберите конфигурацию", filetypes=[("Конфигурация JSON", "*.json")])
        if path:
            try:
                self.set_config(path)
            except ValueError as exc:
                messagebox.showerror("Ошибка исходных данных", str(exc))

    def choose_force(self):
        path = filedialog.askopenfilename(title="Выберите внешнюю нагрузку", filetypes=[("Нагрузка CSV", "*.csv")])
        if path:
            try:
                self.set_force(path)
            except ValueError as exc:
                messagebox.showerror("Ошибка нагрузки", str(exc))

    def run(self):
        if self.busy or self.parameters is None or self.load is None:
            return
        self.invalidate()
        self.busy = True
        for button in (self.run_button, self.config_button, self.force_button):
            button.configure(state="disabled")
        self.progress.start()
        self.status.set("Выполняется расчёт…")
        parameters, load = self.parameters, self.load

        def worker():
            try:
                self.queue.put((simulate(parameters, load), None))
            except Exception as exc:
                self.queue.put((None, str(exc)))

        Thread(target=worker, daemon=True).start()
        self.root.after(50, self.poll)

    def poll(self):
        try:
            result, error = self.queue.get_nowait()
        except Empty:
            self.root.after(50, self.poll)
            return
        self.busy = False
        self.progress.stop()
        for button in (self.run_button, self.config_button, self.force_button):
            button.configure(state="normal")
        if error:
            self.status.set("Расчёт не выполнен. Исправьте исходные данные.")
            messagebox.showerror("Ошибка расчёта", error)
            return
        try:
            self.show_result(result)
        except Exception as exc:
            self.invalidate()
            self.status.set("Не удалось отобразить результаты.")
            messagebox.showerror("Ошибка отображения", str(exc))

    def show_result(self, result):
        self.placeholder.pack_forget()
        figure = make_figure(result)
        self.canvas = FigureCanvasTkAgg(figure, master=self.plot_frame)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame, pack_toolbar=False)
        self.toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.canvas.draw()
        self.table.delete(*self.table.get_children())
        for row in summary(result).itertuples(index=False):
            self.table.insert("", "end", values=(row.parameter, format_value(row.value), row.unit, row.note))
        self.result = result
        self.export_button.configure(state="normal")
        self.png_button.configure(state="normal")
        if result.touchdown is not None:
            self.status.set(f"Расчёт завершён при первом касании: t = {result.touchdown:.9g} с. Скорость дана до удара.")
        elif result.detached:
            self.status.set("Расчёт завершён: касание до tmax не наступило. При необходимости увеличьте tmax.")
        else:
            self.status.set("Расчёт завершён: конструкция оставалась на основании.")

    def export(self):
        if self.result is None:
            return
        directory = filedialog.askdirectory(title="Выберите папку для экспорта")
        if directory:
            try:
                output = Path(directory) / datetime.now().strftime("calculation_%Y%m%d_%H%M%S_%f")
                export_csv(self.result, output)
                self.status.set(f"Сохранены time_history.csv, summary.csv и run_metadata.json: {output}")
            except OSError as exc:
                messagebox.showerror("Ошибка сохранения", str(exc))

    def save_png(self):
        if self.result is None:
            return
        path = filedialog.asksaveasfilename(title="Сохранить графики", defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            try:
                # Export the full-size layout, independently of the current window size.
                figure = make_figure(self.result)
                figure.savefig(path, dpi=160)
                self.status.set(f"Графики сохранены: {path}")
            except OSError as exc:
                messagebox.showerror("Ошибка сохранения", str(exc))


def launch(config=None, force=None):
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise RuntimeError("Не удалось запустить Tkinter. Нужен Python с компонентом Tcl/Tk "
                           "и доступ к графическому рабочему столу. " + str(exc)) from exc
    app = Application(root)
    try:
        if config:
            app.set_config(config)
        if force:
            app.set_force(force)
    except ValueError as exc:
        messagebox.showerror("Ошибка исходных данных", str(exc))
    root.mainloop()
