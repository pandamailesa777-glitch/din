"""Russian desktop wizard; solver work runs outside the Tk event thread."""
from dataclasses import fields
from decimal import Decimal
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from .case import load_case, save_case
from .inputs import InputError, Load, Parameters, load_config, load_force
from .plotting import force_preview, make_figure
from .results import export_csv, format_value, summary
from .solver import simulate


class RussianToolbar(NavigationToolbar2Tk):
    toolitems = (
        ("Исходный вид", "Вернуть исходный масштаб", "home", "home"),
        ("Назад", "Предыдущий масштаб", "back", "back"),
        ("Вперёд", "Следующий масштаб", "forward", "forward"),
        (None, None, None, None),
        ("Перемещение", "Переместить график мышью", "move", "pan"),
        ("Масштаб", "Выделить прямоугольник для увеличения", "zoom_to_rect", "zoom"),
    )


LABELS = {
    "mass": "Масса конструкции, кг", "initial_velocity": "Начальная скорость, м/с",
    "initial_height": "Начальное перемещение, м", "gravity": "Ускорение тяжести, м/с²",
    "force_start": "Начало воздействия, с", "duration": "Продолжительность воздействия, с",
    "end_time": "Общее время моделирования, с", "euler_step": "Шаг Эйлера, с",
    "max_step": "Максимальный шаг RK45, с", "output_step": "Шаг вывода, с",
    "rtol": "Относительная точность RK45", "atol_z": "Абсолютная точность перемещения, м",
    "atol_v": "Абсолютная точность скорости, м/с", "force": "Постоянная вертикальная сила",
}
DEFAULTS = {"mass": "", "initial_velocity": "0", "initial_height": "0", "gravity": "9.81",
            "force_start": "0", "duration": "1", "end_time": "5", "euler_step": "0.001",
            "max_step": "0.05", "output_step": "0.01", "rtol": "1e-8", "atol_z": "1e-10", "atol_v": "1e-10", "force": ""}
METHODS = {"RK45 (основной)": "RK45", "Эйлер (демонстрационный)": "Euler"}


def parse_number(text, label):
    try:
        value = float(text.strip().replace(",", "."))
        if not np.isfinite(value):
            raise ValueError()
        return value
    except ValueError as exc:
        raise InputError(f"{label}: введите конечное число.") from exc


class Application:
    def __init__(self, root):
        self.root = root
        self.parameters = self.load = self.result = None
        self.canvas = self.toolbar = self.preview_canvas = None
        self.result_window = None
        self.queue = Queue()
        self.busy = self.loading = False
        self.revision = 0
        self.preview_job = None
        self.result_title = ""
        root.title("Вертикальная динамика — расчёт отрыва и перемещения")
        root.geometry("1080x880")
        root.minsize(940, 760)
        self.vars = {key: tk.StringVar(value=value) for key, value in DEFAULTS.items()}
        self.title = tk.StringVar(value="Расчёт вертикального движения")
        self.mode = tk.StringVar(value="Постоянная сила")
        self.method = tk.StringVar(value="RK45 (основной)")
        self.force_unit = tk.StringVar(value="Н")
        self.status = tk.StringVar(value="Шаг 1: задайте массу конструкции. Остальные поля можно изменять.")
        outer = ttk.Frame(root, padding=16)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Вертикальная динамика", font=("Segoe UI", 20, "bold")).pack(anchor="w")
        ttk.Label(outer, text="Одна степень свободы • односторонний контакт • до первого касания").pack(anchor="w", pady=(2, 10))
        top = ttk.Frame(outer)
        top.pack(fill="x", pady=4)
        ttk.Label(top, text="Название расчёта:").pack(side="left")
        ttk.Entry(top, textvariable=self.title, width=45).pack(side="left", padx=8)
        ttk.Button(top, text="Открыть JSON…", command=self.open_case).pack(side="left", padx=4)
        ttk.Button(top, text="Сохранить JSON…", command=self.save_inputs).pack(side="left")
        self.tabs = ttk.Notebook(outer)
        self.tabs.pack(fill="both", expand=True, pady=10)
        pages = [ttk.Frame(self.tabs, padding=18) for _ in range(3)]
        for page, title in zip(pages, ("1. Массовые характеристики", "2. Внешнее воздействие", "3. Настройки и расчёт")):
            self.tabs.add(page, text=title)
        mass_page, load_page, settings_page = pages
        for row, key in enumerate(("mass", "initial_velocity", "initial_height", "gravity")):
            self.field(mass_page, key, row)
        ttk.Label(mass_page, text="Масса обязательна. Учитывайте только рассматриваемую конструкцию.\n"
                  "Массы отделяющихся и перемещающихся компонентов автоматически не включаются.\n"
                  "Начальное перемещение задаёт зазор над основанием; положительная скорость направлена вверх.",
                  wraplength=850).grid(row=5, column=0, columnspan=3, sticky="w", pady=25)
        ttk.Button(mass_page, text="Сложить массы компонентов…", command=self.sum_masses).grid(row=6, column=0, sticky="w")
        ttk.Button(mass_page, text="Далее: внешнее воздействие →", command=lambda: self.next_page(1)).grid(row=7, column=0, sticky="w", pady=25)
        self.config_button = ttk.Button(mass_page, text="Импорт прежней конфигурации JSON…", command=self.choose_config)
        self.config_button.grid(row=8, column=0, sticky="w")

        ttk.Label(load_page, text="Способ задания нагрузки:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(load_page, textvariable=self.mode, state="readonly", values=("Постоянная сила", "Таблица / CSV"), width=30).grid(row=0, column=1, sticky="w")
        self.field(load_page, "force_start", 1)
        self.field(load_page, "duration", 2)
        self.field(load_page, "force", 3)
        ttk.Combobox(load_page, textvariable=self.force_unit, values=("Н", "кН", "кгс"), state="readonly", width=7).grid(row=3, column=2, sticky="w")
        ttk.Label(load_page, text="Таблица: время [с]; сила [выбранная единица]. По одной паре в строке.").grid(row=4, column=0, columnspan=3, sticky="w", pady=(10, 3))
        self.load_text = tk.Text(load_page, height=5, width=50, undo=True, font=("Consolas", 10))
        self.load_text.grid(row=5, column=0, columnspan=2, sticky="ew")
        self.load_text.bind("<<Modified>>", self.text_changed)
        self.force_button = ttk.Button(load_page, text="Загрузить CSV…", command=self.choose_force)
        self.force_button.grid(row=5, column=2, padx=8)
        ttk.Label(load_page, text="Вводите силу, передаваемую конструкции. Знак «+» — вверх.\n"
                  "Таблица интерполируется линейно; вне интервала воздействия сила равна нулю.",
                  wraplength=850).grid(row=6, column=0, columnspan=3, sticky="w", pady=7)
        self.preview_frame = ttk.Frame(load_page)
        self.preview_frame.grid(row=7, column=0, columnspan=3, sticky="nsew")
        load_page.rowconfigure(7, weight=1)
        ttk.Button(load_page, text="Обновить график нагрузки", command=self.preview).grid(row=8, column=0, sticky="w", pady=5)
        ttk.Button(load_page, text="Далее: настройки →", command=lambda: self.next_page(2)).grid(row=8, column=1, sticky="w")
        ttk.Button(load_page, text="← Назад", command=lambda: self.tabs.select(0)).grid(row=8, column=2)
        for row, key in enumerate(("duration", "end_time", "euler_step", "max_step", "output_step", "rtol", "atol_z", "atol_v")):
            self.field(settings_page, key, row)
        ttk.Label(settings_page, text="Метод интегрирования:").grid(row=8, column=0, sticky="w", pady=8)
        ttk.Combobox(settings_page, textvariable=self.method, values=tuple(METHODS), state="readonly", width=30).grid(row=8, column=1, sticky="w")
        ttk.Label(settings_page, text="Расчёт продолжается после окончания нагрузки до первого касания или заданного времени.\n"
                  "Эйлер: демонстрационный метод первого порядка. Проверяйте результат уменьшением шага.",
                  wraplength=850).grid(row=9, column=0, columnspan=3, sticky="w", pady=20)
        ttk.Button(settings_page, text="← Назад к нагрузке", command=lambda: self.tabs.select(1)).grid(row=10, column=0, sticky="w")
        self.run_button = ttk.Button(settings_page, text="Выполнить расчёт", command=self.run)
        self.run_button.grid(row=10, column=1, sticky="w")
        footer = ttk.Frame(outer)
        footer.pack(side="bottom", fill="x")
        self.progress = ttk.Progressbar(footer, mode="indeterminate")
        self.progress.pack(fill="x")
        ttk.Label(footer, textvariable=self.status, wraplength=1000).pack(anchor="w", pady=8)
        ttk.Label(footer, text="Модель не определяет частичный подъём опор, вращение и ударные напряжения.\n"
                  "Отсутствие полного отрыва не подтверждает безопасность конструкции.", foreground="#795325", wraplength=1000).pack(anchor="w")
        # Reserve the footer before allocating the resizable notebook area.
        self.tabs.pack_forget()
        self.tabs.pack(fill="both", expand=True, pady=10)
        for var in [*self.vars.values(), self.mode, self.method, self.force_unit, self.title]:
            var.trace_add("write", self.changed)

    def field(self, parent, key, row):
        ttk.Label(parent, text=LABELS[key]).grid(row=row, column=0, sticky="w", pady=5, padx=(0, 20))
        ttk.Entry(parent, textvariable=self.vars[key], width=28).grid(row=row, column=1, sticky="w", pady=5)

    def changed(self, *_):
        if self.loading:
            return
        self.revision += 1
        self.invalidate()
        if self.preview_job:
            self.root.after_cancel(self.preview_job)
        self.preview_job = self.root.after(450, lambda: self.preview(silent=True))

    def text_changed(self, event=None):
        if self.load_text.edit_modified():
            self.load_text.edit_modified(False)
            self.changed()

    def invalidate(self):
        self.result = None
        if self.result_window is not None and self.result_window.winfo_exists():
            self.result_window.destroy()
        self.result_window = self.canvas = self.toolbar = None
        if self.preview_canvas:
            self.preview_canvas.get_tk_widget().destroy()
            self.preview_canvas = None
        self.status.set("Исходные данные изменены. Выполните расчёт для получения новых результатов.")

    def values(self, keys):
        missing = [LABELS[key] for key in keys if not self.vars[key].get().strip()]
        if missing:
            raise InputError("Не заполнены обязательные поля:\n• " + "\n• ".join(missing))
        return {key: parse_number(self.vars[key].get(), LABELS[key]) for key in keys}

    def read_inputs(self):
        required = [key for key in DEFAULTS if key != "force"]
        if self.mode.get() == "Постоянная сила":
            required.append("force")
        values = self.values(required)
        constant_force = values.pop("force", None)
        duration = values.pop("duration")
        values["force_end"] = float(Decimal(str(values["force_start"])) + Decimal(str(duration)))
        values["method"] = METHODS[self.method.get()]
        p = Parameters(**values)
        factor = {"Н": 1., "кН": 1000., "кгс": 9.80665}[self.force_unit.get()]
        if self.mode.get() == "Постоянная сила":
            value = constant_force * factor
            load = Load([p.force_start, p.force_end], [value, value])
        else:
            rows = []
            for i, line in enumerate(self.load_text.get("1.0", "end").splitlines(), 1):
                if not line.strip():
                    continue
                pair = line.split(";") if ";" in line else line.split()
                if len(pair) != 2:
                    raise InputError(f"Строка {i}: нужны время и сила, разделённые точкой с запятой.")
                rows.append([parse_number(pair[0], f"Время в строке {i}"), parse_number(pair[1], f"Сила в строке {i}") * factor])
            if len(rows) < 2:
                raise InputError("Задайте не менее двух строк таблицы или загрузите CSV.")
            load = Load(*np.asarray(rows).T)
        load.validate_coverage(p)
        if not self.title.get().strip() or len(self.title.get()) > 200:
            raise InputError("Название расчёта должно содержать от 1 до 200 символов.")
        return p, load

    def next_page(self, index):
        try:
            if index == 1:
                v = self.values(["mass", "gravity", "initial_height", "initial_velocity"])
                if v["mass"] <= 0 or v["gravity"] <= 0 or v["initial_height"] < 0:
                    raise InputError("Масса и ускорение тяжести должны быть положительными, перемещение — неотрицательным.")
                if v["initial_height"] == 0 and v["initial_velocity"] < 0:
                    raise InputError("Отрицательная скорость при нулевой высоте задаёт начальный удар, который не моделируется.")
            else:
                self.read_inputs()
            self.tabs.select(index)
            self.preview(silent=True)
        except ValueError as exc:
            messagebox.showerror("Проверьте исходные данные", str(exc), parent=self.root)

    def preview(self, silent=False):
        self.preview_job = None
        try:
            p, load = self.read_inputs()
            if self.preview_canvas:
                self.preview_canvas.get_tk_widget().destroy()
            self.preview_canvas = FigureCanvasTkAgg(force_preview(p, load), master=self.preview_frame)
            self.preview_canvas.get_tk_widget().pack(fill="both", expand=True)
            self.preview_canvas.draw()
        except ValueError as exc:
            if not silent:
                messagebox.showerror("Проверьте нагрузку", str(exc), parent=self.root)

    def set_parameters(self, p):
        self.loading = True
        try:
            for field in fields(p):
                if field.name in self.vars:
                    self.vars[field.name].set(repr(getattr(p, field.name)))
            self.vars["duration"].set(str(Decimal(str(p.force_end)) - Decimal(str(p.force_start))))
            self.method.set(next(key for key, value in METHODS.items() if value == p.method))
        finally:
            self.loading = False
        self.parameters = p
        self.changed()

    def set_load(self, load):
        self.loading = True
        try:
            self.mode.set("Таблица / CSV")
            self.force_unit.set("Н")
            self.load_text.delete("1.0", "end")
            self.load_text.insert("1.0", "\n".join(f"{float(t)!r}; {float(f)!r}" for t, f in zip(load.time, load.force)))
            self.load_text.edit_modified(False)
        finally:
            self.loading = False
        self.load = load
        self.changed()

    def set_config(self, path):
        self.set_parameters(load_config(path))

    def set_force(self, path):
        self.set_load(load_force(path))

    def choose_config(self):
        path = filedialog.askopenfilename(title="Прежняя конфигурация JSON", filetypes=[("JSON", "*.json")])
        if path:
            self.guarded(lambda: self.set_config(path))

    def choose_force(self):
        path = filedialog.askopenfilename(title="CSV: time[s],Fz[N] (также ms, kN, kgf)", filetypes=[("CSV", "*.csv")])
        if path:
            self.guarded(lambda: self.set_force(path))

    def open_case(self):
        path = filedialog.askopenfilename(title="Вариант расчёта", filetypes=[("JSON", "*.json")])
        if path:
            def read():
                p, load, title = load_case(path)
                self.set_parameters(p)
                self.set_load(load)
                self.title.set(title)
                self.preview(silent=True)
                self.status.set(f"Вариант загружен: {path}")
            self.guarded(read)

    def save_inputs(self):
        def save():
            p, load = self.read_inputs()
            path = filedialog.asksaveasfilename(title="Сохранить исходные данные", defaultextension=".json", filetypes=[("JSON", "*.json")])
            if path:
                save_case(path, p, load, self.title.get().strip())
                self.status.set(f"Исходные данные сохранены: {path}")
        self.guarded(save)

    def guarded(self, action):
        try:
            action()
        except (ValueError, OSError, RuntimeError) as exc:
            messagebox.showerror("Ошибка", str(exc), parent=self.root)

    def sum_masses(self):
        win = tk.Toplevel(self.root)
        win.title("Суммирование масс")
        ttk.Label(win, text="Введите массы учитываемых компонентов в кг, по одной в строке.\n"
                  "Добавляйте только компоненты рассматриваемого тела.", padding=12).pack()
        text = tk.Text(win, width=45, height=10)
        text.pack(padx=12)
        def apply():
            def calculate():
                rows = [row for row in text.get("1.0", "end").splitlines() if row.strip()]
                masses = [parse_number(row, "Масса компонента") for row in rows]
                if not masses or any(m <= 0 for m in masses):
                    raise InputError("Введите положительные массы компонентов.")
                self.vars["mass"].set(f"{sum(masses):.15g}")
                win.destroy()
            self.guarded(calculate)
        ttk.Button(win, text="Записать суммарную массу", command=apply).pack(pady=12)

    def run(self):
        if self.busy:
            return
        try:
            p, load = self.read_inputs()
        except ValueError as exc:
            messagebox.showerror("Расчёт не запущен", str(exc), parent=self.root)
            return
        self.invalidate()
        self.parameters, self.load = p, load
        self.busy = True
        self.run_button.configure(state="disabled")
        self.progress.start()
        self.status.set("Выполняется расчёт…")
        revision, title = self.revision, self.title.get().strip()
        def worker():
            try:
                self.queue.put((simulate(p, load), None, revision, title))
            except Exception as exc:
                self.queue.put((None, str(exc), revision, title))
        Thread(target=worker, daemon=True).start()
        self.root.after(50, self.poll)

    def poll(self):
        try:
            result, error, revision, title = self.queue.get_nowait()
        except Empty:
            self.root.after(50, self.poll)
            return
        self.busy = False
        self.progress.stop()
        self.run_button.configure(state="normal")
        if revision != self.revision:
            self.status.set("Данные изменились во время вычисления. Запустите новый расчёт.")
            return
        if error:
            self.status.set("Расчёт не выполнен.")
            messagebox.showerror("Ошибка расчёта", error, parent=self.root)
            return
        self.result_title = title
        self.guarded(lambda: self.show_result(result))

    def show_result(self, result):
        self.result = result
        window = self.result_window = tk.Toplevel(self.root)
        window.title("Результаты — " + self.result_title)
        window.geometry("1200x900")
        window.minsize(900, 650)
        from .report import conclusion
        ttk.Label(window, text=conclusion(result), wraplength=1140, padding=12, font=("Segoe UI", 11, "bold")).pack(fill="x")
        actions = ttk.Frame(window, padding=(12, 0, 12, 8))
        actions.pack(fill="x")
        self.export_button = ttk.Button(actions, text="Экспорт результатов CSV…", command=self.export)
        self.export_button.pack(side="left")
        self.png_button = ttk.Button(actions, text="Сохранить PNG…", command=self.save_png)
        self.png_button.pack(side="left", padx=8)
        ttk.Button(actions, text="Сформировать отчёт PDF…", command=self.save_report).pack(side="left")
        tabs = ttk.Notebook(window)
        tabs.pack(fill="both", expand=True, padx=12, pady=10)
        table_frame, plot_frame = ttk.Frame(tabs), ttk.Frame(tabs)
        tabs.add(table_frame, text="Итоговая таблица")
        tabs.add(plot_frame, text="Шесть графиков")
        self.table = ttk.Treeview(table_frame, columns=("parameter", "value", "unit", "note"), show="headings")
        for name, label, width in (("parameter", "Показатель", 440), ("value", "Результат", 150), ("unit", "Единицы", 70), ("note", "Примечание", 400)):
            self.table.heading(name, text=label)
            self.table.column(name, width=width, minwidth=60)
        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        horizontal = ttk.Scrollbar(table_frame, orient="horizontal", command=self.table.xview)
        self.table.configure(yscrollcommand=scroll.set, xscrollcommand=horizontal.set)
        scroll.pack(side="right", fill="y")
        horizontal.pack(side="bottom", fill="x")
        self.table.pack(fill="both", expand=True)
        for row in summary(result).itertuples(index=False):
            self.table.insert("", "end", values=(row.parameter, format_value(row.value), row.unit, row.note))
        self.canvas = FigureCanvasTkAgg(make_figure(result), master=plot_frame)
        self.toolbar = RussianToolbar(self.canvas, plot_frame, pack_toolbar=False)
        self.toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.draw()
        self.status.set(conclusion(result))

    def export(self):
        if self.result is None:
            return
        directory = filedialog.askdirectory(title="Папка для результатов")
        if directory:
            def write():
                output = Path(directory) / datetime.now().strftime("calculation_%Y%m%d_%H%M%S_%f")
                export_csv(self.result, output)
                self.status.set(f"Результаты сохранены: {output}")
            self.guarded(write)

    def save_png(self):
        if self.result is None:
            return
        path = filedialog.asksaveasfilename(title="Сохранить графики", defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            self.guarded(lambda: make_figure(self.result).savefig(path, dpi=160))

    def save_report(self):
        if self.result is None:
            return
        path = filedialog.asksaveasfilename(title="Сформировать отчёт", defaultextension=".pdf", filetypes=[("PDF", "*.pdf")])
        if path:
            from .report import create_report
            def write():
                create_report(self.result, path, self.result_title)
                self.status.set(f"Отчёт сохранён: {path}")
            self.guarded(write)


def launch(config=None, force=None):
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise RuntimeError("Для запуска нужен Python с Tcl/Tk и рабочий стол. " + str(exc)) from exc
    app = Application(root)
    if config:
        app.guarded(lambda: app.set_config(config))
    if force:
        app.guarded(lambda: app.set_force(force))
    root.mainloop()
