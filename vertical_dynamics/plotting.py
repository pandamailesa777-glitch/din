"""Six diagnostics with event markers and cursor data, shared by GUI and PDF."""
import numpy as np
from matplotlib.figure import Figure

from .results import sample, summary, time_history


def force_preview(p, load):
    figure = Figure(figsize=(8, 3), layout="constrained")
    ax = figure.subplots()
    times = np.unique(np.r_[0., p.end_time, p.force_start, p.force_end, load.time,
                           [np.nextafter(t, -np.inf) for t in (p.force_start, p.force_end) if t > 0]])
    times = times[(times >= 0) & (times <= p.end_time)]
    ax.plot(times, load.value(times, p), color="#15749b")
    ax.axhline(p.mass * p.gravity, color="#b97424", linestyle=":", label="Вес mg")
    ax.set(xlabel="Время, с", ylabel="Внешняя сила, Н", title="Заданная сила: положительное направление вверх")
    ax.grid(alpha=.2)
    ax.legend()
    return figure


def make_figure(result):
    history = time_history(result)
    p = result.parameters
    figure = Figure(figsize=(12, 10), layout="constrained", facecolor="#f5f7fa")
    axes = figure.subplots(3, 2).ravel()
    figure.suptitle("Вертикальное движение • " + ("метод Эйлера" if p.method == "Euler" else "RK45"), fontsize=15)
    boundaries = [p.force_start, p.force_end, result.lift_off]
    extra = [np.nextafter(t, -np.inf) for t in boundaries if t is not None and 0 < t <= result.stop_time]
    times = np.unique(np.r_[history.time_s, extra])
    dense = sample(result, times)
    keys = ["Fz_N", "reaction_N", "az_m_s2", "vz_m_s", "z_m", "contact"]
    labels = ["Внешняя сила, Н", "Реакция основания, Н", "Ускорение, м/с²", "Скорость, м/с", "Перемещение, м", "Контакт"]
    peak = summary(result).set_index("key").loc["peak_time", "value"]
    events = [(p.force_start, "Начало воздействия", "#888888", ":"),
              (p.force_end, "Конец воздействия", "#888888", "-."),
              (result.lift_off, "Отрыв", "#248658", "--"),
              (peak, "Максимум высоты", "#a67020", ":"),
              (result.touchdown, "Касание", "#c34646", "--")]
    for ax, key, label in zip(axes, keys, labels):
        ax.set_facecolor("white")
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        if key == "az_m_s2" and result.pieces:
            first = result.pieces[0].start
            ax.plot([0, first], [0, 0], color="#1978a5")
            for piece in result.pieces:
                ax.plot([piece.start, piece.end], [piece.acceleration(piece.start), piece.acceleration(piece.end)], color="#1978a5")
        elif key == "contact":
            ax.step(times, dense.contact.astype(int), where="post", color="#1978a5")
            ax.set_yticks([0, 1], ["Нет", "Есть"])
            ax.set_ylim(-.15, 1.15)
            if result.touchdown is not None:
                ax.scatter([result.touchdown], [1], color="#c34646", zorder=5)
        else:
            ax.plot(times, dense[key], color="#1978a5", linewidth=1.6)
        ax.set(title=label, ylabel=label, xlabel="Время, с", xlim=(0, max(result.stop_time, 1e-9)))
        for t, name, color, style in events:
            if t is not None and t <= result.stop_time:
                ax.axvline(t, color=color, linestyle=style, linewidth=.9, label=f"{name}: {t:.5g} с")
        def coordinate(x, y, key=key):
            if not 0 <= x <= result.stop_time:
                return ""
            value = sample(result, [x]).iloc[0][key]
            shown = ("Есть" if value else "Нет") if key == "contact" else ("Не определено" if np.isnan(value) else f"{value:.7g}")
            return f"t = {x:.7g} с; значение = {shown}"
        ax.format_coord = coordinate
    axes[0].legend(fontsize=7, loc="best")
    if peak is not None:
        axes[4].scatter([peak], [sample(result, [peak]).z_m.iloc[0]], color="#a67020", zorder=5)
    return figure


def save_png(result, path):
    make_figure(result).savefig(path, dpi=160)
    return path
