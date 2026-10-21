"""Five diagnostic plots, usable both without a display and in Tkinter."""
import numpy as np
from matplotlib.figure import Figure

from .results import time_history


def make_figure(result):
    history = time_history(result)
    figure = Figure(figsize=(11, 11), layout="constrained", facecolor="#f5f7fa")
    axes = figure.subplots(5, 1, sharex=True)
    figure.suptitle("Вертикальное движение • односторонний контакт", fontsize=15, fontweight="bold")
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(True, alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].plot(history.time_s, history.z_m, color="#1978a5", linewidth=1.8)
    axes[1].plot(history.time_s, history.vz_m_s, color="#1978a5", linewidth=1.8)
    # Draw acceleration pieces separately so jumps have zero horizontal duration.
    if not result.pieces:
        axes[2].plot([0, result.stop_time], [0, 0], color="#1978a5")
    else:
        first = result.pieces[0]
        if first.start > 0:
            axes[2].plot([0, first.start], [0, 0], color="#1978a5")
        for piece in result.pieces:
            axes[2].plot([piece.start, piece.end],
                         [piece.acceleration(piece.start), piece.acceleration(piece.end)], color="#1978a5", linewidth=1.8)
    # Include the left-hand limit of both external-force jumps.
    boundaries = [result.parameters.force_start, result.parameters.force_end]
    extra = [np.nextafter(t, -np.inf) for t in boundaries if 0 < t <= result.stop_time]
    force_times = np.unique(np.r_[history.time_s, extra])
    force_values = result.load.value(force_times, result.parameters)
    axes[3].plot(force_times, force_values, color="#94612b", linewidth=1.8)
    axes[3].axhline(result.parameters.mass * result.parameters.gravity,
                   color="#8c8c8c", linestyle=":", label="mg")
    axes[3].legend(loc="upper right", fontsize=8)
    contact_codes = history.state.map({"contact": 0, "flight": 1, "touchdown": 2})
    axes[4].step(history.time_s, contact_codes, where="post", color="#1978a5", linewidth=1.8)
    if result.touchdown is not None:
        axes[4].scatter([result.touchdown], [2], color="#c34646", zorder=5)
    axes[4].set_yticks([0, 1, 2], ["На основании", "Полёт", "Касание"])
    axes[4].set_ylim(-0.25, 2.25)
    for ax, label in zip(axes, ["Высота, м", "Скорость, м/с", "Ускорение, м/с²", "Сила Fz, Н", "Состояние"]):
        ax.set_ylabel(label)
        for time, name, color in ((result.lift_off, "Отрыв", "#248658"), (result.touchdown, "Касание", "#c34646")):
            if time is not None:
                ax.axvline(time, color=color, linestyle="--", linewidth=1, label=f"{name}: {time:.6g} с")
    if result.detached:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(handles, labels, loc="upper right", fontsize=9)
    axes[-1].set_xlabel("Время, с")
    axes[-1].set_xlim(0, max(result.stop_time, 1e-9))
    return figure


def save_png(result, path):
    figure = make_figure(result)
    figure.savefig(path, dpi=160)
    return path
