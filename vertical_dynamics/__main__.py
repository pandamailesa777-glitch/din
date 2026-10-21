"""GUI by default; reproducible command-line execution when files are supplied."""
import argparse
from pathlib import Path
import sys

from .inputs import load_config, load_force
from .solver import simulate


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Анализ вертикальной динамики жёсткого тела на основании.")
    parser.add_argument("--config", type=Path, help="Конфигурация JSON с явными единицами")
    parser.add_argument("--force", type=Path, help="CSV внешней силы")
    parser.add_argument("--output", type=Path, help="Папка результатов CLI (по умолчанию output)")
    parser.add_argument("--gui", action="store_true", help="Открыть графический интерфейс с указанными файлами")
    args = parser.parse_args(argv)
    if args.gui or (args.config is None and args.force is None and args.output is None):
        if args.gui and args.output:
            parser.error("--output применяется только к консольному запуску")
        try:
            from .gui import launch
            launch(args.config, args.force)
        except (ImportError, RuntimeError) as exc:
            print(f"Ошибка запуска интерфейса: {exc}", file=sys.stderr)
            return 1
        return 0
    if args.config is None or args.force is None:
        parser.error("для расчёта необходимы оба файла: --config и --force")
    try:
        import matplotlib
        matplotlib.use("Agg")
        from .plotting import save_png
        from .results import export_csv, format_value, summary
        result = simulate(load_config(args.config), load_force(args.force))
        output = args.output or Path("output")
        export_csv(result, output)
        save_png(result, output / "plots.png")
        for row in summary(result).itertuples(index=False):
            note = f" ({row.note})" if row.note else ""
            print(f"{row.parameter}: {format_value(row.value)} {row.unit}{note}")
        print(f"Результаты сохранены: {output.resolve()}")
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
