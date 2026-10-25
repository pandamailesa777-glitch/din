"""Portable PDF report with embedded Cyrillic fonts and complete input history."""
from datetime import datetime, timezone, timedelta
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                               Table, TableStyle, Spacer, PageBreak, NextPageTemplate, Image)

from .plotting import make_figure
from .results import summary, format_value


def conclusion(result):
    if not result.detached:
        return "Полный вертикальный отрыв отсутствует в заданном интервале."
    prefix = "Конструкция изначально находилась над основанием. " if result.initially_airborne else "Полный вертикальный отрыв произошёл. "
    if result.touchdown is not None:
        return prefix + f"Первое касание зарегистрировано при t = {result.touchdown:.9g} с."
    return prefix + "Возвращение в заданном интервале не зарегистрировано."


def create_report(result, path, title="Расчёт вертикального движения"):
    font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    for name, file in (("VD", "DejaVuSans.ttf"), ("VD-Bold", "DejaVuSans-Bold.ttf")):
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(font_dir / file)))
    normal = ParagraphStyle("Body", fontName="VD", fontSize=9, leading=13, spaceAfter=7)
    heading = ParagraphStyle("Heading", parent=normal, fontName="VD-Bold", fontSize=13, leading=18, spaceAfter=12)
    small = ParagraphStyle("Small", parent=normal, fontSize=8, leading=11, spaceAfter=0)
    def paragraph(text, style=normal):
        return Paragraph(escape(str(text)).replace("\n", "<br/>"), style)
    def table(rows, widths):
        content = [[paragraph(cell, small) for cell in row] for row in rows]
        obj = Table(content, colWidths=widths, repeatRows=1, hAlign="LEFT")
        obj.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dceaf1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LINEBELOW", (0, 0), (-1, 0), .6, colors.HexColor("#7699aa")),
        ]))
        return obj
    def footer(canvas, doc):
        canvas.setFont("VD", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(36, 22, "Вертикальная динамика | Предварительный расчёт")
        canvas.drawRightString(canvas._pagesize[0] - 36, 22, str(doc.page))
    doc = BaseDocTemplate(str(path), pagesize=A4, title=title, author="Вертикальная динамика")
    templates = []
    for name, size in (("portrait", A4), ("landscape", landscape(A4))):
        frame = Frame(36, 40, size[0] - 72, size[1] - 76, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        templates.append(PageTemplate(id=name, frames=[frame], pagesize=size, onPage=footer))
    doc.addPageTemplates(templates)
    p = result.parameters
    created = datetime.now(timezone(timedelta(hours=3))).strftime("%d.%m.%Y %H:%M:%S (UTC+03:00)")
    story = [paragraph(title, heading), paragraph("Дата формирования: " + created),
             paragraph(conclusion(result), heading),
             paragraph("Отсутствие полного вертикального отрыва не исключает частичного подъёма отдельных опор и не подтверждает безопасность конструкции."),
             paragraph("1. Исходные данные", heading)]
    rows = [["Параметр", "Значение"]]
    for label, value, unit in (
        ("Масса конструкции", p.mass, "кг"), ("Ускорение тяжести", p.gravity, "м/с²"),
        ("Начальное перемещение", p.initial_height, "м"), ("Начальная скорость", p.initial_velocity, "м/с"),
        ("Начало воздействия", p.force_start, "с"), ("Конец воздействия", p.force_end, "с"),
        ("Продолжительность воздействия", p.force_end-p.force_start, "с"), ("Общее время моделирования", p.end_time, "с"),
        ("Шаг Эйлера", p.euler_step, "с"), ("Максимальный шаг RK45", p.max_step, "с"),
        ("Шаг вывода", p.output_step, "с"), ("Относительная точность RK45", p.rtol, ""),
        ("Абсолютная точность перемещения", p.atol_z, "м"), ("Абсолютная точность скорости", p.atol_v, "м/с")):
        rows.append([label, f"{value:.12g} {unit}"])
    story += [table(rows, [330, 193]), Spacer(1, 12),
              paragraph("Метод интегрирования: " + ("явный метод Эйлера первого порядка" if p.method == "Euler" else "адаптивный RK45 (SciPy solve_ivp)")),
              paragraph("Знак силы и скорости: вверх положительно. Сила задаётся пользователем, не выводится из характеристик внешних агрегатов."), PageBreak(),
              paragraph("2. Математическая модель и допущения", heading),
              paragraph("Абсолютно твёрдое тело, одна поступательная степень свободы. Неподвижное горизонтальное основание: z = 0. Постоянные масса m и ускорение тяжести g.\n"
                        "dz/dt = v; m dv/dt = F(t) - mg + N.\n"
                        "Односторонний контакт: z ≥ 0, N ≥ 0, zN = 0. При покое на основании N = mg - F, пока F ≤ mg. При F > mg начинается свободное движение; в полёте N = 0.\n"
                        "Сила линейно интерполируется между узлами таблицы внутри [t0, t1); вне этого интервала F = 0. Все границы нагрузки обрабатываются явно.\n"
                        "RK45 обнаруживает касание по событию z = 0. Эйлер использует z[k+1] = z[k] + h v[k], v[k+1] = v[k] + h a[k]; касание уточняется на линейном участке высоты. Первый шаг Эйлера из покоя может иметь нулевую высоту при уже положительной скорости.\n"
                        "Расчёт останавливается при первом касании. Скорость приводится непосредственно перед ударом. Реакция и ускорение удара, а также последующие отскоки не определяются."),
              paragraph("3. Основные результаты", heading)]
    rows = [["Показатель", "Результат"]]
    for row in summary(result).itertuples(index=False):
        value = format_value(row.value)
        if value != "Не определено":
            value += " " + row.unit
        if row.note:
            value += "\n" + row.note
        rows.append([row.parameter, value])
    story += [table(rows, [310, 213]), Spacer(1, 12), paragraph("4. Ограничения применимости", heading),
              paragraph("Не учитываются вращение, опрокидывание, деформации, частичный отрыв опор, контактная податливость и ударные напряжения. Масса отделяющихся компонентов автоматически не добавляется. Для реального оборудования требуется независимая проверка заданной нагрузки и физических допущений. Пространственная динамика требует отдельной модели.\n"
                        "Максимумы приведены только для рассчитанного интервала. Если возврат не зарегистрирован, это не означает, что конструкция не вернётся позднее. Результаты Эйлера необходимо проверять последовательным уменьшением шага."),
              NextPageTemplate("landscape"), PageBreak(), paragraph("5. Графики", heading)]
    stream = BytesIO()
    figure = make_figure(result)
    figure.savefig(stream, format="png", dpi=180)
    stream.seek(0)
    story += [Image(stream, width=560, height=466.67), NextPageTemplate("portrait"), PageBreak(),
              paragraph("Приложение. Полная таблица внешней нагрузки", heading),
              paragraph("Время - секунды; сила - ньютоны. Узлы вне интервала воздействия не действуют; экстраполяция запрещена.")]
    rows = [["Время, с", "Внешняя сила, Н"]]
    rows.extend([[repr(float(t)), repr(float(f))] for t, f in zip(result.load.time, result.load.force)])
    story.append(table(rows, [261.5, 261.5]))
    doc.build(story)
    return Path(path)
