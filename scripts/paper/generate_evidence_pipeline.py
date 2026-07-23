"""Render Figure 1 as deterministic vector PDF from verified draw.io labels."""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

from reportlab.lib.colors import HexColor
from reportlab.pdfgen.canvas import Canvas


PAGE = (900, 590)
LATEX_TARGET_WIDTH = 459.0
MIN_LABEL_FONT_SIZE = 16.0
FONT = "Helvetica"
INK = HexColor("#263238")
MUTED = HexColor("#59636e")
PANEL = HexColor("#f7f9fa")
WHITE = HexColor("#ffffff")
BLUE = (HexColor("#e3eef6"), HexColor("#527b9e"))
NEUTRAL = (HexColor("#eceff1"), HexColor("#78909c"))
GREEN = (HexColor("#e7f1ed"), HexColor("#477c74"))
AMBER = (HexColor("#fff3d8"), HexColor("#b27a21"))
RED = (HexColor("#f7e5e1"), HexColor("#9f3f2c"))


EXPECTED_VERTICES = {
    "2": ("Evidence-bound comparison and demonstration lanes", "1", (20, 10, 440, 30)),
    "3": ("Scope map; no new results", "1", (600, 10, 280, 30)),
    "4": ("a  Paper RQ1", "1", (20, 50, 860, 160)),
    "5": ("Protected validation\nadaptive development", "4", (15, 60, 155, 60)),
    "6": ("L191-C0-clean-v3-IMP-control", "4", (215, 35, 230, 50)),
    "7": ("L192-nnUNet-v2-raw-100ep", "4", (215, 100, 230, 50)),
    "8": ("Historical descriptive\npoint estimates", "4", (495, 60, 155, 60)),
    "9": ("Claim gate\nRQ1 adaptive only\nno live equivalence", "4", (685, 45, 155, 90)),
    "10": ("b  Fixed-cache demo", "1", (20, 225, 860, 160)),
    "11": ("Train-screen\nfixed allowlist", "10", (15, 60, 155, 60)),
    "12": ("L206 zero-channel cache\nL206-control-s206", "10", (215, 35, 230, 50)),
    "13": ("L206 contour-channel cache\nL206-contour-channel-s206", "10", (215, 100, 230, 50)),
    "14": ("Audited metrics\nfixed ground truth", "10", (495, 60, 155, 60)),
    "15": ("Claim gate\nfixed cache only\nnot live L192", "10", (685, 45, 155, 90)),
    "16": ("c  Live dual demo", "1", (20, 400, 860, 160)),
    "17": ("Public/synthetic RGB\nsame RGB, both arms", "16", (15, 60, 155, 60)),
    "18": ("live L206-control-s206\nIMP arm", "16", (215, 35, 230, 50)),
    "19": ("reconstructed L192 nnU-Net\nL192-nnUNet-v2-raw-100ep", "16", (215, 100, 230, 50)),
    "20": ("Illustrative masks\nconditional receipt\nno ground truth", "16", (495, 60, 155, 60)),
    "21": ("Claim gate\nboth arms required\nno accuracy\nno RQ1 equivalence\nE2E unverified", "16", (685, 40, 155, 100)),
}
EXPECTED_EDGES = {
    "22": ("4", "5", "6"), "23": ("4", "5", "7"),
    "24": ("4", "6", "8"), "25": ("4", "7", "8"), "26": ("4", "8", "9"),
    "27": ("10", "11", "12"), "28": ("10", "11", "13"),
    "29": ("10", "12", "14"), "30": ("10", "13", "14"), "31": ("10", "14", "15"),
    "32": ("16", "17", "18"), "33": ("16", "17", "19"),
    "34": ("16", "18", "20"), "35": ("16", "19", "20"), "36": ("16", "20", "21"),
}
LANE_IDS = (("4", "5", "6", "7", "8", "9"), ("10", "11", "12", "13", "14", "15"), ("16", "17", "18", "19", "20", "21"))


def _mismatch(detail: str) -> ValueError:
    return ValueError(f"draw.io source spec mismatch: {detail}")


def _geometry(cell: ET.Element) -> tuple[int, int, int, int]:
    geometry = cell.find("mxGeometry")
    if geometry is None:
        raise _mismatch(f"cell {cell.attrib.get('id')} geometry")
    try:
        return tuple(int(geometry.attrib[key]) for key in ("x", "y", "width", "height"))
    except (KeyError, ValueError) as exc:
        raise _mismatch(f"cell {cell.attrib.get('id')} geometry") from exc


def load_figure_spec(source: Path) -> dict[str, ET.Element]:
    try:
        root = ET.parse(source).getroot()
    except (ET.ParseError, OSError) as exc:
        raise _mismatch("invalid XML") from exc
    model = root.find(".//mxGraphModel")
    if model is None or (model.attrib.get("pageWidth"), model.attrib.get("pageHeight")) != ("900", "590"):
        raise _mismatch("page layout")
    cell_list = list(root.iter("mxCell"))
    cell_ids = [cell.attrib.get("id", "") for cell in cell_list]
    if len(cell_ids) != len(set(cell_ids)):
        raise _mismatch("duplicate cell IDs")
    cells = dict(zip(cell_ids, cell_list, strict=True))
    if set(cells) != {str(value) for value in range(37)}:
        raise _mismatch("cell IDs")
    if cells["0"].attrib != {"id": "0"} or cells["1"].attrib != {"id": "1", "parent": "0"}:
        raise _mismatch("root cells")
    for cell_id, (label, parent, geometry) in EXPECTED_VERTICES.items():
        cell = cells[cell_id]
        if cell.attrib.get("vertex") != "1" or cell.attrib.get("value") != label or cell.attrib.get("parent") != parent:
            raise _mismatch(f"vertex {cell_id}")
        if _geometry(cell) != geometry:
            raise _mismatch(f"vertex {cell_id} layout")
    for cell_id, (parent, source_id, target_id) in EXPECTED_EDGES.items():
        cell = cells[cell_id]
        if (
            cell.attrib.get("edge") != "1"
            or cell.attrib.get("parent") != parent
            or cell.attrib.get("source") != source_id
            or cell.attrib.get("target") != target_id
        ):
            raise _mismatch(f"edge {cell_id}")
        geometry = cell.find("mxGeometry")
        if geometry is None or geometry.attrib != {"relative": "1", "as": "geometry"}:
            raise _mismatch(f"edge {cell_id} layout")
    return cells


def _box(
    canvas: Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    colors: tuple,
) -> None:
    fill, stroke = colors
    canvas.setFillColor(fill)
    canvas.setStrokeColor(stroke)
    canvas.setLineWidth(1)
    canvas.roundRect(x, y, width, height, 5, stroke=1, fill=1)
    lines = text.splitlines()
    leading = MIN_LABEL_FONT_SIZE + 1
    baseline = (
        y
        + height / 2
        + (len(lines) - 1) * leading / 2
        - MIN_LABEL_FONT_SIZE * 0.35
    )
    canvas.setFont(FONT, MIN_LABEL_FONT_SIZE)
    canvas.setFillColor(INK)
    for index, line in enumerate(lines):
        canvas.drawCentredString(x + width / 2, baseline - index * leading, line)


def _arrow(
    canvas: Canvas,
    start: tuple[float, float],
    end: tuple[float, float],
) -> None:
    x1, y1 = start
    x2, y2 = end
    mid = (x1 + x2) / 2
    canvas.setStrokeColor(MUTED)
    canvas.setFillColor(MUTED)
    canvas.setLineWidth(1)
    path = canvas.beginPath()
    path.moveTo(x1, y1)
    path.lineTo(mid, y1)
    path.lineTo(mid, y2)
    path.lineTo(x2 - 7, y2)
    canvas.drawPath(path, stroke=1, fill=0)
    head = canvas.beginPath()
    head.moveTo(x2, y2)
    head.lineTo(x2 - 7, y2 + 3.5)
    head.lineTo(x2 - 7, y2 - 3.5)
    head.close()
    canvas.drawPath(head, stroke=0, fill=1)


def _absolute_geometry(
    cells: dict[str, ET.Element], cell_id: str
) -> tuple[int, int, int, int]:
    x, y, width, height = _geometry(cells[cell_id])
    parent = cells[cell_id].attrib.get("parent")
    if parent in EXPECTED_VERTICES:
        parent_x, parent_y, _, _ = _absolute_geometry(cells, parent)
        x += parent_x
        y += parent_y
    return x, y, width, height


def _pdf_geometry(
    cells: dict[str, ET.Element], cell_id: str
) -> tuple[int, int, int, int]:
    x, y, width, height = _absolute_geometry(cells, cell_id)
    return x, PAGE[1] - y - height, width, height


def render(source: Path, output: Path) -> None:
    cells = load_figure_spec(source)
    canvas = Canvas(str(output), pagesize=PAGE, pageCompression=1, invariant=1)
    canvas.setTitle("Evidence-bound comparison and demonstration lanes")
    canvas.setFillColor(WHITE)
    canvas.rect(0, 0, PAGE[0], PAGE[1], stroke=0, fill=1)
    canvas.setFillColor(INK)
    canvas.setFont("Helvetica-Bold", 20)
    canvas.drawString(20, 565, cells["2"].attrib["value"])
    canvas.setFillColor(MUTED)
    canvas.setFont(FONT, MIN_LABEL_FONT_SIZE)
    canvas.drawRightString(880, 566, cells["3"].attrib["value"])

    color_by_id = {
        "5": NEUTRAL, "6": BLUE, "7": BLUE, "8": AMBER, "9": RED,
        "11": GREEN, "12": BLUE, "13": AMBER, "14": GREEN, "15": RED,
        "17": NEUTRAL, "18": BLUE,
        "19": (HexColor("#e8edf4"), HexColor("#687d98")),
        "20": GREEN, "21": RED,
    }
    for lane_id, *child_ids in LANE_IDS:
        x, y, width, height = _pdf_geometry(cells, lane_id)
        canvas.setFillColor(PANEL)
        canvas.setStrokeColor(NEUTRAL[1])
        canvas.setLineWidth(1)
        canvas.rect(x, y, width, height, stroke=1, fill=1)
        canvas.setFillColor(INK)
        canvas.setFont("Helvetica-Bold", MIN_LABEL_FONT_SIZE)
        canvas.drawString(x + 12, y + height - 22, cells[lane_id].attrib["value"])

        for child_id in child_ids:
            _box(
                canvas,
                *_pdf_geometry(cells, child_id),
                cells[child_id].attrib["value"],
                color_by_id[child_id],
            )

    for _parent, source_id, target_id in EXPECTED_EDGES.values():
        source_box = _pdf_geometry(cells, source_id)
        target_box = _pdf_geometry(cells, target_id)
        _arrow(
            canvas,
            (source_box[0] + source_box[2], source_box[1] + source_box[3] / 2),
            (target_box[0], target_box[1] + target_box[3] / 2),
        )

    canvas.showPage()
    canvas.save()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.source, args.output)


if __name__ == "__main__":
    main()
