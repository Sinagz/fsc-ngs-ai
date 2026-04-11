"""
Excel export for lookup results.
Returns bytes (in-memory workbook) so Streamlit can serve it as a download.
"""
from __future__ import annotations

import io
from datetime import date

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from app.lookup_engine import LookupResult, FeeCode

# (key, header label, col width)
COLUMNS: list[tuple[str, str, int]] = [
    ("anchor_code",          "Anchor Code",             14),
    ("row_type",             "Row Type",                10),
    ("sim_score",            "Similarity",              10),
    ("score_method",         "Method",                   9),
    ("province",             "Province",                 8),
    ("fsc_code",             "FSC Code",                12),
    ("fsc_fn",               "FSC Name / Function",     48),
    ("fsc_chapter",          "Chapter",                 26),
    ("fsc_section",          "Section",                 26),
    ("fsc_subsection",       "Subsection",              22),
    ("fsc_description",      "Description",             52),
    ("price",                "Price",                    9),
    ("fsc_rationale",        "FSC Rationale",           34),
    ("fsc_confidence",       "FSC Confidence",          15),
    ("page",                 "Page",                     6),
    ("fsc_key_observations", "FSC Key Observations",    26),
    ("fsc_notes",            "FSC Notes",               36),
    ("fsc_others",           "FSC Others",              22),
    ("NGS_code",             "NGS Code",                11),
    ("NGS_label",            "NGS Label",               36),
    ("NGS_rationale",        "NGS Rationale",           42),
    ("NGS_confidence",       "NGS Confidence",          16),
    ("NGS_key_observations", "NGS Key Observations",    28),
    ("NGS_notes",            "NGS Notes",               32),
    ("NGS_other",            "NGS Other",               22),
    ("map_date",             "Map Date",                12),
]

_HDR_FILL    = PatternFill("solid", fgColor="1F4E79")
_ANCHOR_FILL = PatternFill("solid", fgColor="FFE699")
_NGS_FILL    = PatternFill("solid", fgColor="E2EFDA")
_PROV_FILLS  = {
    "ON": PatternFill("solid", fgColor="DDEEFF"),
    "BC": PatternFill("solid", fgColor="E8F8E8"),
    "YT": PatternFill("solid", fgColor="FFF3DD"),
}

_HDR_FONT    = Font(bold=True, color="FFFFFF", size=10)
_ANCHOR_FONT = Font(bold=True, size=10)
_BODY_FONT   = Font(size=10)
_HDR_ALIGN   = Alignment(horizontal="center", vertical="center", wrap_text=True)
_BODY_ALIGN  = Alignment(vertical="top", wrap_text=False)

_THIN  = Side(border_style="thin",   color="CCCCCC")
_THICK = Side(border_style="medium", color="888888")
_CELL_BORDER = Border(left=_THIN, right=_THIN, top=_THIN,  bottom=_THIN)
_GROUP_TOP   = Border(left=_THIN, right=_THIN, top=_THICK, bottom=_THIN)


def _fee_to_row(fc: FeeCode, anchor_code: str, row_type: str,
                sim_score: float | None, score_method: str,
                ngs_match: bool, today: str) -> dict:
    return {
        "anchor_code":          anchor_code,
        "row_type":             row_type,
        "sim_score":            "1.000" if row_type == "ANCHOR" else (
                                    f"{sim_score:.3f}" if sim_score is not None else ""),
        "score_method":         score_method,
        "province":             fc.province,
        "fsc_code":             fc.fsc_code,
        "fsc_fn":               fc.fsc_fn,
        "fsc_chapter":          fc.fsc_chapter,
        "fsc_section":          fc.fsc_section,
        "fsc_subsection":       fc.fsc_subsection,
        "fsc_description":      fc.fsc_description,
        "price":                fc.price,
        "fsc_rationale":        fc.fsc_rationale,
        "fsc_confidence":       fc.fsc_confidence,
        "page":                 str(fc.page),
        "fsc_key_observations": fc.fsc_key_observations,
        "fsc_notes":            fc.fsc_notes,
        "fsc_others":           fc.fsc_others,
        "NGS_code":             fc.NGS_code,
        "NGS_label":            fc.NGS_label,
        "NGS_rationale":        fc.NGS_rationale,
        "NGS_confidence":       fc.NGS_confidence,
        "NGS_key_observations": fc.NGS_key_observations,
        "NGS_notes":            fc.NGS_notes,
        "NGS_other":            fc.NGS_other,
        "map_date":             today,
        "_ngs_match":           ngs_match,
        "_is_group_start":      False,
    }


def build_workbook(results: list[LookupResult]) -> bytes:
    today    = date.today().isoformat()
    col_keys = [c[0] for c in COLUMNS]
    col_hdrs = [c[1] for c in COLUMNS]
    col_wids = [c[2] for c in COLUMNS]

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "FSC Lookup Results"

    # Header
    for ci, (lbl, wid) in enumerate(zip(col_hdrs, col_wids), 1):
        cell = ws.cell(row=1, column=ci, value=lbl)
        cell.font = _HDR_FONT
        cell.fill = _HDR_FILL
        cell.alignment = _HDR_ALIGN
        cell.border = _CELL_BORDER
        ws.column_dimensions[get_column_letter(ci)].width = wid
    ws.row_dimensions[1].height = 28
    ws.freeze_panes = "A2"

    rows: list[dict] = []

    for result in results:
        anchor_code = result.anchor.fsc_code
        method      = result.score_method

        # Anchor row
        arow = _fee_to_row(result.anchor, anchor_code, "ANCHOR",
                           None, method, False, today)
        arow["_is_group_start"] = True
        rows.append(arow)

        # Match rows
        for mr in result.matches:
            mrow = _fee_to_row(mr.fee_code, anchor_code, "MATCH",
                               mr.sim_score, mr.score_method, mr.ngs_match, today)
            rows.append(mrow)

    for ri, row_dict in enumerate(rows, start=2):
        is_anchor      = row_dict["row_type"] == "ANCHOR"
        is_group_start = row_dict.pop("_is_group_start", False)
        ngs_match      = row_dict.pop("_ngs_match", False)
        province       = row_dict.get("province", "")

        for ci, key in enumerate(col_keys, 1):
            cell = ws.cell(row=ri, column=ci, value=str(row_dict.get(key, "")))
            cell.border    = _GROUP_TOP if is_group_start else _CELL_BORDER
            cell.alignment = _BODY_ALIGN

            if is_anchor:
                cell.fill = _ANCHOR_FILL
                cell.font = _ANCHOR_FONT
            elif ngs_match:
                cell.fill = _NGS_FILL
                cell.font = _BODY_FONT
            else:
                cell.fill = _PROV_FILLS.get(province, PatternFill())
                cell.font = _BODY_FONT

    ws.auto_filter.ref = f"A1:{get_column_letter(len(col_keys))}{ws.max_row}"

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
