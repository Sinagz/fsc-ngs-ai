"""FSC Cross-Province Lookup — Streamlit app.

Run:  streamlit run app/main.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from dotenv import load_dotenv

from app.excel_export import build_workbook
from src.core.loader import load_latest
from src.core.matching import LookupResult, search

load_dotenv()

# ── page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FSC Cross-Province Lookup",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
/* Tighten default streamlit padding a bit */
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }

/* Hierarchy helpers */
.eyebrow { color: #6b7280; font-size: 0.78em; text-transform: uppercase;
           letter-spacing: 0.06em; font-weight: 600; margin: 0 0 2px 0; }
.anchor-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px;
               padding: 24px 28px; margin-bottom: 4px; }
.anchor-code { font-size: 1.9em; font-weight: 700; color: #0f172a;
               font-family: 'JetBrains Mono', ui-monospace, monospace; }
.anchor-desc { color: #334155; line-height: 1.55; margin-top: 4px; }

/* Badges */
.badge         { display: inline-block; border-radius: 999px; padding: 3px 10px;
                 font-size: 0.78em; font-weight: 600; letter-spacing: 0.01em;
                 vertical-align: middle; }
.badge-prov    { background: #1e293b; color: #f1f5f9; }
.badge-ngs     { background: #dcfce7; color: #14532d; }
.badge-ngs-no  { background: #fef3c7; color: #78350f; }
.badge-same    { background: #dbeafe; color: #1e3a8a; }
.badge-diff    { background: #f3f4f6; color: #475569; }
.badge-method  { background: #ede9fe; color: #5b21b6; text-transform: uppercase;
                 font-size: 0.72em; }
.badge-sim     { background: #e0f2fe; color: #075985; font-family: ui-monospace,monospace; }

/* Match card look */
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 10px; }
</style>
""",
    unsafe_allow_html=True,
)

# ── data load ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading fee code database...")
def _load():
    return load_latest(ROOT / "data" / "parsed")


records, embeddings, record_ids, manifest = _load()
available_provinces = sorted({r.province for r in records})
province_counts: dict[str, int] = {p: 0 for p in available_provinces}
ngs_mapped: dict[str, int] = {p: 0 for p in available_provinces}
for r in records:
    province_counts[r.province] = province_counts.get(r.province, 0) + 1
    if r.NGS_code:
        ngs_mapped[r.province] = ngs_mapped.get(r.province, 0) + 1

codes_by_province: dict[str, list[str]] = {p: [] for p in available_provinces}
for r in records:
    codes_by_province[r.province].append(r.fsc_code)
for p in codes_by_province:
    codes_by_province[p].sort()

# ── session defaults (enables swap button) ────────────────────────────────
if "src" not in st.session_state:
    st.session_state["src"] = available_provinces[0]
if "dst" not in st.session_state:
    st.session_state["dst"] = next(
        (p for p in available_provinces if p != st.session_state["src"]),
        available_provinces[0],
    )


def _swap_provinces() -> None:
    st.session_state["src"], st.session_state["dst"] = (
        st.session_state["dst"], st.session_state["src"],
    )
    # Clear picked code on swap — old code likely doesn't exist in new src.
    st.session_state.pop("picked_code", None)


# ── sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚕\u00a0 FSC Lookup")
    st.caption("Physician fee codes, mapped across provinces via CIHI NGS.")
    st.divider()

    st.markdown("<p class='eyebrow'>From / To</p>", unsafe_allow_html=True)
    c_from, c_swap, c_to = st.columns([5, 1, 5])
    with c_from:
        st.selectbox(
            "From", options=available_provinces, key="src",
            label_visibility="collapsed",
        )
    with c_swap:
        st.button(
            "⇄", on_click=_swap_provinces, help="Swap input/output",
            use_container_width=True,
        )
    with c_to:
        dst_options = [p for p in available_provinces if p != st.session_state["src"]]
        if st.session_state["dst"] not in dst_options:
            st.session_state["dst"] = dst_options[0] if dst_options else ""
        st.selectbox(
            "To", options=dst_options, key="dst",
            label_visibility="collapsed",
        )

    src = st.session_state["src"]
    dst = st.session_state["dst"]

    st.markdown("<p class='eyebrow' style='margin-top:12px'>FSC code</p>",
                unsafe_allow_html=True)
    src_codes = codes_by_province.get(src, [])
    picked = st.selectbox(
        "Pick or search",
        options=[""] + src_codes,
        format_func=lambda c: c if c else f"— select a {src} code —",
        key="picked_code",
        label_visibility="collapsed",
    )
    manual = st.text_input(
        "Or type one",
        placeholder="K040, 01712, 0615…",
        help="Typing a code overrides the picker. Case-insensitive.",
        label_visibility="collapsed",
    ).strip().upper()
    fsc_input = manual or picked

    top_n = st.slider("Max matches", 1, 10, 5)

    st.divider()
    st.markdown("<p class='eyebrow'>Dataset</p>", unsafe_allow_html=True)
    embed_dim = embeddings.shape[1] if embeddings.size else None
    embed_model = manifest.models.get("embed", "none")
    st.caption(
        f"**Version:** `{manifest.generated_at[:10]}`  \n"
        f"**Codes:** {len(records):,} ({len(available_provinces)} provinces)  \n"
        f"**Matching:** "
        + (f"semantic ({embed_model}, {embed_dim}-dim)"
           if embed_dim else "Jaccard fallback")
    )

# ── main header ────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='margin-bottom:0'>FSC Cross-Province Lookup</h1>"
    "<p style='color:#6b7280;margin-top:4px;margin-bottom:24px'>"
    "Enter one fee code and find its closest equivalents in the other provinces."
    "</p>",
    unsafe_allow_html=True,
)

# Stats row
stat_cols = st.columns(len(available_provinces) + 1)
stat_cols[0].metric("Total codes", f"{len(records):,}")
for col, prov in zip(stat_cols[1:], available_provinces):
    total = province_counts[prov]
    mapped = ngs_mapped[prov]
    pct = (mapped / total * 100) if total else 0
    col.metric(prov, f"{total:,}", f"{pct:.0f}% NGS-mapped", delta_color="off")

st.divider()

# ── empty state ────────────────────────────────────────────────────────────
if not fsc_input:
    st.markdown(
        "#### Start by picking a code in the sidebar, "
        f"or try one of these **{src}** samples:"
    )
    # Show a handful of sample codes with descriptions
    samples = [r for r in records if r.province == src][:6]
    if samples:
        sample_cols = st.columns(3)
        for i, r in enumerate(samples):
            with sample_cols[i % 3]:
                with st.container(border=True):
                    st.markdown(f"**`{r.fsc_code}`**")
                    if r.fsc_fn:
                        st.caption(r.fsc_fn[:90])
    st.stop()

# ── run search ─────────────────────────────────────────────────────────────
if not dst:
    st.warning("Select an output province in the sidebar.")
    st.stop()

result: LookupResult | None = search(
    fsc_code=fsc_input,
    src=src,
    dst=dst,
    top_n=top_n,
    records=records,
    embeddings=embeddings,
    record_ids=record_ids,
)

if result is None:
    st.error(
        f"Code **`{fsc_input}`** not found in **{src}**. "
        "Double-check the code, or switch the input province."
    )
    st.stop()

anchor = result.anchor

# ── anchor card ────────────────────────────────────────────────────────────
hierarchy = " › ".join(
    p for p in [anchor.fsc_chapter, anchor.fsc_section, anchor.fsc_subsection] if p
)
ngs_badge = (
    f"<span class='badge badge-ngs'>NGS {anchor.NGS_code}</span>"
    if anchor.NGS_code
    else "<span class='badge badge-ngs-no'>no NGS mapping</span>"
)
st.markdown(
    f"""
<div class="anchor-card">
  <p class="eyebrow">Anchor</p>
  <div>
    <span class="badge badge-prov">{anchor.province}</span>
    <span class="anchor-code">&nbsp;{anchor.fsc_code}</span>
    &nbsp;&nbsp;{ngs_badge}
  </div>
  <div class="anchor-desc"><strong>{anchor.fsc_fn or '—'}</strong></div>
  {f'<div class="anchor-desc">{anchor.fsc_description}</div>' if anchor.fsc_description else ''}
  {f'<div style="color:#64748b;font-size:0.85em;margin-top:6px">{hierarchy}</div>' if hierarchy else ''}
</div>
""",
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Price", f"${anchor.price}" if anchor.price else "—")
m2.metric("Page", anchor.page or "—")
m3.metric("Extraction conf.", f"{anchor.extraction_confidence:.2f}")
m4.metric(
    "NGS mapping conf.",
    f"{anchor.NGS_mapping_confidence:.2f}" if anchor.NGS_code else "—",
)

with st.expander("NGS mapping details"):
    st.markdown(
        f"- **Code:** `{anchor.NGS_code or '—'}`\n"
        f"- **Label:** {anchor.NGS_label or '—'}\n"
        f"- **Method:** {anchor.NGS_mapping_method or '—'}\n"
        f"- **Confidence:** {anchor.NGS_mapping_confidence:.2f}"
    )

# ── matches section ────────────────────────────────────────────────────────
st.divider()
header_left, header_right = st.columns([6, 2])
with header_left:
    st.markdown(
        f"### Closest in **{dst}** "
        f"<span class='badge badge-method'>{result.score_method}</span>",
        unsafe_allow_html=True,
    )
with header_right:
    # One-click download — no intermediate button.
    xlsx_bytes = build_workbook([result])
    st.download_button(
        label="⬇  Download Excel",
        data=xlsx_bytes,
        file_name=f"fsc_{anchor.fsc_code}_{anchor.province}_to_{dst}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        type="primary",
    )

if not result.matches:
    st.info(
        f"No matches found in {dst}. This usually means the description "
        "doesn't overlap meaningfully with any {dst} code — try broadening "
        "the anchor or increasing Max matches."
    )
else:
    for rank, mr in enumerate(result.matches, 1):
        fc = mr.fee_code
        ngs_tag = (
            "<span class='badge badge-same'>✓ same NGS</span>"
            if mr.ngs_match
            else "<span class='badge badge-diff'>different NGS</span>"
        )
        ngs_code_tag = (
            f"<span class='badge badge-ngs'>NGS {fc.NGS_code}</span>"
            if fc.NGS_code
            else "<span class='badge badge-ngs-no'>no NGS</span>"
        )
        with st.container(border=True):
            head_l, head_r = st.columns([8, 2])
            with head_l:
                st.markdown(
                    f"<span style='color:#94a3b8;font-weight:600'>#{rank}</span>"
                    f" &nbsp;<span class='badge badge-prov'>{fc.province}</span>"
                    f" &nbsp;<span style='font-family:ui-monospace,monospace;"
                    f"font-size:1.15em;font-weight:600'>{fc.fsc_code}</span>"
                    f" &nbsp;{ngs_code_tag} &nbsp;{ngs_tag}",
                    unsafe_allow_html=True,
                )
            with head_r:
                st.markdown(
                    f"<div style='text-align:right'>"
                    f"<span class='badge badge-sim'>"
                    f"{mr.sim_score * 100:.1f}% match</span></div>",
                    unsafe_allow_html=True,
                )
            st.markdown(f"**{fc.fsc_fn or '—'}**")
            if fc.fsc_description:
                st.caption(fc.fsc_description[:260])

            s1, s2, s3, s4 = st.columns(4)
            s1.caption(f"**Price** · {'$' + str(fc.price) if fc.price else '—'}")
            s2.caption(f"**Page** · {fc.page or '—'}")
            s3.caption(f"**Extract** · {fc.extraction_confidence:.2f}")
            s4.caption(
                f"**NGS map** · "
                f"{fc.NGS_mapping_confidence:.2f}" if fc.NGS_code else "**NGS map** · —"
            )

            with st.expander("Full details"):
                st.markdown(f"**Chapter:** {fc.fsc_chapter or '—'}")
                st.markdown(f"**Section:** {fc.fsc_section or '—'}")
                if fc.fsc_subsection:
                    st.markdown(f"**Subsection:** {fc.fsc_subsection}")
                if fc.fsc_notes:
                    st.markdown(f"**Notes:** {fc.fsc_notes[:300]}")
                st.markdown(
                    f"**NGS:** `{fc.NGS_code or '—'}` "
                    f"{fc.NGS_label or ''}"
                )

# ── footer ─────────────────────────────────────────────────────────────────
st.caption(
    "Same-NGS is shown as a flag only — it is **not** a ranking signal. "
    "Two provinces can legitimately assign equivalent procedures to different NGS codes."
)
