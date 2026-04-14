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

st.set_page_config(
    page_title="FSC Cross-Province Lookup",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.section-label { color:#666; font-size:0.82em; text-transform:uppercase;
                 letter-spacing:0.04em; margin-bottom:2px; }
.ngs-badge     { background:#e2f0d9; color:#375623; border-radius:4px;
                 padding:2px 8px; font-size:0.82em; font-weight:600; }
.sim-badge     { background:#dce6f1; color:#1f3864; border-radius:4px;
                 padding:2px 8px; font-size:0.82em; font-weight:600; }
.method-badge  { background:#ede7f6; color:#4527a0; border-radius:4px;
                 padding:2px 6px; font-size:0.78em; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading fee code database...")
def _load():
    return load_latest(ROOT / "data" / "parsed")


records, embeddings, record_ids, manifest = _load()
available_provinces = sorted({r.province for r in records})

# ── sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏥 FSC Lookup")
    st.caption("Cross-province fee code mapping via NGS")
    st.divider()

    src = st.selectbox("Input province", options=available_provinces, index=0)
    dst = st.selectbox(
        "Output province",
        options=[p for p in available_provinces if p != src],
        index=0,
    )
    fsc_input = (
        st.text_input(
            "FSC code",
            placeholder="e.g. K040, 01712, 0615",
            help="Enter the fee code from the input province",
        )
        .strip()
        .upper()
    )
    top_n = st.slider("Max matches", min_value=1, max_value=10, value=5)
    st.divider()
    embed_dim = embeddings.shape[1] if embeddings.size else None
    st.caption(
        f"Embeddings: **{manifest.models.get('embed', 'none')}**"
        + (f" ({embed_dim}-dim)" if embed_dim else " (Jaccard fallback)")
    )
    st.caption(f"Data version: **{manifest.generated_at[:10]}**")

# ── main ───────────────────────────────────────────────────────────────────
st.title("FSC Cross-Province Lookup")

if not fsc_input:
    st.info("Enter an FSC code in the sidebar to begin.")
    with st.expander("Sample codes to try"):
        cols = st.columns(len(available_provinces))
        for col, prov in zip(cols, available_provinces):
            codes = sorted({r.fsc_code for r in records if r.province == prov})[:15]
            col.markdown(f"**{prov}**")
            col.code("\n".join(codes))
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
        f"Code **{fsc_input}** not found in **{src}**. "
        "Check the code or try a different input province."
    )
    st.stop()

anchor = result.anchor
left, _, right = st.columns([10, 1, 10])

with left:
    st.markdown(
        f"### [{anchor.province}] `{anchor.fsc_code}` &nbsp;"
        f"<span class='ngs-badge'>NGS {anchor.NGS_code or 'NOMAP'}</span>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='section-label'>Name / Function</div>", unsafe_allow_html=True)
    st.markdown(f"**{anchor.fsc_fn or '—'}**")
    if anchor.fsc_description:
        st.markdown("<div class='section-label'>Description</div>", unsafe_allow_html=True)
        st.write(anchor.fsc_description)

    c1, c2, c3 = st.columns(3)
    c1.metric("Price", f"${anchor.price}" if anchor.price else "—")
    c2.metric("Page", anchor.page or "—")
    c3.metric("Confidence", f"{anchor.extraction_confidence:.2f}")

    st.markdown("<div class='section-label'>Chapter / Section</div>", unsafe_allow_html=True)
    hierarchy = " › ".join(
        p for p in [anchor.fsc_chapter, anchor.fsc_section, anchor.fsc_subsection] if p
    )
    st.caption(hierarchy or "—")

    with st.expander("NGS details"):
        st.markdown(f"**Code:** `{anchor.NGS_code or 'NOMAP'}`")
        st.markdown(f"**Label:** {anchor.NGS_label or '—'}")
        st.markdown(f"**Mapping method:** {anchor.NGS_mapping_method or '—'}")
        st.markdown(
            f"**Mapping confidence:** {anchor.NGS_mapping_confidence:.2f}"
        )

with right:
    st.markdown(
        f"### Closest in **{dst}** &nbsp; "
        f"<span class='method-badge'>{result.score_method}</span>",
        unsafe_allow_html=True,
    )
    if not result.matches:
        st.info(f"No matches found in {dst}.")
    else:
        for rank, mr in enumerate(result.matches, 1):
            fc = mr.fee_code
            with st.container(border=True):
                ngs_status = (
                    "<span class='ngs-badge'>same NGS</span>"
                    if mr.ngs_match
                    else "<span style='background:#fce8e8;color:#8b1a1a;"
                    "border-radius:4px;padding:2px 6px;font-size:0.82em;"
                    "font-weight:600'>diff NGS</span>"
                )
                st.markdown(
                    f"**#{rank} &nbsp; [{fc.province}] `{fc.fsc_code}`** &nbsp; "
                    f"<span class='ngs-badge'>NGS {fc.NGS_code or '—'}</span> "
                    f"{ngs_status} &nbsp; "
                    f"<span class='sim-badge'>{mr.sim_score * 100:.1f}%</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**{fc.fsc_fn or '—'}**")
                if fc.fsc_description:
                    st.caption(fc.fsc_description[:220])

                c1, c2, c3 = st.columns(3)
                c1.caption(f"Price: {'$' + str(fc.price) if fc.price else '—'}")
                c2.caption(f"Page: {fc.page or '—'}")
                c3.caption(f"Conf: {fc.extraction_confidence:.2f}")

                with st.expander("Full details"):
                    st.markdown(f"**Chapter:** {fc.fsc_chapter or '—'}")
                    st.markdown(f"**Section:** {fc.fsc_section or '—'}")
                    if fc.fsc_subsection:
                        st.markdown(f"**Subsection:** {fc.fsc_subsection}")
                    if fc.fsc_notes:
                        st.markdown(f"**Notes:** {fc.fsc_notes[:300]}")
                    st.markdown(f"**NGS:** `{fc.NGS_code or '—'}` {fc.NGS_label or ''}")

st.divider()
ecol1, ecol2 = st.columns([2, 4])
with ecol1:
    if st.button("Generate Excel", type="primary"):
        xlsx_bytes = build_workbook([result])
        st.download_button(
            label="Download Excel",
            data=xlsx_bytes,
            file_name=f"fsc_{anchor.fsc_code}_{anchor.province}_to_{dst}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
with ecol2:
    st.caption(
        f"Excel: anchor `{anchor.fsc_code}` ({anchor.province}) + "
        f"top {top_n} matches in {dst}, all FSC and NGS fields included."
    )
