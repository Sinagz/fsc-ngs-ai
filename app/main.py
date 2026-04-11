"""
FSC Cross-Province Lookup — Streamlit App

Run:
    streamlit run app/main.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from app.lookup_engine import get_engine, PROVINCES, LookupResult, MatchResult, FeeCode
from app.excel_export import build_workbook

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FSC Cross-Province Lookup",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.fsc-card       { background:#f8f9fa; border-radius:8px; padding:14px 18px; margin-bottom:8px; }
.anchor-card    { background:#fff9e6; border:1.5px solid #f5c518; border-radius:8px; padding:14px 18px; }
.match-card     { background:#f0f7ff; border-radius:8px; padding:10px 14px; margin-bottom:6px; }
.ngs-badge      { background:#e2f0d9; color:#375623; border-radius:4px;
                  padding:2px 8px; font-size:0.82em; font-weight:600; }
.sim-badge      { background:#dce6f1; color:#1f3864; border-radius:4px;
                  padding:2px 8px; font-size:0.82em; font-weight:600; }
.method-badge   { background:#ede7f6; color:#4527a0; border-radius:4px;
                  padding:2px 6px; font-size:0.78em; }
.section-label  { color:#666; font-size:0.82em; text-transform:uppercase;
                  letter-spacing:0.04em; margin-bottom:2px; }
</style>
""", unsafe_allow_html=True)

# ── load engine ───────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading fee code database...")
def load_engine():
    eng = get_engine()
    eng.load()
    return eng

engine = load_engine()

# ── sidebar ───────────────────────────────────────────────────────────────────

provs = engine.provinces()

with st.sidebar:
    st.title("🏥 FSC Lookup")
    st.caption("Cross-province fee code mapping via NGS")
    st.divider()

    input_province = st.selectbox(
        "Input province",
        options=provs,
        index=0,
        help="Province the FSC code belongs to",
    )

    output_province = st.selectbox(
        "Output province",
        options=[p for p in provs if p != input_province],
        index=0,
        help="Province to find equivalent codes in",
    )

    fsc_input = st.text_input(
        "FSC code",
        placeholder="e.g. K040, 01712, 0615",
        help="Enter the fee code from the input province",
    ).strip().upper()

    top_n = st.slider("Max matches", min_value=1, max_value=10, value=5)

    st.divider()

    embed_status = (
        "**Semantic (BGE-large)**" if engine.embedding_available
        else "**Jaccard similarity** _(run `python src/build_embeddings.py` for semantic)_"
    )
    st.caption(f"Similarity method: {embed_status}")

    st.divider()
    st.caption(
        "**How it works**\n\n"
        "1. Select input and output provinces\n"
        "2. Enter an FSC code from the input province\n"
        "3. See full details for that code + the closest matches in the output province\n"
        "4. Export to Excel"
    )

# ── main ──────────────────────────────────────────────────────────────────────

st.title("FSC Cross-Province Lookup")

if not fsc_input:
    st.info("Enter an FSC code in the sidebar to begin.")

    # Show sample codes to help users get started
    with st.expander("Sample codes to try"):
        cols = st.columns(len(provs))
        for col, prov in zip(cols, provs):
            codes = engine.all_codes_for_province(prov)[:15]
            col.markdown(f"**{prov}**")
            col.code("\n".join(codes))
    st.stop()

# ── lookup ────────────────────────────────────────────────────────────────────

result: LookupResult | None = engine.search(
    fsc_input, input_province, output_province, top_n=top_n
)

if result is None:
    st.error(
        f"Code **{fsc_input}** not found in **{input_province}**. "
        "Check the code or try a different input province."
    )
    # Suggest similar codes
    suggestions = engine.fuzzy_search(fsc_input, input_province, limit=8)
    if suggestions:
        st.markdown("**Did you mean one of these?**")
        for s in suggestions:
            st.write(f"- `{s['fsc_code']}` — {s.get('fsc_fn','')[:60]}")
    st.stop()

anchor = result.anchor
method_badge = (
    '<span class="method-badge">semantic</span>'
    if result.score_method == "semantic"
    else '<span class="method-badge">jaccard</span>'
)

# ── two-column layout: INPUT left, OUTPUT right ───────────────────────────────

left, divider_col, right = st.columns([10, 1, 10])

# ─── LEFT: anchor (input province) ───────────────────────────────────────────
with left:
    st.markdown(
        f"### [{anchor.province}] `{anchor.fsc_code}` &nbsp;"
        f"<span class='ngs-badge'>NGS {anchor.NGS_code or 'NOMAP'}</span>",
        unsafe_allow_html=True,
    )

    st.markdown(f"<div class='section-label'>Name / Function</div>", unsafe_allow_html=True)
    st.markdown(f"**{anchor.fsc_fn or '—'}**")

    if anchor.fsc_description:
        st.markdown(f"<div class='section-label'>Description</div>", unsafe_allow_html=True)
        st.write(anchor.fsc_description)

    c1, c2, c3 = st.columns(3)
    c1.metric("Price", f"${anchor.price}" if anchor.price else "—")
    c2.metric("Page", anchor.page or "—")
    c3.metric("Confidence", anchor.fsc_confidence or "—")

    st.markdown(f"<div class='section-label'>Chapter / Section</div>", unsafe_allow_html=True)
    hierarchy = " › ".join(
        p for p in [anchor.fsc_chapter, anchor.fsc_section, anchor.fsc_subsection] if p
    )
    st.caption(hierarchy or "—")

    with st.expander("NGS details"):
        st.markdown(f"**Code:** `{anchor.NGS_code or 'NOMAP'}`")
        st.markdown(f"**Label:** {result.ngs_label or '—'}")
        if result.ngs_description:
            st.write(result.ngs_description)
        if anchor.NGS_rationale:
            st.markdown(f"**Rationale:** {anchor.NGS_rationale}")
        if anchor.NGS_notes:
            st.markdown(f"**Notes:** {anchor.NGS_notes}")

    with st.expander("Full FSC details"):
        cols_d = st.columns(2)
        cols_d[0].markdown(f"**FSC Rationale:** {anchor.fsc_rationale or '—'}")
        cols_d[0].markdown(f"**Key observations:** {anchor.fsc_key_observations or '—'}")
        if anchor.fsc_notes:
            cols_d[1].markdown(f"**Notes:** {anchor.fsc_notes}")
        if anchor.fsc_others:
            cols_d[1].markdown(f"**Others:** {anchor.fsc_others}")

# divider
with divider_col:
    st.markdown("<div style='border-left:2px solid #e0e0e0;height:100%;margin:0 auto;'></div>",
                unsafe_allow_html=True)

# ─── RIGHT: matches (output province) ────────────────────────────────────────
with right:
    st.markdown(
        f"### Closest in **{output_province}** &nbsp; {method_badge}",
        unsafe_allow_html=True,
    )

    if not result.matches:
        st.info(f"No matches found in {output_province}.")
    else:
        for rank, mr in enumerate(result.matches, 1):
            fc = mr.fee_code
            score_pct = f"{mr.sim_score * 100:.1f}%"
            ngs_tag = (
                f"<span class='ngs-badge'>NGS {fc.NGS_code}</span>" if fc.NGS_code else ""
            )
            if mr.ngs_match:
                ngs_status = "<span class='ngs-badge'>same NGS</span>"
            elif fc.NGS_code and fc.NGS_code not in ("NOMAP", ""):
                ngs_status = ("<span style='background:#fce8e8;color:#8b1a1a;"
                              "border-radius:4px;padding:2px 6px;font-size:0.82em;"
                              "font-weight:600'>diff NGS</span>")
            else:
                ngs_status = ""
            score_tag = f"<span class='sim-badge'>{score_pct}</span>"

            with st.container(border=True):
                st.markdown(
                    f"**#{rank} &nbsp; [{fc.province}] `{fc.fsc_code}`** &nbsp; "
                    f"{ngs_tag} {ngs_status} &nbsp; {score_tag}",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**{fc.fsc_fn or '—'}**")
                if fc.fsc_description:
                    st.caption(fc.fsc_description[:220])

                c1, c2, c3 = st.columns(3)
                c1.caption(f"Price: {'$'+fc.price if fc.price else '—'}")
                c2.caption(f"Page: {fc.page or '—'}")
                c3.caption(f"Conf: {fc.fsc_confidence or '—'}")

                with st.expander("Full details"):
                    d1, d2 = st.columns(2)
                    with d1:
                        st.markdown(f"**Chapter:** {fc.fsc_chapter or '—'}")
                        st.markdown(f"**Section:** {fc.fsc_section or '—'}")
                        if fc.fsc_subsection:
                            st.markdown(f"**Subsection:** {fc.fsc_subsection}")
                        if fc.fsc_notes:
                            st.markdown(f"**Notes:** {fc.fsc_notes[:300]}")
                    with d2:
                        st.markdown(f"**NGS Code:** {fc.NGS_code or '—'}")
                        st.markdown(f"**NGS Label:** {fc.NGS_label or '—'}")
                        if fc.NGS_rationale:
                            st.markdown(f"**NGS Rationale:** {fc.NGS_rationale[:200]}")
                        if fc.NGS_notes:
                            st.markdown(f"**NGS Notes:** {fc.NGS_notes[:200]}")

# ── export ────────────────────────────────────────────────────────────────────

st.divider()
ecol1, ecol2 = st.columns([2, 4])
with ecol1:
    if st.button("Generate Excel", type="primary"):
        xlsx_bytes = build_workbook([result])
        st.download_button(
            label="Download Excel",
            data=xlsx_bytes,
            file_name=f"fsc_{anchor.fsc_code}_{anchor.province}_to_{result.output_province}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
with ecol2:
    st.caption(
        f"Excel: anchor `{anchor.fsc_code}` ({anchor.province}) + "
        f"top {top_n} matches in {output_province}, all FSC and NGS fields included."
    )
