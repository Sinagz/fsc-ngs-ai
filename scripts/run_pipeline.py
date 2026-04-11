"""
One-command pipeline runner.
Runs all data processing steps in order to produce the app's database.

Usage:
    python scripts/run_pipeline.py [--force]

Flags:
    --force   Re-run all steps even if output files exist
"""
import sys
import subprocess
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

STEPS = [
    {
        "name":   "Extract PDFs to layout JSON",
        "script": "src/extract_pdfs.py",
        "output": DATA / "extracted" / "raw_layout" / "on_layout.json",
    },
    {
        "name":   "Parse fee codes from all provinces",
        "script": "src/parse_all_provinces.py",
        "output": DATA / "parsed" / "fee_codes" / "all_fee_codes.json",
    },
    {
        "name":   "Parse NGS reference DOCX files",
        "script": "src/parse_docx_full.py",
        "output": DATA / "parsed" / "ngs" / "ngs_categories.json",
    },
    {
        "name":   "Map FSC codes to NGS categories",
        "script": "src/map_fsc_ngs.py",
        "output": DATA / "parsed" / "fsc_ngs_mapped.json",
    },
    {
        "name":   "Cross-province grouping (top_key assignment)",
        "script": "src/cross_province.py",
        "output": DATA / "parsed" / "fsc_ngs_cross_province.json",
    },
    {
        "name":   "Build semantic embeddings (BGE-large-en-v1.5, GPU)",
        "script": "src/build_embeddings.py",
        "output": DATA / "parsed" / "embeddings.npz",
    },
]


def run_step(script: str, force: bool) -> bool:
    step = next(s for s in STEPS if s["script"] == script)
    if not force and step["output"].exists():
        print(f"  [SKIP] {step['name']} — output exists ({step['output'].name})")
        return True
    print(f"  [RUN]  {step['name']}...")
    result = subprocess.run(
        [sys.executable, str(ROOT / script)],
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        print(f"  [FAIL] {step['name']} exited with code {result.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Run the FSC-NGS pipeline")
    parser.add_argument("--force", action="store_true",
                        help="Re-run all steps even if outputs exist")
    args = parser.parse_args()

    print("FSC-NGS Data Pipeline")
    print("=" * 40)

    for step in STEPS:
        ok = run_step(step["script"], args.force)
        if not ok:
            print(f"\nPipeline stopped at: {step['name']}")
            sys.exit(1)

    print("\nAll steps complete.")
    print(f"App database:  {DATA / 'parsed' / 'fsc_ngs_mapped.json'}")
    print(f"Embeddings:    {DATA / 'parsed' / 'embeddings.npz'}")
    print("\nTo launch the app:")
    print("  streamlit run app/main.py")


if __name__ == "__main__":
    main()
