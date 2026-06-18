#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


INCLUDE_TERMS = ("rheumatoid arthritis",)
EXCLUDE_TERMS = (
    "psoriatic arthritis",
    "axial spondyloarthritis",
    "spondyloarthritis",
    "lupus",
    "osteoarthritis",
    "inflammatory bowel disease",
    "crohn",
    "ulcerative colitis",
    "atopic dermatitis",
    "juvenile idiopathic arthritis",
    "juvenile rheumatoid arthritis",
)

RA_FEATURE_PATTERNS = {
    "population_mtx_naive": ("methotrexate-naive", "methotrexate naive", "mtx-naive", "mtx naive"),
    "population_mtx_ir": ("methotrexate inadequate", "mtx-ir", "mtx ir", "inadequate response to methotrexate"),
    "population_dmard_ir": ("dmard inadequate", "dmard-ir", "inadequate response to dmard"),
    "population_biologic_ir": ("biologic inadequate", "biologic-ir", "inadequate response to biologic"),
    "population_tnf_ir": ("tnf inadequate", "tnf-ir", "anti-tnf inadequate", "tnf inhibitor inadequate"),
    "endpoint_acr20": ("acr20", "acr 20"),
    "endpoint_acr50": ("acr50", "acr 50"),
    "endpoint_acr70": ("acr70", "acr 70"),
    "endpoint_das28": ("das28", "das 28"),
    "endpoint_cdai": ("cdai",),
    "endpoint_sdai": ("sdai",),
    "endpoint_haq_di": ("haq-di", "haq di", "health assessment questionnaire-disability index"),
    "background_mtx": ("background methotrexate", "stable methotrexate", "concomitant methotrexate"),
    "rescue_allowed": ("rescue medication", "rescue therapy", "escape therapy"),
}


def text_blob(frame: pd.DataFrame) -> pd.Series:
    columns = [
        "brief_title",
        "condition",
        "brief_summary/textblock",
        "eligibility/criteria/textblock",
        "intervention/description",
        "intervention/intervention_name",
        "keyword",
        "study_design_info/primary_purpose",
    ]
    available = [column for column in columns if column in frame.columns]
    return frame[available].fillna("").astype(str).agg(" ".join, axis=1).str.lower()


def indication_blob(frame: pd.DataFrame) -> pd.Series:
    columns = ["condition", "condition_browse/mesh_term", "brief_title", "keyword"]
    available = [column for column in columns if column in frame.columns]
    return frame[available].fillna("").astype(str).agg(" ".join, axis=1).str.lower()


def load_phase(source: Path, phase: str) -> pd.DataFrame:
    phase_dir = source / phase
    x_parts = []
    for split in ("train", "test"):
        x_path = phase_dir / f"{split}_x.csv"
        y_path = phase_dir / f"{split}_y.csv"
        if not x_path.exists() or not y_path.exists():
            continue
        x = pd.read_csv(x_path)
        y = pd.read_csv(y_path)
        if "" in x.columns:
            x = x.rename(columns={"": "nctid"})
        if "Unnamed: 0" in x.columns:
            x = x.rename(columns={"Unnamed: 0": "nctid"})
        x["trialbench_outcome_label"] = y["outcome"].values if "outcome" in y.columns else pd.NA
        x["source_split"] = split
        x_parts.append(x)
    return pd.concat(x_parts, ignore_index=True) if x_parts else pd.DataFrame()


def build_dataset(source: Path) -> pd.DataFrame:
    full = pd.concat([load_phase(source, phase) for phase in ("Phase1", "Phase2", "Phase3")], ignore_index=True)
    indication = indication_blob(full)
    include = indication.map(lambda value: any(term in value for term in INCLUDE_TERMS))
    exclude = indication.map(lambda value: any(term in value for term in EXCLUDE_TERMS))
    systemic_drug = full.get("Drug intervention Number", pd.Series(0, index=full.index)).fillna(0).astype(float) > 0
    interventional = full.get("study_type", pd.Series("", index=full.index)).fillna("").str.lower().eq("interventional")
    ra = full[include & ~exclude & systemic_drug & interventional].copy()
    ra_blob = text_blob(ra)
    for feature, patterns in RA_FEATURE_PATTERNS.items():
        ra[feature] = ra_blob.map(lambda value, pats=patterns: int(any(pattern in value for pattern in pats)))
    ra["active_comparator_design"] = full.get("Active Comparator Arm Number", 0)
    ra["placebo_design"] = full.get("Placebo Comparator Arm Number", 0)
    ra["combination_therapy"] = ra_blob.map(lambda value: int("combination" in value or "+" in value))
    return ra


def write_label_audit(path: Path) -> None:
    text = """# RA POS Label Audit

The current TrialBench `outcome` label is an approval/outcome forecasting label, not yet proven to mean
prespecified primary endpoint success. Treat the generated RA dataset as a candidate feature table.

For a production RA POS model, convert labels to endpoint success/failure by reviewing pre-readout
evidence and post-readout results from ClinicalTrials.gov results, papers, press releases, congress
abstracts, and regulatory documents. Freeze evidence snapshots before readout dates to prevent leakage.
"""
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an RA-only candidate dataset from TrialBench approval data.")
    parser.add_argument("--source", default="data/trial-approval-forecasting")
    parser.add_argument("--out", default="data/ra_pos/ra_trials.csv")
    args = parser.parse_args()

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(Path(args.source))
    dataset.to_csv(output, index=False)
    write_label_audit(output.parent / "label_audit.md")
    print({"rows": int(len(dataset)), "output": str(output), "label_audit": str(output.parent / "label_audit.md")})


if __name__ == "__main__":
    main()
