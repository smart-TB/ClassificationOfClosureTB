"""Aggregate-only audit before excluding unknown/non-target outcomes. No training."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from tb_outcomes.config import load_config
from tb_outcomes.maturation import assess_maturation


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/analysis_decisions.yaml")
    out = root / "data/correction_audit/maturation"
    out.mkdir(parents=True, exist_ok=True)
    source = root / "data/harmonized.parquet"
    df = pd.read_parquet(source)
    year = pd.to_numeric(df.NU_ANO, errors="coerce")
    declared = year.between(cfg.study.cohort_year_min, cfg.study.cohort_year_max)
    postmortem = pd.to_numeric(df.TRATAMENTO, errors="coerce").eq(6)
    d = df.loc[declared & ~postmortem].copy()
    notified = pd.to_datetime(d.DT_NOTIFIC, errors="coerce")
    start = pd.to_datetime(d.DT_INIC_TR, errors="coerce")
    close = pd.to_datetime(d.DT_ENCERRA, errors="coerce")
    code = pd.to_numeric(d.SITUA_ENCE, errors="coerce")
    d["unresolved"] = code.isna() | code.eq(0)
    d["not_primary_target"] = ~code.isin([1, 2, 3, 10]) & ~d.unresolved
    d["closure_date_missing"] = close.isna()
    d["start_missing"] = start.isna()
    d["start_after_notification"] = start.gt(notified)
    d["closure_before_notification"] = close.lt(notified)
    d["notification_year_mismatch"] = notified.dt.year.ne(pd.to_numeric(d.NU_ANO))
    d["maturation_group"] = assess_maturation(d, cfg.study.data_extraction_date).maturation_group
    audit_frames = []
    totals = {}
    for rule in ["calendar_by_form", "days_by_form", "legacy_days"]:
        a = assess_maturation(d, cfg.study.data_extraction_date, rule=rule)
        d["mature"] = a.mature
        d["unresolved_mature"] = d.unresolved & a.mature
        d["late_recorded_closure"] = close.gt(a.deadline)
        d["resolved_with_date"] = ~d.unresolved & close.notna()
        d["target_mature"] = code.isin([1, 2, 3, 10]) & a.mature
        aggregations = {col: (col, "sum") for col in [
            "mature", "unresolved", "unresolved_mature", "not_primary_target",
            "closure_date_missing", "start_missing", "start_after_notification",
            "closure_before_notification", "notification_year_mismatch",
            "late_recorded_closure", "resolved_with_date", "target_mature"]}
        table = d.groupby(["NU_ANO", "REGIAO", "maturation_group"], dropna=False).agg(
            n=("mature", "size"), **aggregations).reset_index()
        table["rule"] = rule
        audit_frames.append(table)
        totals[rule] = {"mature": int(a.mature.sum()),
                        "unresolved_mature": int(d.unresolved_mature.sum()),
                        "target_mature": int(d.target_mature.sum())}
    pd.concat(audit_frames, ignore_index=True).to_csv(out / "by_year_region_form.csv", index=False)
    delays = (close - notified).dt.days
    closure = pd.DataFrame({"year": d.NU_ANO, "group": d.maturation_group,
                            "outcome_code": code, "closure_delay_days": delays})
    closure.groupby(["year", "group", "outcome_code"], dropna=False).closure_delay_days.agg(
        ["count", "median", "min", "max"]).to_csv(out / "closure_delay.csv")
    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "extraction_date": cfg.study.data_extraction_date,
        "extraction_date_evidence": "PI declaration; file hash identifies supplied snapshot, not download date",
        "raw_source": "data/sinnan.parquet", "raw_sha256": sha256(root / "data/sinnan.parquet"),
        "harmonized_sha256": sha256(source), "script_sha256": sha256(Path(__file__)),
        "maturation_code_sha256": sha256(root / "src/tb_outcomes/maturation.py"),
        "retrieved": len(df), "declared": int(declared.sum()),
        "declared_postmortem": int((declared & postmortem).sum()),
        "audited_before_other_eligibility_and_outcome_filter": len(d),
        "totals": totals,
        "date_flags": {"missing_start": int(start.isna().sum()),
                       "start_after_notification": int(start.gt(notified).sum()),
                       "start_after_extraction": int(start.gt(pd.Timestamp(cfg.study.data_extraction_date)).sum()),
                       "closure_before_notification": int(close.lt(notified).sum())},
        "limitations": ["Mature means administratively observable, not complete.",
                        "Clinical form is a regimen proxy; uncertain forms use 15 months.",
                        "Late notification can explain closure before notification.",
                        "Maturation and the predictive outcome horizon are distinct."],
    }
    (out / "manifest.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
