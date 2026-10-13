"""Freeze the 2010/2000 IPEA census context for the whole 2015+ cohort."""
import json
from pathlib import Path

import pandas as pd

from tb_outcomes.context_features import assert_pre_study_census
from tb_outcomes.run_identity import file_sha256


def main():
    source = Path("data/indicadores_IPEA.csv")
    df = pd.read_csv(source, low_memory=False)
    baseline = df.loc[df.ANO_DIAG.eq(2014)].drop(columns="ANO_DIAG").copy()
    if baseline.empty or baseline.ID_MUNICIP.duplicated().any():
        raise ValueError("Invalid pre-study municipal lookup")
    values = [c for c in baseline if c.startswith("IPEA_") and not c.endswith("_ANO_CENSO")]
    assert_pre_study_census(baseline, values)
    out = Path("data/correction_audit/baseline_context.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    baseline.to_parquet(out, index=False)
    manifest = {
        "decision": "PI approved pre-study census context for primary; contemporary context sensitivity",
        "lookup_year": 2014, "allowed_census_years": [2000, 2010],
        "n_municipalities": len(baseline), "value_columns": values,
        "source": str(source), "source_sha256": file_sha256(source),
        "output_sha256": file_sha256(out), "script_sha256": file_sha256(Path(__file__)),
        "availability_evidence": "Atlas Brasil 2013 launched 29 July 2013, before the 2015 cohort",
        "source_url": "https://brasil.un.org/pt-br/63320-plataforma-online-do-atlas-brasil-2013-facilita-compreens%C3%A3o-do-desenvolvimento-humano-no-pa%C3%ADs",
        "limitations": ["Current extract of historical census series, not an archived 2013 database vintage",
                        "No backfill for municipalities without a pre-study observation",
                        "Time-varying IBGE/CNES indicators excluded from primary absent release metadata"],
    }
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
