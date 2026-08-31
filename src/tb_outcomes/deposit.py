"""Montagem do depósito de dados com DOI (ESPECIFICACAO §8, passo 7).

Separa o material em **dois registros**, porque eles têm regimes de acesso diferentes e
misturá-los obrigaria o mais restritivo a valer para tudo:

- `open/`      — resultados e artefatos de auditoria. Nenhuma linha por paciente. É o que
                 o DOI citado no manuscrito aponta; entra sob embargo e abre na publicação.
- `restricted/` — predições out-of-fold e valores SHAP: uma linha por paciente. Sem
                 identificador direto, mas `record_pos` é a posição na coorte, então quem
                 tiver o SINAN reconstrói o vínculo. Pseudonimizado, não anônimo. O regime
                 de acesso é decisão do PI, não default deste código.

O microdado harmonizado (`harmonized.parquet`, os `sinnan*.parquet`) **não entra em nenhum
dos dois**: a fonte é pública e o caminho correto é reproduzir a extração pelo bot.

A montagem é por CÓPIA declarada em manifesto, com hash de cada arquivo — um revisor
consegue verificar que o que baixou é o que foi depositado.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import date
from pathlib import Path

import pandas as pd

# Quem é quem no depósito. Cada entrada: (destino, origem, obrigatório?)
# Os caminhos de origem são relativos à raiz do projeto.
OPEN_LAYOUT: list[tuple[str, str, bool]] = [
    # --- leaderboard principal (k=50) ---
    ("results/main/benchmark_metrics.csv", "data/benchmark_metrics.csv", True),
    ("results/main/calibration_manifest.json", "data/calibration_manifest.json", True),
    # --- robustez à granularidade espacial (§1.3) ---
    ("results/sweep/robustness_table.csv", "data/sweep/robustness_table.csv", True),
    ("results/sweep/rank_stability.csv", "data/sweep/rank_stability.csv", True),
    ("results/sweep/k27_benchmark_metrics.csv", "data/sweep/k27/benchmark_metrics.csv", True),
    ("results/sweep/k75_benchmark_metrics.csv", "data/sweep/k75/benchmark_metrics.csv", True),
    # --- ablação territorial (§3.3) ---
    ("results/ablation/ablation_leaderboard.csv", "data/ablation/ablation_leaderboard.csv", True),
    ("results/ablation/ablation_completeness.csv", "data/ablation/ablation_completeness.csv", True),
    ("results/ablation/individual_benchmark_metrics.csv",
     "data/ablation/individual/benchmark_metrics.csv", True),
    ("results/ablation/municipal_benchmark_metrics.csv",
     "data/ablation/municipal/benchmark_metrics.csv", True),
    # --- holdout temporal (§4.1) ---
    ("results/temporal/temporal_vs_spatial.csv", "data/temporal/temporal_vs_spatial.csv", True),
    ("results/temporal/benchmark_metrics.csv", "data/temporal/benchmark_metrics.csv", True),
    # --- equidade e subgrupos (§3.2) ---
    ("results/equity/equity_table.csv", "data/equity_table.csv", True),
    ("results/equity/equity_disparity.csv", "data/equity_disparity.csv", True),
    ("results/equity/equity_discrimination_by_group.csv",
     "data/equity_discrimination_by_group.csv", True),
    ("results/equity/equity_schooling_confounding.csv",
     "data/equity_schooling_confounding.csv", False),
    ("results/equity/equity_indigena_investigation.csv",
     "data/equity_indigena_investigation.csv", False),
    ("results/equity/equity_indigena_age_mortality.csv",
     "data/equity_indigena_age_mortality.csv", False),
    # --- SHAP (§4.2): resumo e manifesto; os valores por paciente vão ao restrito ---
    ("results/shap/shap_summary.csv", "data/shap_summary.csv", True),
    ("results/shap/shap_stability.csv", "data/shap_stability.csv", True),
    ("results/shap/shap_manifest.json", "data/shap_manifest.json", True),
    # --- utilidade programática (§3.8) ---
    ("results/utility/utility_topk.csv", "data/utility_topk.csv", True),
    ("results/utility/utility_decision_curve.csv", "data/utility_decision_curve.csv", True),
    ("results/utility/utility_summary.csv", "data/utility_summary.csv", True),
    ("results/utility/clinical_baseline_benchmark_metrics.csv",
     "data/clinical_baseline/benchmark_metrics.csv", True),
    # --- definição das dobras e da coorte (§6) ---
    ("protocol/cluster_summary.csv", "data/cluster_summary.csv", True),
    ("protocol/fold_summary.csv", "data/fold_summary.csv", True),
    ("protocol/municipality_clusters_k27.parquet", "data/municipality_clusters_k27.parquet", True),
    ("protocol/municipality_clusters_k50.parquet", "data/municipality_clusters_k50.parquet", True),
    ("protocol/municipality_clusters_k75.parquet", "data/municipality_clusters_k75.parquet", True),
    ("protocol/cohort_flow.csv", "data/cohort_flow_sensitive_tb_3class.csv", True),
    ("protocol/cohort_flow.json", "data/cohort_flow_sensitive_tb_3class.json", True),
    ("protocol/exclusion_reasons.csv", "data/exclusion_reasons_sensitive_tb_3class.csv", True),
    ("protocol/feature_dictionary.csv", "data/feature_dictionary.csv", True),
    ("protocol/feature_availability.csv", "data/feature_availability.csv", True),
    ("protocol/imbalance_audit.json", "data/imbalance_audit.json", True),
    ("protocol/leakage_checks.json", "data/leakage_checks.json", True),
    ("protocol/model_capabilities.csv", "data/model_capabilities.csv", True),
    # --- reprodutibilidade (§6, R3 #2) ---
    ("compute/compute_cost_by_pair.csv", "data/compute_cost_by_pair.csv", True),
    ("compute/compute_cost_by_model.csv", "data/compute_cost_by_model.csv", True),
    ("compute/compute_manifest.json", "data/compute_manifest.json", True),
    ("compute/poetry.lock", "poetry.lock", True),
    ("compute/pyproject.toml", "pyproject.toml", True),
]

# Diretórios copiados inteiros: (destino, origem, padrão, obrigatório?)
OPEN_TREES: list[tuple[str, str, str, bool]] = [
    ("figures", "artifacts/figures", "*.png", True),
    ("figures/data", "artifacts/figures/data", "*.csv", True),
    ("protocol/configs", "configs", "*.yaml", True),
]

# Material por paciente. Decisão do PI (2026-08-31): depositar SÓ o OOF do modelo final,
# não os 70 pares do benchmark. O par campeão é o que sustenta equidade, utilidade,
# calibração e SHAP — é o que um revisor precisa para verificar no nível do indivíduo. Os
# demais pares são reproduzíveis pelo código, e as métricas deles já estão no registro
# aberto. Efeito: 1.667 MB -> ~31 MB.
RESTRICTED_LAYOUT: list[tuple[str, str, bool]] = [
    ("shap/shap_values.parquet", "data/shap_values.parquet", True),
    ("folds/outer_folds.parquet", "data/outer_folds.parquet", True),
]


def final_model_pair(root: Path) -> tuple[str, str]:
    """Modelo final = topo do leaderboard k=50, lido do artefato, não fixado no código."""
    from tb_outcomes.robustness import leaderboard

    lb = leaderboard(pd.read_csv(root / "data" / "benchmark_metrics.csv"))
    topo = lb.sort_values("f1_macro", ascending=False).iloc[0]
    return str(topo["model"]), str(topo["best_strategy"])


def extract_final_oof(root: Path, destino: Path) -> dict | None:
    """Recorta o OOF do par campeão do arquivo de 70 pares."""
    origem = root / "data" / "oof_predictions.parquet"
    # sem o OOF ou sem o leaderboard não há como saber QUAL par é o final; ausência é
    # reportada como artefato faltando, não improvisada.
    if not origem.exists() or not (root / "data" / "benchmark_metrics.csv").exists():
        return None
    model, strategy = final_model_pair(root)
    d = pd.read_parquet(origem, filters=[("model", "==", model),
                                         ("strategy", "==", strategy)])
    if d.empty:
        raise ValueError(f"o OOF não contém o par final {model} × {strategy}")
    destino.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(destino, index=False)
    return {"model": model, "strategy": strategy, "linhas": int(len(d))}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for bloco in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def _copiar(root: Path, destino_base: Path, layout, trees=()) -> tuple[list[dict], list[str]]:
    registros, faltando = [], []
    for destino, origem, obrigatorio in layout:
        src = root / origem
        if not src.exists():
            if obrigatorio:
                faltando.append(origem)
            continue
        dst = destino_base / destino
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        registros.append({"arquivo": destino, "origem": origem,
                          "bytes": dst.stat().st_size, "sha256": _sha256(dst)})
    for destino, origem, padrao, obrigatorio in trees:
        srcdir = root / origem
        if not srcdir.is_dir():
            if obrigatorio:
                faltando.append(origem + "/")
            continue
        for src in sorted(srcdir.glob(padrao)):
            dst = destino_base / destino / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            registros.append({"arquivo": f"{destino}/{src.name}",
                              "origem": str(src.relative_to(root)),
                              "bytes": dst.stat().st_size, "sha256": _sha256(dst)})
    return registros, faltando


_TEMPLATES = Path(__file__).parent / "deposit_templates"


def build(root: Path, out: Path, embargo_date: str, git_state: dict,
          include_restricted: bool = True) -> dict:
    """Monta a árvore do depósito e devolve o resumo da operação."""
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)

    aberto, faltando_aberto = _copiar(root, out / "open", OPEN_LAYOUT, OPEN_TREES)
    restrito, faltando_restrito = ([], [])
    par_final = None
    if include_restricted:
        restrito, faltando_restrito = _copiar(root, out / "restricted", RESTRICTED_LAYOUT)
        # o OOF do modelo final é RECORTADO, não copiado: o arquivo de origem tem os 70
        # pares do benchmark e só o par campeão é depositado.
        alvo = out / "restricted" / "oof" / "final_model.parquet"
        par_final = extract_final_oof(root, alvo)
        if par_final is None:
            faltando_restrito.append("data/oof_predictions.parquet")
        else:
            restrito.append({"arquivo": "oof/final_model.parquet",
                             "origem": (f"data/oof_predictions.parquet "
                                        f"[{par_final['model']} × {par_final['strategy']}]"),
                             "bytes": alvo.stat().st_size, "sha256": _sha256(alvo)})

    for nome, registros in (("open", aberto), ("restricted", restrito)):
        if not registros:
            continue
        base = out / nome
        # o README do registro é template versionado: `build` apaga e recria a árvore,
        # então um arquivo escrito à mão aqui dentro se perderia na próxima montagem.
        modelo = _TEMPLATES / f"README_{nome}.md"
        if modelo.exists():
            shutil.copy2(modelo, base / "README.md")
        pd.DataFrame(registros).to_csv(base / "MANIFEST.csv", index=False)

    resumo = {
        "gerado_em": date.today().isoformat(),
        "git": git_state,
        "embargo_ate": embargo_date,
        "open": {"n_arquivos": len(aberto),
                 "bytes": sum(r["bytes"] for r in aberto),
                 "faltando": faltando_aberto},
        "restricted": {"n_arquivos": len(restrito),
                       "bytes": sum(r["bytes"] for r in restrito),
                       "faltando": faltando_restrito,
                       "modelo_final": par_final,
                       "escopo": ("apenas o OOF do modelo final; os demais pares do "
                                  "benchmark são reproduzíveis pelo código e suas "
                                  "métricas estão no registro aberto")},
    }
    with (out / "DEPOSIT_SUMMARY.json").open("w", encoding="utf-8") as fh:
        json.dump(resumo, fh, ensure_ascii=False, indent=2)
    return resumo


def zenodo_metadata(embargo_date: str, version: str, github_url: str,
                    creators: list[str], restricted: bool = False) -> dict:
    """Metadados no formato `.zenodo.json`.

    `access_right: embargoed` com `embargo_date` faz o registro receber o DOI imediatamente
    — citável no manuscrito — e abrir sozinho na data. O registro restrito usa
    `access_right: restricted`, em que o download depende de aprovação do depositante.

    A licença dos DADOS é CC-BY-4.0, não a MIT do código: MIT é licença de software e não
    se aplica bem a conjuntos de dados.

    O auxílio do CNPq NÃO entra em `grants`: o vocabulário do Zenodo é o do OpenAIRE e não
    cobre o CNPq, então um identificador inventado faria o envio falhar. Vai na descrição.
    """
    titulo = ("Tuberculosis treatment closure outcomes in Brazil: "
              + ("restricted patient-level predictions"
                 if restricted else "reanalysis results and audit artefacts"))
    if restricted:
        descricao = (
            "<p>Patient-level out-of-fold predictions, SHAP values and fold assignments "
            "from the nationwide benchmark of tuberculosis treatment closure outcomes.</p>"
            "<p><strong>Access is restricted.</strong> Records carry no direct identifiers, "
            "but each row corresponds to one notification and the row index maps back to the "
            "analytic cohort, so re-identification is possible for a holder of the source "
            "SINAN extract. Requests are reviewed by the depositor.</p>"
            "<p>Aggregated results are in the companion open record; analysis code is on "
            f"GitHub at {github_url}.</p>")
    else:
        descricao = (
            "<p>Results and audit artefacts of a reanalysis of tuberculosis treatment "
            "closure outcomes in Brazil, using nested spatially blocked validation with a "
            "temporal holdout.</p>"
            "<p>Contents: leaderboard metrics for the model grid; robustness of the ranking "
            "to spatial granularity; territorial ablation; temporal holdout; equity and "
            "subgroup analysis; SHAP summaries; programmatic utility including decision "
            "curves; the protocol configuration, fold and cluster definitions, cohort flow "
            "and leakage checks; computational cost; and every published figure with the "
            "numerical file that produced it.</p>"
            "<p>No patient-level data is included in this record. Analysis code is on GitHub "
            f"at {github_url}. Funded by CNPq, grant 445458/2023-2.</p>")
    return {
        "title": titulo,
        "description": descricao,
        "upload_type": "dataset",
        "access_right": "restricted" if restricted else "embargoed",
        **({} if restricted else {"embargo_date": embargo_date}),
        **({"access_conditions": (
            "Requests are reviewed by the depositor. Access is granted for research "
            "purposes on terms compatible with Brazilian data protection law.")}
           if restricted else {}),
        "license": "cc-by-4.0",
        "language": "eng",
        "version": version,
        "creators": [{"name": n} for n in creators],
        "keywords": ["tuberculosis", "Brazil", "SINAN", "treatment outcome",
                     "machine learning", "spatial cross-validation", "model calibration",
                     "algorithmic equity", "TRIPOD+AI"],
        "related_identifiers": [
            {"identifier": github_url, "relation": "isSupplementTo", "scheme": "url"},
        ],
        "notes": ("Funding: CNPq grant 445458/2023-2. The CNPq is not covered by the "
                  "OpenAIRE grant vocabulary, so the award is recorded here rather than "
                  "in the structured grants field."),
    }
