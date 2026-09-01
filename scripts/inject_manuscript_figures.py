"""Embute as figuras no corpo do manuscrito, ou as remove.

A PLOS Digital Health pede as figuras EMBUTIDAS na submissão inicial ("Submit embedded in
initial submission; if accepted, provide individual files"). O guia de formatação, escrito
para a fase de produção, pede o contrário. Os dois momentos são diferentes, então o script
faz os dois sentidos:

    python scripts/inject_manuscript_figures.py            # embute (padrão)
    python scripts/inject_manuscript_figures.py --strip    # volta a só legenda

Cada figura vira um `figure` com `\\includegraphics` e a legenda no formato da PLOS. O
mapeamento entre o número no texto e o arquivo em `artifacts/figures/` fica aqui, em um
lugar só, para não haver figura trocada.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

TEX = Path("manuscrito/PLOS_DigitalHealth/manuscript.tex")
FIGS = Path("artifacts/figures")

# número no texto -> arquivo da figura
# número no texto -> (arquivo, altura máxima como fração de \textheight)
# A altura padrão serve às figuras largas. As altas, como o painel de SHAP empilhado,
# precisam de mais: com o teto baixo é a ALTURA que aperta primeiro, e a figura encolhe
# sem chegar a preencher a largura do texto.
MAPA = {
    1: ("15_cohort_funnel", 0.42),
    2: ("01_ranking_f1", 0.42),
    3: ("07_reliability_curves", 0.42),
    4: ("25_leaderboard_across_k", 0.42),
    5: ("29_ablation_arms", 0.42),
    6: ("27_shap_by_class", 0.80),
    7: ("30_equity_by_group", 0.42),
    8: ("28_decision_curves", 0.42),
}

INI = "% <<FIGURA:{n}>>"
FIM = "% <<FIMFIGURA:{n}>>"


def _legenda(tex: str, n: int) -> tuple[str, str] | None:
    """Extrai título e corpo da legenda `\\paragraph*{Fig n. Título} corpo`."""
    m = re.search(rf"\\paragraph\*\{{Fig {n}\. ([^}}]+)\}}\n(.*?)(?=\n\n)", tex, re.S)
    if not m:
        return None
    return m.group(1).strip(), " ".join(m.group(2).split())


def embutir(tex: str) -> tuple[str, list[int]]:
    feitas = []
    for n, (arquivo, altura) in MAPA.items():
        caminho = FIGS / f"{arquivo}.png"
        if not caminho.exists():
            continue
        leg = _legenda(tex, n)
        if leg is None:
            continue
        titulo, corpo = leg
        bloco = (
            f"{INI.format(n=n)}\n"
            r"\begin{figure}[!ht]" "\n"
            r"\centering" "\n"
            rf"\includegraphics[width=\linewidth,height={altura}\textheight,"
            rf"keepaspectratio]"
            rf"{{../../{caminho.as_posix()}}}" "\n"
            r"\caption{{\bf " + titulo + r"} " + corpo + r"}" "\n"
            r"\end{figure}" "\n"
            f"{FIM.format(n=n)}"
        )
        # substitui o parágrafo-legenda pelo bloco completo
        alvo = re.compile(rf"\\paragraph\*\{{Fig {n}\. [^}}]+\}}\n.*?(?=\n\n)", re.S)
        tex = alvo.sub(lambda _: bloco, tex, count=1)
        feitas.append(n)
    return tex, feitas


def remover(tex: str) -> tuple[str, list[int]]:
    """Volta ao formato só-legenda exigido na fase de produção."""
    feitas = []
    for n in MAPA:
        bloco = re.compile(rf"{re.escape(INI.format(n=n))}.*?{re.escape(FIM.format(n=n))}",
                           re.S)
        m = bloco.search(tex)
        if not m:
            continue
        cap = re.search(r"\\caption\{\{\\bf ([^}]+)\} (.*?)\}\n\\end\{figure\}", m.group(0),
                        re.S)
        if not cap:
            continue
        titulo, corpo = cap.group(1), " ".join(cap.group(2).split())
        tex = bloco.sub(f"\\\\paragraph*{{Fig {n}. {titulo}}}\n{corpo}", tex, count=1)
        feitas.append(n)
    return tex, feitas


def main() -> None:
    tex = TEX.read_text(encoding="utf-8")
    if "--strip" in sys.argv:
        tex, feitas = remover(tex)
        acao = "removidas (só legenda)"
    else:
        # idempotente: desfaz antes de refazer
        tex, _ = remover(tex)
        tex, feitas = embutir(tex)
        acao = "embutidas"
    TEX.write_text(tex, encoding="utf-8")
    ausentes = [n for n in MAPA if n not in feitas]
    print(f"figuras {acao}: {feitas}")
    if ausentes:
        print(f"NÃO processadas: {ausentes}")


if __name__ == "__main__":
    main()
