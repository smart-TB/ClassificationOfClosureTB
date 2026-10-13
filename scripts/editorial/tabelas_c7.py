"""Tabelas do corpo do manuscrito, em inglês e em português, a partir do dossiê do L5.

Mesmo mecanismo de injeção por marcador dos scripts anteriores: a PLOS exige fonte LaTeX
único, sem `\\input`, e o protocolo proíbe número copiado à mão. As duas versões saem do
mesmo dossiê, de modo que não podem divergir; só mudam rótulos e separador decimal.

Cada legenda declara braço, regra de decisão e agregação. O intervalo é sempre o do macro-F1
agrupado por registros; a média entre dobras vem acompanhada do desvio-padrão entre dobras e
nunca do intervalo.

Uso:
    python scripts/editorial/tabelas_c7.py           # injeta nas duas versões
"""
from __future__ import annotations

import re
import sys

from catalogo import RAIZ
from dossie_c7 import CLASSES, Dossie, construir

sys.path.insert(0, str(RAIZ / "src"))
from tb_outcomes import labels as rot  # noqa: E402

BASE = RAIZ / "manuscrito/PLOS_DigitalHealth"
TEX_EN = BASE / "manuscript.tex"
TEX_PT = BASE / "manuscrito_ptbr.tex"

ORDEM_BRACOS = ["primary", "temporal", "clinical", "individual", "municipal", "contemporary",
                "without_diagnostics", "hiv_uncertain", "k27", "k75", "random_group", "all_cause"]
SEM_DIFERENCA = {"primary", "temporal", "all_cause"}  # sem comparação pareada com o principal
CLS_EN = {"cure": "Cure", "tb_death": "TB death", "treatment_interruption": "Treatment interruption"}
CLS_PT = {"cure": "Cura", "tb_death": "Óbito por TB",
          "treatment_interruption": "Interrupção do tratamento"}


def num(v: float, casas: int = 3, pt: bool = False, sinal: bool = False) -> str:
    """Decimal com ponto (EN) ou vírgula (PT); sinal de menos tipográfico do LaTeX."""
    s = f"{v:+.{casas}f}" if sinal else f"{v:.{casas}f}"
    if float(s) == 0:  # evita "-0.000"
        s = f"{0:.{casas}f}"
    s = s.replace("-", "$-$").replace("+", "$+$") if sinal else s.replace("-", "$-$")
    return s.replace(".", ",") if pt else s


def inteiro(n: int, pt: bool = False) -> str:
    s = f"{int(n):,}"
    return s.replace(",", ".") if pt else s


def ic(d: Dossie, chave: str, casas: int = 3, pt: bool = False, sinal: bool = False,
       traco: bool = False) -> str:
    """Estimativa e intervalo. `traco=True` usa meia-risca, só para valores sem sinal: numa
    diferença, a meia-risca ao lado do sinal de menos seria ilegível."""
    sep = "--" if traco and not sinal else (" a " if pt else " to ")
    return (f"{num(d[chave], casas, pt, sinal)} ({num(d[chave + '.lo'], casas, pt, sinal)}{sep}"
            f"{num(d[chave + '.hi'], casas, pt, sinal)})")


def _amb(pt: bool) -> tuple[str, str, str]:
    """Réguas: `\\hline` no manuscrito em inglês (padrão PLOS), booktabs no português."""
    return (r"\toprule", r"\midrule", r"\bottomrule") if pt else (r"\hline", r"\hline", r"\hline")


def t_byclass(d: Dossie, pt: bool = False) -> str:
    topo, meio, fim = _amb(pt)
    cls = CLS_PT if pt else CLS_EN
    if pt:
        legenda = (r"\caption{{\bf Desempenho por classe de desfecho do procedimento de seleção "
                   r"aninhada.} Braço principal, desenvolvimento de 2015 a 2023 (n = "
                   f"{inteiro(d['coorte.dev'], True)}" r"), predições das dobras externas agrupadas "
                   r"por registro. O arg max das probabilidades calibradas é a regra principal; os "
                   r"limiares por classe são a política secundária. Entre parênteses, intervalos de "
                   r"95\% por bootstrap de agrupamentos geográficos, condicionais às predições fixas. "
                   r"AUROC e AUPRC não dependem da regra e não têm intervalo. VPP, valor preditivo "
                   r"positivo; AUROC, área sob a curva ROC; AUPRC, área sob a curva de "
                   r"precisão-sensibilidade.}")
        cab = r"Classe & Regra & Sensibilidade & VPP & F1 & AUROC & AUPRC \\"
        regras = {"argmax": "Arg max", "policy": "Limiares"}
        rotulo = "tab:classe"
    else:
        legenda = (r"\caption{{\bf Performance by outcome class for the nested selection "
                   r"procedure.} Primary arm, development period 2015 to 2023 (n = "
                   f"{inteiro(d['coorte.dev'])}" r"), outer-fold predictions pooled by record. The "
                   r"arg max of the calibrated probabilities is the primary rule; per-class "
                   r"thresholds are the secondary policy. In parentheses, 95\% intervals from a "
                   r"bootstrap of geographic clusters, conditional on the fixed predictions. AUROC "
                   r"and AUPRC do not depend on the rule and carry no interval. PPV, positive "
                   r"predictive value; AUROC, area under the receiver operating characteristic "
                   r"curve; AUPRC, area under the precision-recall curve.}")
        cab = r"Class & Rule & Sensitivity & PPV & F1 & AUROC & AUPRC \\"
        regras = {"argmax": "Arg max", "policy": "Thresholds"}
        rotulo = "tab:byclass"
    L = [r"\begin{table}[!ht]", r"\begin{adjustwidth}{-2.25in}{0in}", r"\centering", r"\footnotesize",
         legenda, rf"\label{{{rotulo}}}", r"\begin{tabular}{llccccc}", topo, cab, meio]
    for i, c in enumerate(CLASSES.values()):
        for regra in ("argmax", "policy"):
            p = f"primary.{regra}.{c}"
            extra = (f"{num(d[f'primary.prob.{c}.auroc'], 3, pt)} & "
                     f"{num(d[f'primary.prob.{c}.auprc'], 3, pt)}" if regra == "argmax" else " & ")
            L.append(f"{cls[c] if regra == 'argmax' else ''} & {regras[regra]} & "
                     f"{ic(d, p + '.sensitivity', 3, pt, traco=True)} & "
                     f"{ic(d, p + '.ppv', 3, pt, traco=True)} & "
                     f"{ic(d, p + '.f1', 3, pt, traco=True)} & {extra} " + r"\\")
        if i < 2:
            L.append(meio)
    L += [fim, r"\end{tabular}", r"\end{adjustwidth}", r"\end{table}"]
    return "\n".join(L)


def t_arms(d: Dossie, pt: bool = False) -> str:
    topo, meio, fim = _amb(pt)
    if pt:
        legenda = (r"\caption{{\bf Braços de análise: F1 macro do procedimento de seleção "
                   r"aninhada.} Cada braço é um conjunto próprio de ajustes; o número de famílias e "
                   r"de pares algoritmo-estratégia candidatos difere entre eles (Tabela S2). A média e o "
                   r"desvio-padrão (DP) são calculados entre as cinco dobras externas, sem "
                   r"ponderação. O valor agrupado é calculado sobre todas as predições externas, com "
                   r"intervalo de 95\% por bootstrap de agrupamentos geográficos; esse intervalo não se "
                   r"refere à média entre dobras. A diferença é pareada contra o braço principal, "
                   r"com a seleção refeita no universo comum de candidatos, nos oito braços de conjunto de "
                   r"preditores e de desenho. Para o comparador clínico, a diferença é principal menos "
                   r"comparador e confronta os dois procedimentos completos nos mesmos registros, sem "
                   r"universo comum de candidatos. A diferença não é calculada para a "
                   r"avaliação temporal, que usa outra população, nem para o óbito por qualquer "
                   r"causa, que usa outro alvo e outra coorte.}")
        cab = (r"Braço & n & Média (DP) & Agrupado (IC 95\%) & "
               r"Agrupado (IC 95\%) & Diferença (IC 95\%) \\")
        grupo = (r" & & \multicolumn{2}{c}{Arg max (regra principal)} & Limiares & "
                 r"Arg max, vs.\ principal \\")
        rotulo, nd = "tab:bracos", "n.c."
    else:
        legenda = (r"\caption{{\bf Analysis arms: macro-averaged F1 of the nested selection "
                   r"procedure.} Each arm is a separate set of fits, and the number of candidate "
                   r"families and algorithm-strategy pairs differs between them (S2 Table). The mean and "
                   r"standard deviation (SD) are taken across the five outer folds, unweighted. The "
                   r"pooled value is computed on all outer-fold predictions, with a 95\% interval "
                   r"from a bootstrap of geographic clusters; that interval does not refer to the "
                   r"fold mean. The difference is paired against the primary arm, with selection "
                   r"repeated on the common universe of candidates, in the eight feature-set and design "
                   r"arms. For the clinical comparator the difference is primary minus comparator and "
                   r"sets the two complete procedures side by side on the same records, with no common "
                   r"universe of candidates. The difference is not computed for the "
                   r"temporal evaluation, which uses another population, nor for death from any "
                   r"cause, which uses another target and another cohort.}")
        cab = (r"Arm & n & Mean (SD) & Pooled (95\% CI) & "
               r"Pooled (95\% CI) & Difference (95\% CI) \\")
        grupo = (r" & & \multicolumn{2}{c}{Arg max (primary rule)} & Thresholds & "
                 r"Arg max, vs.\ primary \\")
        rotulo, nd = "tab:arms", "n/c"
    L = [r"\begin{table}[!ht]", r"\begin{adjustwidth}{-2.25in}{0in}", r"\centering",
         r"\scriptsize", r"\setlength{\tabcolsep}{3pt}", legenda, rf"\label{{{rotulo}}}",
         r"\begin{tabular}{lrclll}", topo,
         grupo, cab, meio]
    for b in ORDEM_BRACOS:
        sd = d[f"{b}.argmax.f1.fold_sd"]
        media = (f"{num(d[f'{b}.argmax.f1.fold_mean'], 3, pt)} ({num(sd, 3, pt)})" if sd is not None
                 else ("partição única" if pt else "single partition"))
        dif = nd if b in SEM_DIFERENCA else ic(d, f"{b}.argmax.dif", 4, pt, sinal=True)
        if b == "primary":
            dif = "referência" if pt else "reference"
        L.append(f"{rot.braco(b, pt)} & {inteiro(d[f'{b}.n'], pt)} & "
                 f"{media} & "
                 f"{ic(d, f'{b}.argmax.f1.pooled', 3, pt, traco=True)} & "
                 f"{ic(d, f'{b}.policy.f1.pooled', 3, pt, traco=True)} & "
                 f"{dif} " + r"\\")
        if b in ("clinical", "hiv_uncertain", "random_group"):
            L.append(meio)
    L += [fim, r"\end{tabular}", r"\end{adjustwidth}", r"\end{table}"]
    return "\n".join(L)


def t_utility(d: Dossie, pt: bool = False) -> str:
    topo, meio, fim = _amb(pt)
    cls = CLS_PT if pt else CLS_EN
    if pt:
        legenda = (r"\caption{{\bf Seleção retrospectiva sob restrição de capacidade.} Valor "
                   r"preditivo positivo, fração dos eventos observados incluída e notificações "
                   r"selecionadas por evento observado quando se toma a fração de maior probabilidade "
                   r"calibrada. Braço principal e comparador clínico, desenvolvimento de 2015 a 2023, "
                   r"predições externas agrupadas por registro; ordenamento apenas, sem atribuição de "
                   r"rótulo e sem intervalo. O comparador é uma regressão logística sobre idade, AIDS "
                   r"declarada, uso de álcool ou drogas e situação de rua. Nenhuma intervenção foi "
                   r"realizada: os valores descrevem a concentração de eventos, não um benefício "
                   r"observado.}")
        cab = r"Desfecho & Abordagem & Capacidade & VPP & Eventos incluídos & Notificações por evento \\"
        nomes = {"primary": "Procedimento de seleção", "clinical": "Comparador clínico"}
        rotulo = "tab:utilidade"
    else:
        legenda = (r"\caption{{\bf Retrospective selection under a capacity constraint.} Positive "
                   r"predictive value, share of observed events included, and notifications selected "
                   r"per observed event when the fraction with the highest calibrated probability is "
                   r"taken. Primary arm and clinical comparator, development period 2015 to 2023, "
                   r"outer-fold predictions pooled by record; ranking only, with no label assigned and "
                   r"no interval. The comparator is a logistic regression on age, declared AIDS, "
                   r"alcohol or drug use, and homelessness. No intervention was delivered: the values "
                   r"describe the concentration of events, not an observed benefit.}")
        cab = r"Outcome & Approach & Capacity & PPV & Events included & Notifications per event \\"
        nomes = {"primary": "Selection procedure", "clinical": "Clinical comparator"}
        rotulo = "tab:utility"
    L = [r"\begin{table}[!ht]", r"\begin{adjustwidth}{-2.25in}{0in}", r"\centering", legenda,
         rf"\label{{{rotulo}}}", r"\begin{tabular}{llcccc}", topo, cab, meio]
    for i, c in enumerate(("tb_death", "treatment_interruption")):
        for cap in (5, 10, 20):
            for b in ("primary", "clinical"):
                p = f"{b}.cap{cap}.{c}"
                L.append(f"{cls[c]} & {nomes[b]} & {cap}" r"\% & " f"{num(d[p + '.ppv'], 3, pt)} & "
                         f"{num(100 * d[p + '.capturado'], 1, pt)}" r"\% & "
                         f"{num(d[p + '.nns'], 2, pt)} " + r"\\")
        if i == 0:
            L.append(meio)
    L += [fim, r"\end{tabular}", r"\end{adjustwidth}", r"\end{table}"]
    return "\n".join(L)


def _moldura_pt(tabela: str) -> str:
    """A versão em português não usa o modelo da PLOS: sem `adjustwidth`, a tabela larga é
    reduzida à largura do texto por `adjustbox`, que não altera tabelas que já cabem."""
    linhas = []
    for linha in tabela.split("\n"):
        if "{adjustwidth}" in linha:
            continue
        if linha.startswith(r"\begin{tabular}"):
            linhas.append(r"\begin{adjustbox}{max width=\linewidth}")
        linhas.append(linha)
        if linha.startswith(r"\end{tabular}"):
            linhas.append(r"\end{adjustbox}")
    return "\n".join(linhas)


GERADORES = {"byclass": t_byclass, "arms": t_arms, "utility": t_utility}


def injetar(d: Dossie) -> dict[str, list[str]]:
    feito = {}
    for tex, pt, abre, fecha in ((TEX_EN, False, "TABELA", "FIM"), (TEX_PT, True, "TABELAPT", "FIMPT")):
        s = tex.read_text(encoding="utf-8")
        ok = []
        for nome, fn in GERADORES.items():
            marcador = re.compile(rf"% <<{abre}:{nome}>>.*?% <<{fecha}:{nome}>>", re.S)
            if not marcador.search(s):
                raise SystemExit(f"{tex.name}: marcador da tabela {nome!r} ausente")
            tabela = _moldura_pt(fn(d, pt)) if pt else fn(d, pt)
            corpo = f"% <<{abre}:{nome}>>\n{tabela}\n% <<{fecha}:{nome}>>"
            s = marcador.sub(lambda _: corpo, s)
            ok.append(nome)
        tex.write_text(s, encoding="utf-8")
        feito[tex.name] = ok
    return feito


if __name__ == "__main__":
    dossie, _ = construir()
    for arquivo, nomes in injetar(dossie).items():
        print(f"{arquivo}: tabelas injetadas {nomes}")
