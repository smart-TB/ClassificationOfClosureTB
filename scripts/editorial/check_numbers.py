"""Confere os números do manuscrito contra o dossiê do L5 e a paridade entre inglês e português.

A instrução do L5 pede que se revisem também os números fora dos marcadores. Reler duas
versões de trinta páginas à procura de um decimal trocado não é confiável; este verificador
extrai todo número com casa decimal, e todo inteiro a partir de mil, do resumo, dos resultados
e da discussão, e exige que cada um corresponda a um valor do dossiê, na escala original, em
porcentagem ou por mil, com o arredondamento que o texto usa.

O que ele NÃO verifica: se o número certo está na frase certa. Com milhares de valores no
dossiê, a coincidência por acaso é frequente, de modo que isto não é prova de origem contextual
nem de paridade semântica entre as versões; essa conferência continua sendo leitura (ver
`numeros_em_contexto.py`). Também não cobre métodos nem introdução. O que ele garante é que
nenhum número das regiões cobertas ficou sem valor correspondente, com o mesmo sinal, no
catálogo liberado, e que as duas versões trazem os mesmos números com os mesmos sinais.

Uso:
    python scripts/editorial/check_numbers.py
"""
from __future__ import annotations

import re
from collections import Counter

from catalogo import RAIZ
from dossie_c7 import construir

BASE = RAIZ / "manuscrito/PLOS_DigitalHealth"
REGIOES = {
    "manuscript.tex": [(r"\section*{Abstract}", r"\section*{Author summary}"),
                       (r"\section*{Results}", r"\section*{Supporting information}")],
    "manuscrito_ptbr.tex": [(r"\section*{Resumo}", r"\noindent\textbf{Palavras-chave:}"),
                            (r"\section{Resultados}", r"\section*{Contribuição dos autores}")],
}
# Números que são texto e não resultado: nível de confiança, capacidades, faixas etárias,
# limiares da curva de decisão e a base de "por 1.000 notificações".
FIXOS = {"1000", "0.05", "0.10", "0.20", "0.1", "0.2"}
# o sinal de menos do texto é sempre `$-$`; a meia-risca de intervalo (`--`) não é sinal
NUMERO = re.compile(r"(?P<neg>\$-\$)?(?P<num>(?<![\w.,])\d[\d.,]*\d|(?<![\w.,])\d)")


def trecho(texto: str, ini: str, fim: str) -> str:
    i = texto.index(ini)
    return texto[i:texto.index(fim, i)]


def limpar(t: str) -> str:
    t = re.sub(r"(?m)^%.*$", "", t)                       # comentários e marcadores
    t = re.sub(r"\\(includegraphics|label|ref|cite)(\[[^\]]*\])?\{[^}]*\}", " ", t)
    t = re.sub(r"\\begin\{(tabular|adjustwidth|adjustbox)\}(\{[^}]*\})+", " ", t)
    t = re.sub(r"\\(setlength|multicolumn)(\{[^}]*\})+", " ", t)
    t = re.sub(r"(Fig|Figura|Table|Tabela|Tables|Figs)[~ ]S?\d+", " ", t)   # remissões
    t = re.sub(r"S\d+ (Table|Fig|Tables|Figs)", " ", t)
    return t


def numeros(arquivo: str) -> list[str]:
    """Decimais e inteiros >= 1000, normalizados para ponto decimal e sem separador de milhar."""
    pt = arquivo.endswith("ptbr.tex")
    texto = (BASE / arquivo).read_text(encoding="utf-8")
    out = []
    for ini, fim in REGIOES[arquivo]:
        for m in NUMERO.finditer(limpar(trecho(texto, ini, fim))):
            s = m.group("num").rstrip(".,")
            if pt:
                s = s.replace(".", "").replace(",", ".")
            else:
                s = s.replace(",", "")
            if "." not in s and (len(s) < 4 or (len(s) == 4 and 2000 <= int(s) <= 2030)):
                continue  # inteiros pequenos e anos: texto, não resultado
            out.append(("-" if m.group("neg") else "") + s)
    return out


def formas_do_dossie() -> set[str]:
    d, _ = construir()
    formas: set[str] = set()
    for v in d.valores.values():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        if isinstance(v, int):
            formas.add(str(v))
            continue
        for escala in (1, 100, 1000):
            for casas in range(0, 7):
                f = f"{v * escala:.{casas}f}"
                formas.add(f if float(f) != 0 else f.lstrip("-"))
    return formas


def conferir() -> tuple[dict[str, list[str]], Counter, Counter]:
    formas = formas_do_dossie() | FIXOS
    sem_origem = {}
    contagem = {}
    for arquivo in REGIOES:
        ns = numeros(arquivo)
        contagem[arquivo] = Counter(ns)
        sem_origem[arquivo] = sorted({n for n in ns if n not in formas})
    en, pt = contagem["manuscript.tex"], contagem["manuscrito_ptbr.tex"]
    return sem_origem, en - pt, pt - en


def main() -> int:
    sem_origem, so_en, so_pt = conferir()
    ruim = False
    for arquivo, lista in sem_origem.items():
        total = len(numeros(arquivo))
        print(f"{arquivo}: {total} números conferidos, {len(lista)} sem origem no dossiê")
        for n in lista:
            print(f"    {n}")
        ruim |= bool(lista)
    if so_en or so_pt:
        ruim = True
        print(f"paridade EN/PT: só no inglês {dict(so_en)}; só no português {dict(so_pt)}")
    else:
        print("paridade EN/PT: os dois textos trazem exatamente os mesmos números")
    return 1 if ruim else 0


if __name__ == "__main__":
    raise SystemExit(main())
