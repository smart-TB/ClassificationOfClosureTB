"""Confere as páginas citadas no S1 Checklist contra o PDF compilado do manuscrito.

O checklist do TRIPOD+AI aponta, item a item, a página em que cada requisito é relatado.
Toda vez que um parágrafo entra ou sai do manuscrito, as seções deslocam e essas páginas
silenciosamente deixam de valer — um erro que ninguém percebe até o editor conferir.

O script reconstrói a FAIXA de páginas de cada seção, e não a página do seu título: uma
citação a "p. 3, Introduction" é legítima quando a introdução vai da 2 à 4, ainda que o
título esteja na 2. As faixas saem da ordem dos títulos no .tex casada com a primeira
ocorrência de cada um no PDF, procurando sempre adiante do título anterior — títulos curtos
como "Cohort" reaparecem no corpo do texto e casariam cedo demais sem essa restrição.

Uso:
    poetry run python scripts/check_checklist_pages.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

BASE = Path("manuscrito/PLOS_DigitalHealth")
CHECKLIST = BASE / "S1_Checklist.tex"
TEX = BASE / "manuscript.tex"
PDF = BASE / "manuscript.pdf"

# `p.~7--8, Analyses` ou `p.~4, Study design and setting`. O nome termina no primeiro
# separador que o checklist usa para encadear referências ou emendar prosa.
CITACAO = re.compile(r"p\.~(\d+)(?:--(\d+))?,\s*([A-Z][^;,.(\\\n]*)")


def paginas_do_pdf() -> list[str]:
    if not PDF.exists():
        sys.exit(f"{PDF} não existe; compile o manuscrito antes.")
    saida = subprocess.run(["pdftotext", "-layout", str(PDF), "-"],
                           capture_output=True, text=True, check=True).stdout
    return saida.split("\f")


def faixas(paginas: list[str]) -> dict[str, range]:
    """Nome da seção -> faixa de páginas que ela ocupa."""
    titulos = [re.sub(r"\\[a-zA-Z]+", "", t).strip()
               for t in re.findall(r"\\(?:sub)?section\*\{([^}]+)\}",
                                   TEX.read_text(encoding="utf-8"))]
    inicio: dict[str, int] = {}
    ordem: list[str] = []
    cursor = 0
    for t in titulos:
        achou = next((i for i in range(cursor, len(paginas)) if t in paginas[i]), None)
        if achou is None:
            continue
        # o primeiro registro vence: seções homônimas em Métodos e Resultados
        # (Temporal holdout, Cohort) não podem sobrescrever uma à outra
        if t not in inicio:
            inicio[t] = achou + 1
            ordem.append(t)
        cursor = achou

    out: dict[str, range] = {}
    for i, t in enumerate(ordem):
        fim = inicio[ordem[i + 1]] if i + 1 < len(ordem) else len(paginas)
        out[t] = range(inicio[t], fim + 1)
    return out


def main() -> int:
    paginas = paginas_do_pdf()
    span = faixas(paginas)
    texto = CHECKLIST.read_text(encoding="utf-8")

    problemas, conferidas, ignoradas = [], 0, set()
    for m in CITACAO.finditer(texto):
        ini, fim, secao = int(m.group(1)), m.group(2), " ".join(m.group(3).split())
        if secao not in span:
            ignoradas.add(secao)
            continue
        conferidas += 1
        citada = range(ini, (int(fim) if fim else ini) + 1)
        if not set(citada) & set(span[secao]):
            problemas.append((m.group(0).strip(), secao, span[secao]))

    print(f"{conferidas} citações conferidas em {CHECKLIST.name}")
    if ignoradas:
        print(f"  ({len(ignoradas)} nomes não são seções e foram ignorados)")
    if problemas:
        print(f"\n{len(problemas)} DESATUALIZADAS:")
        for citacao, secao, faixa in problemas:
            ultima = faixa.stop - 1
            fx = f"p. {faixa.start}" if ultima == faixa.start else f"p. {faixa.start}--{ultima}"
            print(f"  «{citacao}» — a seção «{secao}» está em {fx}")
        return 1
    print("todas conferem com o PDF compilado")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
