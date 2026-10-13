"""Lista cada número da prosa em inglês com a frase e as chaves do dossiê que o reproduzem.

Complementa `check_numbers.py`, que só garante que o número existe em algum lugar do dossiê.
Com milhares de valores, um decimal trocado tem boa chance de coincidir com outro valor
legítimo; a conferência que resta é ler o número junto da frase e da chave de origem. As
tabelas injetadas ficam de fora, porque são geradas por código a partir das mesmas chaves.

Uso:
    python scripts/editorial/numeros_em_contexto.py > .../logs/L5_numeros_contexto.txt
"""
from __future__ import annotations

import re

from check_numbers import BASE, NUMERO, REGIOES, limpar, trecho
from dossie_c7 import construir


def main() -> None:
    d, _ = construir()
    indice: dict[str, list[str]] = {}
    for k, v in d.valores.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        formas = {str(v)} if isinstance(v, int) else {
            (lambda f: f if float(f) != 0 else f.lstrip("-"))(f"{v * e:.{c}f}")
            for e in (1, 100, 1000) for c in range(0, 7)}
        for f in formas:
            indice.setdefault(f, []).append(k)

    texto = (BASE / "manuscript.tex").read_text(encoding="utf-8")
    for ini, fim in REGIOES["manuscript.tex"]:
        t = trecho(texto, ini, fim)
        t = re.sub(r"% <<TABELA:.*?% <<FIM:\w+>>", " ", t, flags=re.S)
        t = " ".join(limpar(t).split())
        for m in NUMERO.finditer(t):
            s = ("-" if m.group("neg") else "") + m.group("num").rstrip(".,").replace(",", "")
            if "." not in s and (len(s.lstrip("-")) < 4 or (len(s) == 4 and 2000 <= int(s) <= 2030)):
                continue
            ctx = t[max(0, m.start() - 95):m.end() + 25]
            chaves = indice.get(s, [])
            print(f"{s:>10} | {len(chaves):>3} | {ctx}")
            print(f"{'':>10} |     | {'; '.join(chaves[:7])}")


if __name__ == "__main__":
    main()
