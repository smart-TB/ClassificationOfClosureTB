"""Lista os marcadores `<<L5:…>>` deixados nos manuscritos durante o L2.

Cada marcador assinala um ponto cujo número depende da liberação do Codex. Mantê-los numa
lista gerada, e não copiada à mão, evita que um marcador seja esquecido quando as linhas
deslocarem — e serve de conferência final: ao terminar o L5, a contagem tem de ser zero.

Uso:
    python scripts/editorial/list_l5_markers.py
    python scripts/editorial/list_l5_markers.py --md > docs/.../claude/plano_L5.md
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
ALVOS = [
    "manuscrito/PLOS_DigitalHealth/manuscript.tex",
    "manuscrito/PLOS_DigitalHealth/manuscrito_ptbr.tex",
    "manuscrito/PLOS_DigitalHealth/S1_Checklist.tex",
    "manuscrito/PLOS_DigitalHealth/S1_File.tex",
]
PADRAO = re.compile(r"<<L5:([^>]*)>>")


def achar() -> list[dict]:
    achados = []
    for rel in ALVOS:
        p = RAIZ / rel
        if not p.is_file():
            continue
        linhas = p.read_text(encoding="utf-8").splitlines()
        for n, linha in enumerate(linhas, 1):
            m = PADRAO.search(linha)
            if not m:
                continue
            # a frase que o marcador qualifica costuma estar na linha anterior
            contexto = linhas[n - 2].strip() if n >= 2 else ""
            achados.append({"arquivo": rel, "linha": n,
                            "pedido": m.group(1).strip(),
                            "contexto": contexto[:110]})
    return achados


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true", help="saída em Markdown")
    args = ap.parse_args()

    achados = achar()
    if args.md:
        print("# Plano do L5 — marcadores pendentes\n")
        print("Gerado por `scripts/editorial/list_l5_markers.py`. Ao concluir o L5 a")
        print("contagem tem de ser **zero**.\n")
        print(f"**{len(achados)} marcadores** em {len({a['arquivo'] for a in achados})} arquivos.\n")
        por_pedido: dict[str, list[dict]] = {}
        for a in achados:
            por_pedido.setdefault(a["pedido"], []).append(a)
        for pedido, itens in sorted(por_pedido.items()):
            print(f"## {pedido}\n")
            for i in itens:
                arq = i["arquivo"].split("/")[-1]
                print(f"- `{arq}:{i['linha']}` — …{i['contexto']}")
            print()
        return 0

    print(f"{len(achados)} marcadores <<L5:>>")
    for a in achados:
        print(f"  {a['arquivo'].split('/')[-1]}:{a['linha']:>4}  {a['pedido']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
