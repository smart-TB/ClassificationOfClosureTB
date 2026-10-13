"""Rasteriza um PDF e monta folhas de contato para a inspeção visual do L6.

A instrução do L6 pede inspeção de todas as páginas, e não apenas compilação sem erro: tabela
cortada, coluna sobreposta e figura fora da margem não aparecem no log do LaTeX. As folhas
ficam em `claude/logs/qa_visual/` como evidência do que foi olhado.

Uso:
    python scripts/editorial/folhas_de_contato.py manuscript.pdf [páginas por folha] [dpi]
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

RAIZ = Path(__file__).resolve().parents[2]
PDFS = RAIZ / "manuscrito/PLOS_DigitalHealth"
SAIDA = RAIZ / "docs/manuscrito/colaboracao/claude/logs/qa_visual"


def main() -> None:
    nome = sys.argv[1]
    por_folha = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    dpi = sys.argv[3] if len(sys.argv) > 3 else "88"
    SAIDA.mkdir(parents=True, exist_ok=True)
    base = Path(nome).stem
    for antigo in SAIDA.glob(f"{base}_p*.png"):
        antigo.unlink()
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdftoppm", "-r", dpi, "-png", str(PDFS / nome), f"{tmp}/p"], check=True)
        paginas = sorted(Path(tmp).glob("p-*.png"))
        for i in range(0, len(paginas), por_folha):
            grupo = [Image.open(p).convert("RGB") for p in paginas[i:i + por_folha]]
            larg = sum(g.width for g in grupo) + 12 * (len(grupo) - 1)
            folha = Image.new("RGB", (larg, max(g.height for g in grupo)), (120, 120, 120))
            x = 0
            for g in grupo:
                folha.paste(g, (x, 0))
                x += g.width + 12
            destino = SAIDA / f"{base}_p{i + 1:02d}-{min(i + por_folha, len(paginas)):02d}.png"
            folha.save(destino)
            print(destino.relative_to(RAIZ))


if __name__ == "__main__":
    main()
