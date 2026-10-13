"""Exporta as figuras do corpo no formato que a PLOS exige para submissão.

A PLOS pede arquivos SEPARADOS, nomeados `Fig1`, `Fig2`, ..., em TIFF ou EPS, com pelo
menos 300 dpi. As figuras do projeto são geradas a 150 dpi para leitura em tela, o que não
serve à impressão.

Este script REGENERA cada figura a 600 dpi em vez de converter o PNG existente: converter
um raster de 150 dpi para TIFF não recupera resolução nenhuma, apenas produz um arquivo
maior com a mesma qualidade. Regenerar redesenha o vetor no dpi de destino.

Compressão LZW, que é sem perda e aceita pela PLOS. As figuras vão para
`artifacts/plos_figures/`.

Uso:
    poetry run python scripts/export_plos_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import tb_outcomes.figures as figs  # noqa: E402

OUT = Path("artifacts/plos_figures")

# A PLOS especifica os limites em PIXELS, não em polegadas: largura de 789 a 2250 px e
# altura de até 2625 px, com resolução entre 300 e 600 dpi. A 600 dpi, 2250 px equivalem a
# 3,75 polegadas, e não a 7,5 — foi o erro da primeira versão deste script.
#
# A saída fica a 300 dpi, o mínimo aceito, porque é o que permite a largura física máxima
# dentro do teto de pixels: 2250 px / 300 dpi = 7,5 polegadas.
MAX_PX_LARGURA = 2250
MAX_PX_ALTURA = 2625
DPI_SAIDA = 300

# Renderiza acima do alvo e reamostra para baixo. Reduzir a figura no matplotlib não reduz
# o texto, porque o tamanho da fonte é medido em pontos, que é absoluto: a escala trava num
# ponto fixo em que o texto domina. Reamostrar resolve isso e ainda melhora o traçado, por
# supersampling.
DPI_RENDER = 600

# número no manuscrito -> função que produz a figura
# Espelha o MAPA de scripts/inject_manuscript_figures.py; se um mudar, o outro tem de mudar.
FIGURAS = {
    1: ("Cohort definition", lambda: figs.fig_cohort_funnel()),
    2: ("Model ranking", lambda: figs.fig_ranking_f1()),
    3: ("Calibration", lambda: figs.fig_reliability_curves()),
    4: ("Ranking stability across k", lambda: figs.fig_leaderboard_across_k(
        pd.read_csv("data/sweep/robustness_table.csv"))),
    5: ("Territorial ablation", lambda: figs.fig_ablation_arms()),
    6: ("Feature attribution", lambda: figs.fig_shap_by_class()),
    7: ("Discrimination by race and colour", lambda: figs.fig_equity_by_group()),
    8: ("Decision curves", lambda: figs.fig_decision_curves()),
}


def main() -> None:
    import sys as _sys
    _sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent / "editorial"))
    from guarda_historico import exigir_historico
    exigir_historico("export_plos_figures.py", "python scripts/editorial/figuras_c7.py")
    OUT.mkdir(parents=True, exist_ok=True)
    original = figs._save
    feitas, falhas = [], []

    for n, (titulo, fn) in FIGURAS.items():
        destino = OUT / f"Fig{n}.tif"

        # intercepta o _save do módulo: a figura é desenhada pela mesma função que produz
        # a versão do manuscrito, então as duas não podem divergir.
        def _save_tif(fig, name, _dest=destino):
            """Renderiza acima do alvo e reamostra para caber no teto de pixels da PLOS."""
            from io import BytesIO

            from PIL import Image

            buf = BytesIO()
            fig.savefig(buf, dpi=DPI_RENDER, format="png", bbox_inches="tight",
                        facecolor="white")
            plt.close(fig)
            buf.seek(0)
            im = Image.open(buf).convert("RGB")

            escala = min(MAX_PX_LARGURA / im.width, MAX_PX_ALTURA / im.height, 1.0)
            if escala < 1.0:
                im = im.resize((max(1, round(im.width * escala)),
                                max(1, round(im.height * escala))), Image.LANCZOS)
            im.save(_dest, format="TIFF", compression="tiff_lzw",
                    dpi=(DPI_SAIDA, DPI_SAIDA))
            return _dest

        figs._save = _save_tif
        try:
            fn()
            mb = destino.stat().st_size / 1048576
            feitas.append((n, titulo, mb))
            print(f"  Fig{n}.tif  {mb:6.2f} MB  {titulo}")
        except Exception as exc:  # noqa: BLE001
            falhas.append((n, titulo, str(exc)[:100]))
            print(f"  Fig{n}     FALHOU: {str(exc)[:100]}")
        finally:
            figs._save = original

    print(f"\n{len(feitas)} figuras em {OUT}/, TIFF com compressão LZW")
    if falhas:
        print("FALHAS:")
        for n, t, e in falhas:
            print(f"  Fig{n} ({t}): {e}")

    # conferência final: o limite da PLOS é em pixels
    from PIL import Image
    fora = []
    for n, _, _ in feitas:
        im = Image.open(OUT / f"Fig{n}.tif")
        if im.width > MAX_PX_LARGURA or im.height > MAX_PX_ALTURA:
            fora.append((n, im.width, im.height))
    if fora:
        print("\nFORA DO LIMITE DA PLOS:")
        for n, w, h in fora:
            print(f"  Fig{n}: {w} x {h} px")
    else:
        print(f"todas dentro de {MAX_PX_LARGURA} x {MAX_PX_ALTURA} px, a {DPI_SAIDA} dpi")

    grandes = [(n, mb) for n, _, mb in feitas if mb > 10]
    if grandes:
        print("\nAcima de 10 MB, verificar o limite do sistema de submissão:")
        for n, mb in grandes:
            print(f"  Fig{n}: {mb:.1f} MB")


if __name__ == "__main__":
    main()
