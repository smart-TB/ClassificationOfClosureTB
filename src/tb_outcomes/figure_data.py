"""Dados numéricos que originaram cada figura (ESPECIFICACAO §6, última linha).

Atende o comentário secundário do R3: cada figura publicada precisa vir acompanhada dos
números que a produziram, para que um leitor possa refazer o gráfico ou conferir um valor
sem reexecutar o pipeline.

**A extração é feita dos artistas do matplotlib, não da fonte de dados.** É uma escolha
deliberada: o que interessa ao revisor é o que foi *desenhado*, e re-derivar os números a
partir do CSV de origem abriria espaço para o acompanhamento divergir da figura em silêncio
— exatamente o defeito que a §6 existe para fechar. Aqui, se a linha está no gráfico, ela
está no arquivo.

Cobre os tipos usados nas figuras deste projeto: linhas e marcadores (`Line2D`), barras e
histogramas (`Rectangle`), dispersões (`PathCollection`) e mapas de calor (`AxesImage`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Teto de pontos por série. Uma curva PR/ROC sobre 827 mil registros tem um vértice por
# limiar único — 617 mil pontos numa figura de 8 cm. Isso não é resolução, é o dado bruto
# de novo, e num acompanhamento público seria um segundo canal de microdado.
# O corte é por passo uniforme, preserva primeiro e último ponto, e fica DECLARADO nas
# colunas `subamostrada` / `n_pontos_originais` — nunca em silêncio.
MAX_PONTOS_POR_SERIE = 2000


def _rotulo(artista, indice: int) -> str:
    """Rótulo do artista, ignorando os automáticos do matplotlib (`_child0`, `_nolegend_`)."""
    try:
        lab = artista.get_label()
    except Exception:  # noqa: BLE001
        return f"série_{indice}"
    if not lab or str(lab).startswith("_"):
        return f"série_{indice}"
    return str(lab)


def _limita(xy: np.ndarray) -> tuple[np.ndarray, bool, int]:
    """Aplica o teto de pontos por passo uniforme, preservando as pontas."""
    n = len(xy)
    if n <= MAX_PONTOS_POR_SERIE:
        return xy, False, n
    passo = int(np.ceil(n / MAX_PONTOS_POR_SERIE))
    idx = list(range(0, n, passo))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return xy[idx], True, n


def extract_axes(ax, ax_idx: int) -> list[dict]:
    linhas: list[dict] = []

    # linhas e marcadores
    for i, ln in enumerate(getattr(ax, "lines", [])):
        xy = np.asarray(ln.get_xydata(), dtype=float)
        if xy.size == 0:
            continue
        rot = _rotulo(ln, i)
        xy, subamostrada, n_orig = _limita(xy)
        for j, (x, y) in enumerate(xy):
            linhas.append({"eixo": ax_idx, "tipo": "linha", "serie": rot,
                           "ponto": j, "x": x, "y": y,
                           "subamostrada": subamostrada, "n_pontos_originais": n_orig})

    # histogramas em degrau (`hist(histtype="stepfilled")`) produzem Polygon, não
    # Rectangle: são um contorno único com os vértices dos degraus.
    for i, p in enumerate(getattr(ax, "patches", [])):
        if p.__class__.__name__ != "Polygon":
            continue
        try:
            xy = np.asarray(p.get_xy(), dtype=float)
        except Exception:  # noqa: BLE001
            continue
        if xy.size == 0 or xy.ndim != 2:
            continue
        rot = _rotulo(p, i)
        xy, sub, n_orig = _limita(xy)
        for j, (x, y) in enumerate(xy):
            linhas.append({"eixo": ax_idx, "tipo": "contorno", "serie": rot,
                           "ponto": j, "x": x, "y": y,
                           "subamostrada": sub, "n_pontos_originais": n_orig})

    # barras e histogramas em barra
    patches = [p for p in getattr(ax, "patches", [])
               if p.__class__.__name__ == "Rectangle"]
    for i, p in enumerate(patches):
        try:
            x, y, w, h = p.get_x(), p.get_y(), p.get_width(), p.get_height()
        except Exception:  # noqa: BLE001
            continue
        # barra vertical -> o valor é a altura; horizontal -> a largura
        vertical = abs(h) >= abs(w)
        linhas.append({
            "eixo": ax_idx, "tipo": "barra", "serie": _rotulo(p, i), "ponto": i,
            "x": x + w / 2 if vertical else y + h / 2,
            "y": h if vertical else w,
            "x_inicio": x, "y_inicio": y, "largura": w, "altura": h,
        })

    # dispersões
    for i, col in enumerate(getattr(ax, "collections", [])):
        try:
            off = np.asarray(col.get_offsets(), dtype=float)
        except Exception:  # noqa: BLE001
            continue
        if off.size == 0 or off.ndim != 2:
            continue
        rot = _rotulo(col, i)
        for j, (x, y) in enumerate(off):
            linhas.append({"eixo": ax_idx, "tipo": "dispersao", "serie": rot,
                           "ponto": j, "x": x, "y": y})

    # mapas de calor / matrizes de confusão
    for i, im in enumerate(getattr(ax, "images", [])):
        try:
            arr = np.asarray(im.get_array(), dtype=float)
        except Exception:  # noqa: BLE001
            continue
        if arr.ndim != 2:
            continue
        for (r, c), v in np.ndenumerate(arr):
            linhas.append({"eixo": ax_idx, "tipo": "celula", "serie": _rotulo(im, i),
                           "ponto": r * arr.shape[1] + c, "x": c, "y": r, "valor": v})

    # rótulos dos ticks, quando são categóricos — sem eles um eixo de barras fica ilegível
    for nome, eixo in (("x", ax.get_xaxis()), ("y", ax.get_yaxis())):
        try:
            textos = [t.get_text() for t in eixo.get_ticklabels()]
            posicoes = list(eixo.get_ticklocs())
        except Exception:  # noqa: BLE001
            continue
        for pos, txt in zip(posicoes, textos):
            if txt and not txt.replace("−", "-").replace(".", "").replace("-", "").isdigit():
                linhas.append({"eixo": ax_idx, "tipo": f"rotulo_tick_{nome}",
                               "serie": "tick", "ponto": None, "x": pos, "rotulo": txt})
    return linhas


def extract_figure(fig, name: str) -> pd.DataFrame:
    """Tabela longa com tudo o que foi desenhado na figura."""
    # os rótulos de tick só existem depois do desenho: num eixo categórico, antes disso
    # `get_ticklabels()` devolve strings vazias. No fluxo real o savefig já desenhou;
    # forçar aqui faz a função valer também quando chamada sozinha.
    try:
        fig.canvas.draw()
    except Exception:  # noqa: BLE001 — backend sem canvas não impede extrair as séries
        pass
    linhas: list[dict] = []
    for i, ax in enumerate(fig.get_axes()):
        linhas.extend(extract_axes(ax, i))
    d = pd.DataFrame(linhas)
    if d.empty:
        return d
    d.insert(0, "figura", name)
    ordem = [c for c in ("figura", "eixo", "tipo", "serie", "ponto", "x", "y", "valor",
                         "rotulo", "x_inicio", "y_inicio", "largura", "altura",
                         "subamostrada", "n_pontos_originais")
             if c in d.columns]
    return d[ordem]
