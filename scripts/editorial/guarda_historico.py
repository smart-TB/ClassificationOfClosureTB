"""Trava dos consumidores editoriais que leem as tabelas históricas de `data/`.

Depois da reanálise corrigida, os scripts anteriores ao L5 continuam existindo como registro
de como a versão substituída foi montada, mas executá-los por engano recolocaria números da
corrida antiga no manuscrito: três deles usam os mesmos marcadores de tabela e de figura que
os geradores novos. Cada um chama `exigir_historico` no início do `main`; sem a opção
explícita, o script explica o que o substituiu e sai sem tocar em arquivo nenhum.
"""
from __future__ import annotations

import sys

OPCAO = "--historico"


def exigir_historico(script: str, substituto: str) -> None:
    if OPCAO in sys.argv:
        sys.argv.remove(OPCAO)
        return
    raise SystemExit(
        f"{script}: lê as tabelas históricas de data/, substituídas pela reanálise corrigida.\n"
        f"Para o manuscrito atual use: {substituto}\n"
        f"Para reproduzir a versão histórica, de propósito, acrescente {OPCAO}.")
