"""Registro do Claude no protocolo de execução conjunta: diário e status.

O contrato exige que eventos sejam ACRESCENTADOS ao `journal.jsonl` e que o `status.json`
seja gravado por rename atômico, para o Codex nunca ler um arquivo pela metade. Centralizar
isso num utilitário evita que cada chamada reinvente o formato e esqueça um campo
obrigatório.

Uso:
    python scripts/editorial/registro.py evento --task L0 --event completed \
        --summary "…" --files a.py b.tex --evidence log.txt --next L1
    python scripts/editorial/registro.py status --task L1 --set L0=completed L1=in_progress \
        --next "…" --codex-seen 2026-09-19T16:48:18+00:00
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parents[2] / "docs/manuscrito/colaboracao/claude"
DIARIO = BASE / "journal.jsonl"
STATUS = BASE / "status.json"

CAMPOS = ("timestamp", "task_id", "event", "summary", "files", "command",
          "exit_code", "evidence", "next_step", "blocker")


def agora() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def evento(a: argparse.Namespace) -> None:
    reg = {
        "timestamp": agora(),
        "task_id": a.task,
        "event": a.event,
        "summary": a.summary,
        "files": a.files or None,
        "command": a.command,
        "exit_code": a.exit_code,
        "evidence": a.evidence or None,
        "next_step": a.next,
        "blocker": a.blocker,
    }
    assert set(reg) == set(CAMPOS), "campos do diário divergem do contrato"
    with DIARIO.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(reg, ensure_ascii=False) + "\n")
    print(f"diário += {a.task}/{a.event}")


def status(a: argparse.Namespace) -> None:
    doc = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.is_file() else {}
    doc["agent"] = "claude"
    doc["updated_at"] = agora()
    if a.task is not None:
        doc["current_task"] = a.task
    if a.state is not None:
        doc["status"] = a.state
    tarefas = doc.setdefault("tasks", {})
    for par in a.set or []:
        k, _, v = par.partition("=")
        tarefas[k] = v
    if a.files is not None:
        doc["files_in_progress"] = a.files
    if a.blockers is not None:
        doc["blockers"] = a.blockers
    if a.next is not None:
        doc["next_step"] = a.next
    if a.codex_seen is not None:
        doc["last_codex_update_seen"] = a.codex_seen

    # rename atômico: o Codex nunca deve ler um status pela metade
    fd, tmp = tempfile.mkstemp(dir=str(BASE), prefix=".status-", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.replace(tmp, STATUS)
    print(f"status: {doc.get('current_task')} | {doc.get('status')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("evento")
    e.add_argument("--task", required=True)
    e.add_argument("--event", required=True,
                   choices=["started", "progress", "completed", "error", "blocked", "note"])
    e.add_argument("--summary", required=True)
    e.add_argument("--files", nargs="*")
    e.add_argument("--command", default=None)
    e.add_argument("--exit-code", type=int, default=None, dest="exit_code")
    e.add_argument("--evidence", nargs="*")
    e.add_argument("--next", default=None)
    e.add_argument("--blocker", default=None)
    e.set_defaults(fn=evento)

    s = sub.add_parser("status")
    s.add_argument("--task", default=None)
    s.add_argument("--state", default=None)
    s.add_argument("--set", nargs="*", help="L0=completed L1=in_progress")
    s.add_argument("--files", nargs="*", default=None)
    s.add_argument("--blockers", nargs="*", default=None)
    s.add_argument("--next", default=None)
    s.add_argument("--codex-seen", default=None, dest="codex_seen")
    s.set_defaults(fn=status)

    a = ap.parse_args()
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
