"""Content-addressed output directories; historical artifacts cannot be resumed."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity_hash(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode()).hexdigest()


def bind_identity(directory, manifest):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "run_identity.json"
    record = {"run_id": identity_hash(manifest), "manifest": manifest}
    if path.exists():
        if json.loads(path.read_text()) != record:
            raise ValueError("Run fingerprint changed; use a new output directory")
    else:
        if any(directory.iterdir()):
            raise ValueError("Cannot bind an existing unidentified output directory")
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(record, indent=2), encoding="utf-8")
        temp.replace(path)
    return directory


def correction_run_root(*, sanity=False, root=Path(".")):
    root = Path(root)
    sources = sorted((root / "configs").glob("*.yaml"))
    # Presentation-only modules belong to Claude and cannot affect training.
    sources += [p for p in sorted((root / "src/tb_outcomes").glob("*.py"))
                if p.name not in {"figures.py", "labels.py"}]
    sources += [root / "data/harmonized.parquet",
                root / "data/correction_audit/partitions/manifest.json"]
    sources += sorted((root / "data/correction_audit/partitions").glob("*.parquet"))
    feature_source = root / "data/correction_audit/baseline_context.parquet"
    if feature_source.exists():
        sources.append(feature_source)
    packages = {}
    for name in ["numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "catboost", "torch"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "unavailable"
    manifest = {"protocol": "correction-v3", "sanity": bool(sanity), "packages": packages,
                "files": {p.relative_to(root).as_posix(): file_sha256(p) for p in sources}}
    directory = root / "data/reanalysis" / identity_hash(manifest)[:20]
    return bind_identity(directory, manifest)
