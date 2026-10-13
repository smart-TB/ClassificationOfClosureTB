"""Restore the public snapshot to working paths after verifying every checksum."""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'datasets/2026-10-13'
for item in json.loads((D/'manifest.json').read_text())['files']:
 src=D/item['file'];dst=ROOT/item['restore_to'];digest=hashlib.sha256(src.read_bytes()).hexdigest()
 if digest!=item['sha256']:raise ValueError(f"Source checksum mismatch: {src.name}")
 if dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest()!=digest:raise FileExistsError(f"Different local data: {dst}")
 dst.parent.mkdir(parents=True,exist_ok=True)
 if not dst.exists():shutil.copy2(src,dst)
print('Published data verified and restored; no models trained.')
