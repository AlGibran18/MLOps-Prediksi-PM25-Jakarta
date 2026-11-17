"""Pulihkan snapshot fitur dari cache DVC yang sudah dilacak Git (tanpa MinIO)."""
import hashlib
from pathlib import Path
import shutil

import yaml

pointer = Path("data/processed/features.csv.dvc")
metadata = yaml.safe_load(pointer.read_text())["outs"][0]
checksum = metadata["md5"]
source = Path("dvc_ci_store/files/md5") / checksum[:2] / checksum[2:]
if hashlib.md5(source.read_bytes()).hexdigest() != checksum:
    raise ValueError("Checksum snapshot fitur tidak cocok dengan pointer DVC")
shutil.copyfile(source, pointer.parent / metadata["path"])
