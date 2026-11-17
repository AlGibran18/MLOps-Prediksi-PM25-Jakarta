"""Pulihkan raw dan fitur dari cache Git/DVC, tanpa memalsukan versi data."""
import hashlib
from pathlib import Path
import shutil
import yaml

for pointer in [Path("data/raw/air_quality_raw.csv.dvc"), Path("data/processed/features.csv.dvc")]:
    metadata = yaml.safe_load(pointer.read_text())["outs"][0]
    checksum = metadata["md5"]
    source = Path("dvc_ci_store/files/md5") / checksum[:2] / checksum[2:]
    if hashlib.md5(source.read_bytes()).hexdigest() != checksum:
        raise ValueError(f"Checksum snapshot tidak cocok: {pointer}")
    shutil.copyfile(source, pointer.parent / metadata["path"])
