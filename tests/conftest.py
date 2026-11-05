import sys
from pathlib import Path

# Agar `from train import ...` dan `from evaluate_model import ...` bisa dipakai di tes.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "models"))
