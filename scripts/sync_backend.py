"""Copy the shared feature code and exported model artifacts into backend/.

src/features.py is the single source of truth; the backend gets an exact copy
so the deployed model builds features the same way as training did.
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
shutil.copy(ROOT / "src" / "features.py", ROOT / "backend" / "app" / "features.py")
dst = ROOT / "backend" / "models"
dst.mkdir(exist_ok=True)
for name in ("xgb_reg.json", "xgb_clf.json", "metadata.json", "past_ipos.json"):
    shutil.copy(ROOT / "models" / name, dst / name)
print("synced features.py and", ", ".join(p.name for p in sorted(dst.iterdir())))
