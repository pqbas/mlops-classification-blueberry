import numpy as np
from PIL import Image
from pathlib import Path

root = Path("data/blueberry_five_classes_chopped_depurated")
for cls in sorted(p.name for p in root.iterdir() if p.is_dir()):
    files = sorted((root/cls).glob("*.jpg"))[:30]
    fracs = []
    for f in files:
        a = np.asarray(Image.open(f).convert("RGB"))
        frac = (a.max(axis=2) < 30).mean()  # pixeles casi negros
        fracs.append(frac)
    print(f"{cls:14s} n={len(files)} frac_negro_medio={np.mean(fracs):.3f}")
