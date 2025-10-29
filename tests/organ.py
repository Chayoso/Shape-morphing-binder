from pathlib import Path
import shutil, re

ROOT = Path(r"C:/dev/res2")       # ← Modify this
DEST = Path(r"C:/dev/res2/collected")  # ← Modify this
DEST.mkdir(parents=True, exist_ok=True)

pat = re.compile(r"^ep_?\d{3}_render($|\.)", re.IGNORECASE)  # Match both ep000_render and ep_000_render

def unique_path(p: Path) -> Path:
    if not p.exists():
        return p
    i = 1
    while True:
        cand = p.with_name(f"{p.stem}_{i}{p.suffix}")
        if not cand.exists():
            return cand
        i += 1

matches = [p for p in ROOT.rglob("*") if p.is_file() and pat.match(p.name)]
print(f"Found {len(matches)} files.")

for src in matches:
    parent = src.parent.name
    base = src.name

    # ★ Duplicate prevention: skip prefix if base already starts with 'parent_'
    if base.lower().startswith((parent + "_").lower()):
        dst_name = base
    else:
        dst_name = f"{parent}_{base}"

    dst = unique_path(DEST / dst_name)
    shutil.copy2(src, dst)
    print(f"Copied: {src} -> {dst}")
