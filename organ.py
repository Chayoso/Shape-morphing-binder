from pathlib import Path
import shutil, re

ROOT = Path(r"C:/dev/res2")       # ← 수정
DEST = Path(r"C:/dev/res2/collected")  # ← 수정
DEST.mkdir(parents=True, exist_ok=True)

pat = re.compile(r"^ep_?\d{3}_render($|\.)", re.IGNORECASE)  # ep000_render, ep_000_render 모두 허용

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

    # ★ 중복 방지: base가 이미 'parent_'로 시작하면 prefix 생략
    if base.lower().startswith((parent + "_").lower()):
        dst_name = base
    else:
        dst_name = f"{parent}_{base}"

    dst = unique_path(DEST / dst_name)
    shutil.copy2(src, dst)
    print(f"Copied: {src} -> {dst}")
