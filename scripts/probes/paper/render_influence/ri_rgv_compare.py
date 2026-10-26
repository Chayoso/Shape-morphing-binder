"""ri_rgv_compare.py RGV_HELD_DIR -- render influence (CPU, PIL): "tag | render weight held" from RGV's 2 x 2 (A, B2, C,
D; front, 1080p panels) of job 01: (1) one still per time (t = 1.2 and 5.0 s) side by side at full size (JPEG) and both
times stacked at half size (PNG); (2) side-by-side frames for a video (half-size 2 x 2s; the shorter run holds its last
frame, labelled). The bodies are already flipped back by the renderer; nothing is redrawn here but the header strip."""
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib

INK, INK2 = (236, 238, 240), (184, 191, 199)
LEFT = "tag freeze-2026-10-15: λ by the rule (λ|g_render| = |g_physics| / 2 in every window)"
RIGHT = "render weight held (--lambda_ema 0: λ kept at the first window's value)"


def font(size):
    return ImageFont.truetype(os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf"), size)


def frames(d):
    return sorted(int(p.stem) for p in Path(d).glob("*.jpg"))


def header(w, h, t_label, note_l="", note_r=""):
    s = (np.arange(h) / max(1, h - 1))[:, None]
    im = Image.fromarray(np.repeat((((1 - s) * 40 + s * 52)).astype(np.uint8)[:, None, :].repeat(3, 2), w, 1), "RGB")
    dr = ImageDraw.Draw(im)
    f, f2 = font(int(h * 0.36)), font(int(h * 0.28))
    dr.text((24, h * 0.12), LEFT + note_l, font=f, fill=INK)
    dr.text((w // 2 + 24, h * 0.12), RIGHT + note_r, font=f, fill=INK)
    dr.text((24, h * 0.58), t_label, font=f2, fill=INK2)
    dr.line([(w // 2, 0), (w // 2, h)], fill=(90, 97, 105), width=3)
    return im


def pair(dt, dh, k, scale=1.0):
    kt, kh = frames(dt), frames(dh)
    a = Image.open(Path(dt) / f"{min(k, kt[-1]):04d}.jpg").convert("RGB")
    b = Image.open(Path(dh) / f"{min(k, kh[-1]):04d}.jpg").convert("RGB")
    if scale != 1.0:
        a = a.resize((int(a.width * scale), int(a.height * scale)), Image.LANCZOS)
        b = b.resize((int(b.width * scale), int(b.height * scale)), Image.LANCZOS)
    im = Image.new("RGB", (a.width + b.width, a.height), (40, 45, 52))
    im.paste(a, (0, 0))
    im.paste(b, (a.width, 0))
    ImageDraw.Draw(im).line([(a.width, 0), (a.width, a.height)], fill=(40, 45, 52), width=6)
    return im, k > kt[-1], k > kh[-1]


def main():
    D = Path(sys.argv[1])
    full_t, full_h = D / "render_tag/gridB2_35_1080", D / "render_held/gridB2_35_1080"
    half_t, half_h = D / "render_tag/gridB2half_35_1080", D / "render_held/gridB2half_35_1080"
    out = D / "compare"
    out.mkdir(exist_ok=True)
    rows = []
    for t in (1.2, 5.0):
        k = int(round(t * 20))
        im, et, eh = pair(full_t, full_h, k)
        hd = header(im.width, 110, f"bunny 300k, seed 97, t = {t:.1f} s (kept frame {k}); front; panels: A where the render "
                    "acts, B2 its pull against the typical pull now, C render vs physics agree / disagree, D what the "
                    "render did to dFᴄ", "  [ended: last frame]" if et else "", "  [ended: last frame]" if eh else "")
        st = Image.new("RGB", (im.width, im.height + hd.height))
        st.paste(hd, (0, 0))
        st.paste(im, (0, hd.height))
        st.save(out / f"tag_vs_held_t{t:.1f}s_full.jpg", quality=92)
        imh, _, _ = pair(half_t, half_h, k)
        hdh = header(imh.width, 70, f"t = {t:.1f} s", "  [ended]" if et else "", "  [ended]" if eh else "")
        rows += [hdh, imh]
        print("still", t, k, im.size, flush=True)
    st = Image.new("RGB", (rows[1].width, sum(r.height for r in rows)))
    y = 0
    for r in rows:
        st.paste(r, (0, y))
        y += r.height
    st.save(out / "tag_vs_held_1.2s_5.0s.png")
    vd = out / "sbs_frames"
    vd.mkdir(exist_ok=True)
    kt, kh = frames(half_t), frames(half_h)
    K = max(kt[-1], kh[-1])
    for k in range(K + 1):
        im, et, eh = pair(half_t, half_h, k)
        hd = header(im.width, 70, f"t = {k / 20:.2f} s", "  [run ended: its last frame]" if et else "",
                    "  [run ended: its last frame]" if eh else "")
        fr = Image.new("RGB", (im.width, im.height + hd.height))
        fr.paste(hd, (0, 0))
        fr.paste(im, (0, hd.height))
        fr.save(vd / f"{k:04d}.jpg", quality=92)
    print("side-by-side frames", K + 1, flush=True)


if __name__ == "__main__":
    main()
