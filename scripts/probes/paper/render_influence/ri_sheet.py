"""ri_sheet.py VIDEO_DIR OUT_PNG -- render influence: the hero still sheet of panel 3 in the RGV look (CPU, PIL): the
map's front view at t = 0.5, 1, 2 s and the end, bunny and dragon 300k, the bodies as ri_video.py drew them (without
the frame text), each mesh's colour bar and noise swatch cut from its own last frame (each run has its own fixed scale)."""
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib

INK, INK2, INK3 = (236, 238, 240), (184, 191, 199), (160, 168, 178)


def font(size):
    return ImageFont.truetype(os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf"), size)


def backdrop(w, h):
    s = (np.arange(h) / max(1, h - 1))[:, None]
    lin = (1 - s) * np.array((.115, .137, .165)) + s * np.array((.215, .237, .265))
    m = lin / (1 + lin)
    srgb = np.where(m <= .0031308, 12.92 * m, 1.055 * np.power(m, 1 / 2.4) - .055)
    return Image.fromarray(np.repeat((srgb * 255 + .5).astype(np.uint8)[:, None, :], w, 1), "RGB")


def bbox(imgs):
    """The union box of the body (pixels off the backdrop, whose rows are uniform) over the images."""
    box = None
    for im in imgs:
        a = np.asarray(im, np.int16)
        bg = np.median(a[:, :20], axis=1, keepdims=True)
        m = np.abs(a - bg).max(2) > 8
        ys, xs = np.nonzero(m)
        b = (xs.min(), ys.min(), xs.max(), ys.max())
        box = b if box is None else (min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3]))
    return box


def main():
    vdir, outp = Path(sys.argv[1]), Path(sys.argv[2])
    meshes = ("bunny", "dragon")
    CH = 560                                                   # crop height on the sheet
    rows = []
    for m in meshes:
        meta = json.load(open(vdir / f"vals_{m}.json"))
        tt = np.array([r["t"] for r in meta["rows"]])
        K = len(tt)
        ks = sorted({int(np.argmin(np.abs(tt - t_))) for t_ in (0.5, 1.0, 2.0)} | {K - 1})
        ims = [Image.open(vdir / "frames" / m / "hero" / f"{k:04d}_35.png").convert("RGB") for k in ks]
        x0, y0, x1, y1 = bbox(ims)
        pad = 30
        x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(ims[0].width, x1 + pad), min(ims[0].height, y1 + pad)
        crops = [im.crop((x0, y0, x1, y1)) for im in ims]
        sc = CH / (y1 - y0)
        crops = [c.resize((int(c.width * sc), CH), Image.LANCZOS) for c in crops]
        last = Image.open(vdir / "frames" / m / "map_35" / f"{K - 1:04d}.jpg").convert("RGB")
        W, H = last.size
        leg = last.crop((W - 700, H - 262, W - 30, H - 40))
        win = json.loads(next(ln for ln in open("/data/relcfd/chayo/physmorph_v2/output/gpu/d159/read/window.jsonl")
                              if f'"{m}_tag_L97"' in ln))
        t_arr = (win["arrival_window"] + 1) / 6.0
        caps = []
        for k in ks:
            t = tt[k]
            if k == K - 1:
                caps.append(f"end (t = {t:.1f} s)")
            else:
                caps.append(f"t = {t:.1f} s" + (" (arriving)" if t < t_arr else " (arrived)"))
        rows.append((m, crops, leg, caps, t_arr))
    gap, margin = 18, 40
    rw = max(sum(c.width for c in crops) + gap * (len(crops) - 1) + 30 + leg.width for _, crops, leg, _, _ in rows)
    Wc = margin * 2 + rw
    Hc = 150 + len(rows) * (CH + 90) + 60
    canvas = backdrop(Wc, Hc)
    dr = ImageDraw.Draw(canvas)
    dr.text((margin, 30), "Where does the render change the shape?", font=font(40), fill=INK)
    dr.text((margin, 86), "dFᴄ + layer u + render loss (full) against its twin dFᴄ + layer u (λ = 0) at the same "
            "moment: blue = the render pulled the surface in, red = pushed it out, grey = within the run-to-run noise",
            font=font(22), fill=INK2)
    y = 150
    for m, crops, leg, caps, t_arr in rows:
        dr.text((margin, y), f"{m} 300k   (arrival: 90 % of the motion done at {t_arr:.1f} s)", font=font(24), fill=INK)
        x = margin
        for c, cap in zip(crops, caps):
            dr.text((x + 6, y + 40), cap, font=font(20), fill=INK2)
            canvas.paste(c, (x, y + 70))
            x += c.width + gap
        canvas.paste(leg, (x + 12, y + 70 + (CH - leg.height) // 2))
        y += CH + 90
    dr.text((margin, Hc - 44), "front view (az 35°, el 18°), the deliverable's display surface in the studio look; colour "
            "averaged over 1.5 pitches and ±0.1 s; each mesh its own fixed colour scale; images flipped left-right "
            "(the studio camera mirrors); tag freeze-2026-10-15, 300k, seed 97", font=font(16), fill=INK3)
    canvas.save(outp)
    print("wrote", outp, canvas.size)


if __name__ == "__main__":
    main()
