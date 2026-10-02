"""compare_ref.py REF_PNG FRAME_PNG OUT_JPG x0,y0,x1,y1 [x0,y0,x1,y1 ...] — the renderer's picture of the target sample
(left) beside a morph frame (right), cropped at native 4K resolution, one row per box; and the edge sharpness of both:
the 10-90 % rise width of the silhouette (pixels) sampled along horizontal scan lines of each box."""
import sys
import numpy as np
from PIL import Image, ImageDraw

ref, frm = Image.open(sys.argv[1]).convert("RGB"), Image.open(sys.argv[2]).convert("RGB")
boxes = [[int(v) for v in b.split(",")] for b in sys.argv[4:]]
bg = np.asarray(ref, np.float32)[:40, :40].reshape(-1, 3).mean(0)


def edge_widths(img, box):
    a = np.abs(np.asarray(img.crop(box), np.float32) - bg).sum(2)          # distance from the background colour
    widths = []
    for row in a[:: max(1, a.shape[0] // 60)]:
        hi = row.max()
        if hi < 30:
            continue
        on = row > 0.9 * min(hi, np.percentile(a, 90))
        idx = np.nonzero(on)[0]
        if len(idx) == 0:
            continue
        level = row[idx[0]: idx[0] + 8].mean()
        left = idx[0]
        j = left
        while j > 0 and row[j] > 0.1 * level:
            j -= 1
        widths.append(left - j)
    return widths


rows = []
for b in boxes:
    w, h = b[2] - b[0], b[3] - b[1]
    t = Image.new("RGB", (2 * w + 6, h), (20, 20, 20))
    t.paste(ref.crop(b), (0, 0)); t.paste(frm.crop(b), (w + 6, 0))
    d = ImageDraw.Draw(t)
    d.text((6, 4), "target sample through the renderer", fill=(255, 255, 0)); d.text((w + 12, 4), "morph frame", fill=(255, 255, 0))
    rows.append(t)
    wr, wf = edge_widths(ref, b), edge_widths(frm, b)
    print(f"box {b}: left silhouette edge 10-90 % width, median px (scan lines): target sample {np.median(wr) if wr else float('nan'):.1f} ({len(wr)}), "
          f"morph frame {np.median(wf) if wf else float('nan'):.1f} ({len(wf)}); p90 {np.percentile(wr, 90) if wr else float('nan'):.1f} / {np.percentile(wf, 90) if wf else float('nan'):.1f}")
W = max(r.width for r in rows)
s = Image.new("RGB", (W, sum(r.height for r in rows) + 6 * (len(rows) - 1)), (20, 20, 20))
y = 0
for r in rows:
    s.paste(r, (0, y)); y += r.height + 6
s.save(sys.argv[3], quality=93)
print("wrote", sys.argv[3], s.size)
