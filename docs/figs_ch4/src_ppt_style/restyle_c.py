"""restyle_c.py IN_DIR OUT_DIR -- FIGS_PPT (2026-10-23): the letters, colour, held-share and stamp figures in the user's
PowerPoint style, rebuilt from their composed JPEGs (the tiles are cut out unchanged; headers, labels, cards and
shadows are drawn anew; every text in Times New Roman). Rows of renders on the studio backdrop become pop-out rows
(the body cut from the backdrop, overflowing a shorter white strip); close-up rows (the plaque) become white strip cards."""
import sys
from pathlib import Path

from PIL import Image, ImageDraw

import ppt

IN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)


def grid_figure(cells, headers, row_groups, popout, tile_w, tile_h, gap=22, left=120, top=118, hsize=40, lsize=38,
                strip_pad=16, row_gap=34, heads_wrap=None):
    R, C = len(cells), len(cells[0])
    pitch = tile_h + 2 * strip_pad + row_gap
    W = left + C * tile_w + (C - 1) * gap + strip_pad + 40
    H = top + R * pitch + 10
    img = ppt.canvas(W, H)
    d = ImageDraw.Draw(img)
    for c, (parts, col) in enumerate(headers):
        cx = left + c * (tile_w + gap) + tile_w / 2
        lines = parts if isinstance(parts[0], list) else [parts]
        for k, ln in enumerate(lines):
            ppt.rich(d, (cx, top - 34 - (len(lines) - 1 - k) * hsize * 1.12), ln, hsize, fill=col, anchor="m", bold=True)
    for r in range(R):
        y0 = top + r * pitch + strip_pad
        x_end = left + C * tile_w + (C - 1) * gap
        if popout:
            ppt.shadow_card(img, (left - strip_pad, y0 + .30 * tile_h, x_end + strip_pad, y0 + .86 * tile_h), radius=10)
            for c in range(C):
                body = ppt.cut_body(cells[r][c])
                if body is None:
                    continue
                x = left + c * (tile_w + gap)
                ppt.contact_shadow(img, (x, y0, x + tile_w, y0 + tile_h), body.split()[3])
                img.alpha_composite(body, (int(x), int(y0)))
        else:
            ppt.shadow_card(img, (left - strip_pad, y0 - strip_pad, x_end + strip_pad, y0 + tile_h + strip_pad), radius=10)
            for c in range(C):
                img.alpha_composite(cells[r][c].convert("RGBA"), (int(left + c * (tile_w + gap)), int(y0)))
    for text, ra, rb, col in row_groups:
        ya = top + ra * pitch + strip_pad
        yb = top + rb * pitch + strip_pad + tile_h
        ppt.vlabel(img, (left - strip_pad - lsize * 0.95, (ya + yb) / 2), text, lsize, fill=col)
    return img


def crop(im, box):
    return im.crop(tuple(int(v) for v in box))


# ------------------------------------------------------------------ letters (4 x 4 close-ups of the plaque)
im = Image.open(IN / "fig_render_letters.jpg")
rows = [(80, 485), (491, 896), (906, 1311), (1317, 1722)]
cols = [(64, 784), (790, 1510), (1516, 2236), (2242, 2962)]
cells = [[crop(im, (c0, r0, c1, r1)) for c0, c1 in cols] for r0, r1 in rows]
heads = [(["Target pictures (relief mesh)"], ppt.GREY), (ppt.arm_parts("dF_c + layer u"), ppt.BLUE),
         ([ppt.arm_parts("dF_c + layer u + render loss (full)"), ["render weight held"]], ppt.GREEN),
         (["Lettered mesh sampled at 300k"], ppt.GREY)]
fig = grid_figure(cells, heads, [("SIGGRAPH (front)", 0, 1, ppt.INK), ("VANDERBILT (back)", 2, 3, ppt.INK)],
                  False, 720, 405, top=150, row_gap=20)
ppt.save(fig, str(OUT / "fig_render_letters.jpg"))

# ------------------------------------------------------------------ colour (2 views x 3, renders on the studio backdrop)
im = Image.open(IN / "fig_render_colour.jpg")
cells = [[crop(im, (c * 960, 70 + r * 540, (c + 1) * 960, 70 + (r + 1) * 540)) for c in range(3)] for r in range(2)]
heads = [([ppt.arm_parts("dF_c + layer u + render loss (full)"), ["colour weight held"]], ppt.GREEN),
         (ppt.arm_parts("dF_c + layer u"), ppt.BLUE), (["Target paint"], ppt.GREY)]
fig = grid_figure(cells, heads, [("front", 0, 0, ppt.INK), ("back", 1, 1, ppt.INK)], True, 960, 540, top=150)
ppt.save(fig, str(OUT / "fig_render_colour.jpg"))

# ------------------------------------------------------------------ held share (2 x 2, coloured bodies on the backdrop)
im = Image.open(IN / "fig_render_held_share.jpg")
rows = [(76, 466), (472, 862)]
cols = [(64, 584), (590, 1110)]
cells = [[crop(im, (c0, r0, c1, r1)) for c0, c1 in cols] for r0, r1 in rows]
heads = [([["full method,"], ["λ by the rule"]], ppt.GREEN), ([["full method,"], ["render weight held"]], ppt.BLUE)]
fig = grid_figure(cells, heads, [("t = 1.2 s", 0, 0, ppt.INK), ("t = 5.0 s", 1, 1, ppt.INK)], True, 520, 390,
                  top=120, hsize=34, lsize=32, left=96)
ppt.save(fig, str(OUT / "fig_render_held_share.jpg"))

# ------------------------------------------------------------------ stamp (two panels of close-ups, then the depth maps)
im = Image.open(IN / "fig_render_stamp.jpg")
mid = 1504
top_cells = [[crop(im, (64, 76, mid - 3, 417)), crop(im, (mid + 3, 76, 2944, 417))],
             [crop(im, (64, 423, mid - 3, 764)), crop(im, (mid + 3, 423, 2944, 764))]]
tw = top_cells[0][0].width
part1 = grid_figure(top_cells, [(["Plaque face, settled (raking light)"], ppt.INK),
                                (["Print in the soft slab, mirrored (500k)"], ppt.INK)],
                    [("full method", 0, 0, ppt.GREEN), ("twin + decal", 1, 1, ppt.BLUE)], False, tw, 341, top=96,
                    row_gap=20)
bot = [[crop(im, (64, 858, 1484, 1122)), crop(im, (1524, 858, 2944, 1122))]]
part2 = grid_figure([[c.resize((tw, int(c.height * tw / c.width))) for c in bot[0]]],
                    [(ppt.arm_parts("dF_c + layer u + render loss (full)") + [": depth of the print (300k)"], ppt.GREEN),
                     (ppt.arm_parts("dF_c + layer u") + [" + decal: depth of the print (300k)"], ppt.BLUE)],
                    [], False, tw, int(264 * tw / 1420), top=96, row_gap=10)
fig = ppt.canvas(max(part1.width, part2.width), part1.height + part2.height)
fig.alpha_composite(part1, (0, 0))
fig.alpha_composite(part2, (0, part1.height))
ppt.save(fig, str(OUT / "fig_render_stamp.jpg"))
