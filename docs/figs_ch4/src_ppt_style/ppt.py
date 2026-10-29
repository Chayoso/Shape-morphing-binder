"""ppt.py -- FIGS_PPT (2026-10-23): the user's PowerPoint figure style for Chapter 4, as PIL helpers (hyde06 only).
White ground; white cards and strips with a soft grey drop shadow below-right; pop-out rows (bodies cut from the
studio backdrop overflow a shorter strip); bold serif column headers coloured per arm; italic rotated row labels;
red bold-italic callouts; every text in Times New Roman (tmp/fonts)."""
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy import ndimage

sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/fonts")
import tnr  # noqa: E402

GREY, BLUE, GREEN, RED, NAVY, INK = "#6b6b6b", "#3a78b5", "#2e7d4f", "#c0161b", "#1f2d4d", "#111111"
ARM = {"only": GREY, "twin": BLUE, "full": GREEN, "ch3": GREY, "ref": GREY}


def font(size, bold=False, italic=False):
    return ImageFont.truetype(tnr.path(bold, italic), int(size))


def canvas(w, h):
    return Image.new("RGBA", (int(w), int(h)), (255, 255, 255, 255))


def shadow_card(img, box, radius=14, blur=16, offset=6, opacity=0.42, fill=(255, 255, 255, 255), outline=None):
    """A white (rounded) card at box with a soft drop shadow below-right."""
    x0, y0, x1, y1 = [int(v) for v in box]
    pad = 3 * blur
    sh = Image.new("L", (x1 - x0 + 2 * pad, y1 - y0 + 2 * pad), 0)
    ImageDraw.Draw(sh).rounded_rectangle((pad, pad, pad + x1 - x0, pad + y1 - y0), radius, fill=int(255 * opacity))
    sh = sh.filter(ImageFilter.GaussianBlur(blur))
    layer = Image.new("RGBA", sh.size, (60, 60, 60, 0))
    layer.putalpha(sh)
    img.alpha_composite(layer, (x0 - pad + offset, y0 - pad + offset))
    ImageDraw.Draw(img).rounded_rectangle((x0, y0, x1, y1), radius, fill=fill, outline=outline,
                                          width=2 if outline else 0)


def rich(draw, xy, parts, size, fill=INK, anchor="la", bold=False, italic=False):
    """Draw text parts: str or (str, 'sub'|'it'|'bf'|'sup'); xy is the left baseline unless anchor='mm'/'ma' etc. (centre
    handled by measuring first). Returns the width."""
    segs = []
    for p in parts if isinstance(parts, (list, tuple)) else [parts]:
        s, kind = (p, "") if isinstance(p, str) else p
        f = font(size * (0.68 if kind in ("sub", "sup") else 1), bold=bold or kind == "bf", italic=italic or kind == "it")
        dy = 0.22 * size if kind == "sub" else (-0.38 * size if kind == "sup" else 0)
        segs.append((s, f, dy))
    w = sum(draw.textlength(s, font=f) for s, f, _ in segs)
    x, y = xy
    if anchor[0] == "m":
        x -= w / 2
    elif anchor[0] == "r":
        x -= w
    for s, f, dy in segs:
        draw.text((x, y + dy), s, font=f, fill=fill, anchor="ls")
        x += draw.textlength(s, font=f)
    return w


def arm_parts(name):
    """The user's arm names, with dF_c typeset (italic d? no: 'dF' then a subscript c)."""
    out = []
    for i, chunk in enumerate(name.split("dF_c")):
        if i:
            out += ["dF", ("c", "sub")]
        if chunk:
            out.append(chunk)
    return out


def vlabel(img, center, text, size, fill=INK, italic=True, bold=False):
    """A rotated (90 degrees, reading upwards) label centred at center."""
    f = font(size, bold=bold, italic=italic)
    d = ImageDraw.Draw(Image.new("L", (1, 1)))
    w = int(d.textlength(text, font=f)) + 8
    h = int(size * 1.4)
    t = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    ImageDraw.Draw(t).text((4, int(size * 1.05)), text, font=f, fill=fill, anchor="ls")
    t = t.rotate(90, expand=True)
    img.alpha_composite(t, (int(center[0] - t.width / 2), int(center[1] - t.height / 2)))


def cut_body(tile, thresh=5, erode=1, feather=1.0):
    """RGBA of the body in a studio tile: the difference to the per-row backdrop (its left and right edges), the largest
    component, holes filled, eroded and feathered."""
    a = np.asarray(tile.convert("RGB")).astype(float)
    e = max(4, a.shape[1] // 25)
    bg = np.median(np.concatenate([a[:, :e], a[:, -e:]], 1), axis=1)
    m = np.abs(a - bg[:, None, :]).max(-1) > thresh
    # the backdrop is bluish (B - R ~ 11), the light-grey body neutral: the hue separates the dark body from the backdrop
    m |= np.abs((a[..., 2] - a[..., 0]) - (bg[:, 2] - bg[:, 0])[:, None]) > 3.5
    m = ndimage.binary_closing(ndimage.binary_opening(m, iterations=1), iterations=4)
    lab, n = ndimage.label(m)
    if n == 0:
        return None
    sizes = np.bincount(lab.ravel())[1:]
    keep = np.flatnonzero(sizes >= 0.02 * sizes.max()) + 1
    m = ndimage.binary_fill_holes(np.isin(lab, keep))
    if erode:
        m = ndimage.binary_erosion(m, iterations=erode)
    al = ndimage.gaussian_filter(m.astype(float), feather) * 255
    rgba = np.dstack([a, al]).clip(0, 255).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def contact_shadow(img, body_box, alpha_mask, strength=0.35):
    """A soft elliptical contact shadow under a pasted body (alpha_mask: the body's alpha, in img coordinates box)."""
    x0, y0, x1, y1 = body_box
    m = np.asarray(alpha_mask) > 40
    if not m.any():
        return
    rows = np.where(m.any(1))[0]; cols = np.where(m.any(0))[0]
    yb = y0 + rows[-1]; xa, xb = x0 + cols[0], x0 + cols[-1]
    w = xb - xa
    sh = Image.new("L", img.size, 0)
    ImageDraw.Draw(sh).ellipse((xa + .1 * w, yb - .035 * w, xb - .1 * w, yb + .05 * w), fill=int(255 * strength))
    sh = sh.filter(ImageFilter.GaussianBlur(max(4, .04 * w)))
    layer = Image.new("RGBA", img.size, (40, 40, 45, 0))
    layer.putalpha(sh)
    img.alpha_composite(layer)


def tiles(img, white=245, frac=0.985, min_size=120):
    """The tile grid of a composed figure: row bands and, per band, column spans, separated by near-white lines."""
    a = np.asarray(img.convert("L")).astype(float)

    def spans(mask_line, n):
        out, s = [], None
        for i in range(n):
            if not mask_line[i] and s is None:
                s = i
            if mask_line[i] and s is not None:
                if i - s >= min_size:
                    out.append((s, i))
                s = None
        if s is not None and n - s >= min_size:
            out.append((s, n))
        return out
    wr = (a > white).mean(1) > frac
    rows = spans(wr, a.shape[0])
    grid = []
    for r0, r1 in rows:
        wc = (a[r0:r1] > white).mean(0) > frac
        grid.append(((r0, r1), spans(wc, a.shape[1])))
    return grid


def save(img, path, quality=93):
    img = img.convert("RGB")
    if path.endswith(".png"):
        img.save(path, optimize=True)
    else:
        img.save(path, quality=quality)
    print("wrote", path, img.size)


def _italic_u(parts):
    out = []
    for p in parts:
        if isinstance(p, str) and "layer u" in p:
            a, b = p.split("layer u", 1)
            out += [a + "layer ", ("u", "it")] + ([b] if b else [])
        else:
            out.append(p)
    return out


_arm_parts_plain = arm_parts


def arm_parts(name):  # noqa: F811  (the arm names with dF_c typeset and the layer's u in italics)
    return _italic_u(_arm_parts_plain(name))


# 2026-10-23 (the user): no drop shadows and no shading in the figures; a strip that only carried a shadow is dropped,
# so the layout is plain on white. PLAIN = True turns the card/strip and the contact shadow into no-ops.
PLAIN = True
_shadow_card, _contact_shadow = shadow_card, contact_shadow


def shadow_card(img, box, *a, **k):  # noqa: F811
    if not PLAIN:
        _shadow_card(img, box, *a, **k)


def contact_shadow(img, body_box, alpha_mask, strength=0.35):  # noqa: F811
    if not PLAIN:
        _contact_shadow(img, body_box, alpha_mask, strength)
