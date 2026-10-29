"""d173_summary.py OUT_MD -- D173's master summary (CPU): (1) the real scales' representability table (d173_probe.py), (2) the
colour edge widths (d173_edge.py) and the deliverables, (3) the textured duck (d173_texprobe.py's ceiling, d173_texread.py's runs)."""
import glob
import json
import os
import sys
import time
from pathlib import Path

O = Path("/data/relcfd/chayo/physmorph_v2/output/gpu/d173")
T300_MIN, MEM300_GB, DISCS300 = 25.0, 20.0, 174823     # the 300k dragon run at r = 3, 188 px: ~25 min, ~20 GB, 174,823 discs


def f(x, p=3):
    return "-" if x is None else f"{x:.{p}f}"


def main(out_md):
    L = ["# D173: the dragon's real scales (representability), the colour edge, a real texture", ""]
    rows = []
    for p in sorted(glob.glob(str(O / "probe/probe_*.jsonl"))):
        rows += [json.loads(x) for x in open(p) if x.strip()]
    L += ["## 1. Can a perfect sample + the exterior field + the picture carry the real scales? (gate: corr >= 0.25)", "",
          "| N | surface density F | field radius (pitches) | picture px | pixel (pitches) | layer corr | field corr | picture corr | "
          "discs | interior discs | probe card memory (GB) | run est.: minutes / card GB |", "|" + "---|" * 12]
    best = None
    for r in rows:
        pic = r.get("picture", {}).get("corr")
        scale = 0.6 * r["N"] / 300000 + 0.4 * (r.get("discs") or DISCS300) / DISCS300
        est = f"{T300_MIN * scale:.0f} / {MEM300_GB * scale:.0f}"
        L.append(f"| {r['N'] // 1000}k | {r['F']:g} | {r['r']:g} | {r['res']} | {f(r.get('picture', {}).get('pixel_pitch'), 2)} | "
                 f"{f(r['layer']['corr'])} | {f(r.get('field', {}).get('corr'))} | {f(pic)} | {r.get('discs', '-')} | "
                 f"{f(r.get('discs_deeper_than_3a'), 2)} | {f(r.get('mem_peak_gb'), 1)} | {est} |")
        if pic is not None and pic >= 0.25 and (best is None or scale < best[1]):
            best = (r, scale)
    L += ["", "Readings: layer = the outer-layer particles' offsets from the smoothed canvas against the scale height H (high-pass 4 "
          "pitches); field = the exterior discs' offsets (render/exterior.make_field zb, Lattice.discs at target.py's lattice); "
          "picture = the loss's shading pictures of those discs against the true surfaces' (D159 mesh shells) in the 18 views, "
          "(sample - smooth shell) vs (detailed shell - smooth shell), high-passed at 4 pitches. Interior discs = the share deeper "
          "than 3 pitches below the canvas (the field's zero set breaking into pockets). Run estimates scale the 300k r = 3 run "
          "(25 min, 20 GB) by 0.6 N / 300k + 0.4 discs / 174,823 (D96: the exterior searches are about 0.4 of a window).",
          "", ("**Cheapest passing setting: " + (f"N {best[0]['N'] // 1000}k, F {best[0]['F']:g}, radius {best[0]['r']:g}, {best[0]['res']} px "
               f"(picture corr {best[0]['picture']['corr']:.3f}, est. {T300_MIN * best[1]:.0f} min, {MEM300_GB * best[1]:.0f} GB)." if best
               else "none of the settings measured reaches corr 0.25.") + "**"), ""]
    # 2. the colour edge
    E = []
    for p in [O / "edge/edge.jsonl", O / "auto/cmb300/edge.jsonl"]:
        if p.exists():
            E += [json.loads(x) for x in open(p) if x.strip()]
    L += ["## 2. The colour edge (10-90 % rise of the red share across the band plane, pitches)", "",
          "| state | particles (identity) | displayed discs, kernel (R = 3 pitches) | displayed discs, nearest particle | scatter > 2 pitches |",
          "|---|---|---|---|---|"]
    for e in E:
        L.append(f"| {e['name'].replace('@tgt', ' target paint')} | {f(e['particles']['width'], 2)} | {f(e['discs_k3']['width'], 2)} | "
                 f"{f(e['discs_nn']['width'], 2)} | {f(e['scatter_2a'], 3)} |")
    L += ["", "Twins: no 10-90 % crossing near the plane (their boundary sits 6+ pitches off, scatter 8 %). Where the width comes "
          "from: (a) the display / loss colour kernel, render/exterior.py:400-413 found_disc_albedo (weights (1 - d^2/R^2)^3 over the "
          "particles within R = the field radius, 3 pitches, + one pitch of skin; the radius from cfg.exterior_radius, "
          "pipeline/target.py:206-209; the deliverable renderer the same, tmp/d170/d170_studio.py surface()): it widens even the "
          "target's perfectly sharp paint to 2.1-2.3 pitches; (b) the particles' own layout (identity scatter of the morph): 2.2-2.8 "
          "pitches for Rx at 40k / 300k; displayed width ~ sqrt(a^2 + b^2). Fix made (display, for the target and both arms alike): "
          "each disc shows its nearest particle's own colour (tmp/d173/d173_studio.py +colour=nn): the target edge falls to 1.2-1.3 "
          "pitches (the sampling floor), the arm shows its own 2.1-2.3; nothing hides the physics (the colour is a particle's "
          "identity). The loss still reads the kernel (unchanged; a narrower colour kernel in the loss is the next definition to test, "
          "the particle mixing being its own half).", ""]
    rr, cc = O / "auto/cmb300/read_relief.json", O / "auto/cmb300/read_colour.jsonl"
    if rr.exists() and cc.exists():
        rd = json.load(open(rr))
        cA = sum(v["relief"]["A"] for k, v in rd["refs"].items() if k.startswith("relief_s")) / max(1, sum(1 for k in rd["refs"] if k.startswith("relief_s")))
        col = {json.loads(x)["name"]: json.loads(x) for x in open(cc) if x.strip()}
        L += ["", "Combined relief + paint at 300k (job 201; D161's / D160's fast readers):", "",
              "| arm | relief share end | colour r end | mis end |", "|---|---|---|---|"]
        mt, fl = col["cmb300k_T_97"]["end"]["mis"], col["cmb300k_Rx_97"]["floor"]
        for lab, nm, arm in (("full97", "cmb300k_Rx_97", "Rx (full)"), ("twin97", "cmb300k_T_97", "twin")):
            ser = [q for q in rd["runs"][lab]["series"] if q["relief"].get("A") is not None]
            m = col[nm]["end"]["mis"]
            L.append(f"| {arm} | {ser[-1]['relief']['A'] / cA:.2f} | {(mt - m) / (mt - fl):.2f} | {m:.3f} |")
        L.append("")
    for d in sorted(glob.glob(str(O / "deliver/*"))):
        files = [Path(x).relative_to(O) for x in glob.glob(d + "/**/*", recursive=True) if x.endswith((".png", ".mp4", ".jpg"))]
        if files:
            L.append(f"- deliverables {Path(d).name}: " + ", ".join(str(x) for x in sorted(files)[:8]))
    # 3. texture
    L += ["", "## 3. A real texture (Keenan Crane's Bob, bob_diffuse.png, on assets/bob.obj)", ""]
    tp = O / "texprobe/texprobe.jsonl"
    if tp.exists():
        L += ["| N | pixel (pitches) | texture half-correlation (pitches) | ceiling PSNR / SSIM held-out (kernel) | (nearest) | flat colour PSNR / SSIM |",
              "|---|---|---|---|---|---|"]
        for x in open(tp):
            r = json.loads(x)
            L.append(f"| {r['N'] // 1000}k | {r['pixel_pitch']:.2f} | {r['texture_half_corr_pitch']} | {r['held_kernel']['psnr']:.2f} / "
                     f"{r['held_kernel']['ssim']:.3f} | {r['held_nn']['psnr']:.2f} / {r['held_nn']['ssim']:.3f} | "
                     f"{r['held_flat_colour']['psnr']:.2f} / {r['held_flat_colour']['ssim']:.3f} |")
    L += ["", "The source: each source particle carries one of the texture's colours, the texture projected radially from the "
          "source's centroid (appearance.TextureMap.radial); the target's samples take the image's colour at their nearest surface "
          "point. The twin does not get the layout for free: every physics term reads only the set of positions (D160), so which "
          "particle ends where follows the transport, not the paint; the radial map is a fixed layout the transport does not follow, "
          "and the colours land wherever it sends them. Only the render's colour pictures can move the material to its colour.", ""]
    for j in ("tex40", "tex300", "cmb300"):
        p = O / f"auto/{j}/summary.md"
        if p.exists():
            L += ["---", f"<!-- {p} -->", p.read_text().strip(), ""]
    for p in sorted(glob.glob(str(O / "auto/*/summary.md"))):
        if Path(p).parent.name not in ("tex40", "tex300", "cmb300"):
            L += ["---", f"<!-- {p} -->", open(p).read().strip(), ""]
    L.append(f"\nWritten {time.strftime('%Y-%m-%d %H:%M %Z')}. Code: repo_d173 = wip/d173 98303b79 (the texture appearance; D172's "
             "--render_split, the render_x paint fix).")
    tmp = Path(out_md + f".{os.getpid()}.tmp")
    tmp.write_text("\n".join(L) + "\n")
    os.replace(tmp, out_md)


if __name__ == "__main__":
    main(sys.argv[1])
