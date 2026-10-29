"""d173_summary.py OUT_MD -- D173's master summary (CPU): (1) the real scales' representability table (d173_probe.py), (2) the
colour edge widths (d173_edge.py) and the deliverables, (3) the textured duck (d173_texprobe.py's ceiling, d173_texread.py's runs),
(4) the three-arm attribution on the stylized scales E, (5) option C (the dragon's own scales magnified), (6) Bob with the
nearest-particle colour in the loss."""
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
          "| N (interior) | band density F (surface spacing / F^(1/3)) | particles total | field radius (field pitches) | picture px | pixel (base pitches) | layer corr | field corr | picture corr | "
          "discs | interior discs | probe card memory (GB) | run est.: minutes / card GB |", "|" + "---|" * 13]
    best = None
    for r in rows:
        pic = r.get("picture", {}).get("corr")
        scale = 0.6 * r["n_particles"] / 300000 + 0.4 * (r.get("discs") or DISCS300) / DISCS300
        est = f"{T300_MIN * scale:.0f} / {MEM300_GB * scale:.0f}"
        L.append(f"| {r['N'] // 1000}k | {r['F']:g} | {r['n_particles'] // 1000}k | {r['r']:g} | {r['res']} | {f(r.get('picture', {}).get('pixel_pitch'), 2)} | "
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
    for p in [O / "edge2/edge.jsonl", O / "auto/colour/edge.jsonl"]:
        if p.exists():
            E += [json.loads(x) for x in open(p) if x.strip()]
    L += ["## 2. The colour edge (10-90 % rise of the red share across the band plane, pitches)", "",
          "| state | particles (identity): logistic fit (running-max profile) | displayed discs, kernel R = 3 pitches | displayed discs, nearest particle | scatter > 2 pitches |",
          "|---|---|---|---|---|"]
    for e in E:
        L.append(f"| {e['name'].replace('@tgt', ' target paint')} | {f(e['particles'].get('logistic'), 2)} ({f(e['particles']['width'], 2)}) | {f(e['discs_k3'].get('logistic'), 2)} | "
                 f"{f(e['discs_nn'].get('logistic'), 2)} | {f(e['scatter_2a'], 3)} |")
    L += ["", "Twins: no 10-90 % crossing near the plane (their boundary sits 6+ pitches off, scatter 8 %). Where the width comes "
          "from: (a) the display / loss colour kernel, render/exterior.py:400-413 found_disc_albedo (weights (1 - d^2/R^2)^3 over the "
          "particles within R = the field radius, 3 pitches, + one pitch of skin; the radius from cfg.exterior_radius, "
          "pipeline/target.py:206-209; the deliverable renderer the same, tmp/d170/d170_studio.py surface()): it widens even the "
          "target's perfectly sharp paint to 2.1-2.3 pitches; (b) the particles' own layout (identity scatter of the morph): 2.2-2.8 "
          "pitches for Rx at 40k / 300k; displayed width ~ sqrt(a^2 + b^2). Fix made (display, for the target and both arms alike): "
          "each disc shows its nearest particle's own colour (tmp/d173/d173_studio.py +colour=nn): the target edge falls to 1.2-1.3 "
          "pitches (the sampling floor), the arm shows its own 2.1-2.3; nothing hides the physics (the colour is a particle's "
          "identity). The loss: --colour_kernel nearest (wip/d173 0ac73e72; render/exterior.nearest_weights, a softmax of -d^2 / (pitch / 2)^2 "
          "over the disc's particles, in the target's pictures and the body's discs alike, differentiable in the positions that compete to be "
          "nearest; the rollout untouched): the Rxn rows. Widths are logistic fits (robust to strays; the running-max profile in brackets). "
          "", ""]
    cn = O / "auto/colour/read_colour.jsonl"
    if cn.exists():
        col = {json.loads(x)["name"]: json.loads(x) for x in open(cn) if x.strip()}
        L += ["", "The loss on the nearest-particle colour (--colour_kernel nearest, Rxn; job 205) against Rx (the 3-pitch kernel):", "",
              "| run | colour r end (D160, twin-referenced) | mis end | Rx's r (D166 / D170) |", "|---|---|---|---|"]
        ref_r = {"band40k_Rxn_97": "0.93", "band40k_Rxn_99": "0.92", "band300k_Rxn_97": "0.92"}
        for nm, tw in (("band40k_Rxn_97", "band40k_T_97"), ("band40k_Rxn_99", "band40k_T_99"), ("band300k_Rxn_97", "band300k_T_97")):
            if nm in col and tw in col:
                mt, fl, m = col[tw]["end"]["mis"], col[nm]["floor"], col[nm]["end"]["mis"]
                L.append(f"| {nm} | {(mt - m) / (mt - fl):.2f} | {m:.3f} | {ref_r[nm]} |")
        L.append("")
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
        if Path(d).name.startswith(("attr", "magC")):
            continue
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

    def part(names):
        out = []
        for j in names:
            p = O / f"auto/{j}/summary.md"
            if p.exists():
                out += ["---", f"<!-- {p} -->", p.read_text().strip(), ""]
            for s in sorted(glob.glob(str(O / f"deliver/{j}/**/*.jpg"), recursive=True) + glob.glob(str(O / f"deliver/{j}/*.mp4"))):
                out.append(f"- deliverable: {Path(s).relative_to(O)}")
        return out + [""]
    # 4. the attribution proof on the stylized scales E
    L += ["", "## 4. Attribution: do the scales come from the render? (stylized scales E, 300k, three arms)", "",
          "Three arms share the 3D target (the smoothed dragon: the physics objective, the proximity term and the relaxation never "
          "see a scale) and differ only in the pictures: (1) the twin (lambda = 0, no pictures act), (2) Hu (D172: lambda held on u, "
          "the rule on dF_c) with pictures of the dragon carrying the stylized scales E, (3) the same Hu with pictures of the "
          "smoothed dragon itself (the render acts at the same weights but carries no scale). Scales that appear in (2) and in "
          "neither (1) nor (3) are the render's alone. (4) at seed 97: Hu with u off (--u_off), the render reaching dF_c alone at the "
          "rule's weight: how much of the relief the stress increment writes on its own. Every table gives the render's share of "
          "each block's step (u, and dF_c after PCGrad with the weight as applied) at the start, the arrival and the stop.", ""]
    L += part(("attr97", "attr97_uoff", "attr99"))
    # 5. option C: the dragon's own scales, magnified
    L += ["## 5. Option C: the dragon's OWN scales, magnified K times (Hu, 300k, three arms)", "",
          "Construction (tmp/d173/d173_magscale.py, CPU): (1) the real relief H = the detailed Stanford dragon's signed offset from "
          "the Taubin-smoothed dragon along the smoothed surface's normal, at 1M points of the smoothed surface (ray cast both "
          "ways, the nearer hit, |H| <= 3 pitches); (2) a flat chart: D161's flank map (orthographic along the body's narrower "
          "horizontal axis onto its + side, the faces that see that side, front-most), H rastered on it at 0.2-pitch texels (the "
          "mean of the facing points per texel, the gaps filled linearly); (3) magnified K times about the chart's origin, "
          "H_K(u, v) = K H(u / K, v / K): the scales K times wider and K times taller, their shape and aspect kept (the central "
          "1/K^2 of the flank's pattern spread over the whole flank); (4) the picture mesh: the smoothed dragon, its flank-facing "
          "faces subdivided to edges below a third of a pitch, each vertex moved along its normal by m (H_K - mean), m the chart's "
          "support weight (facing ramp x front-most): mean height zero (volume kept), nothing moves off the flank, no step at its "
          "edge. Only the pictures are rendered from it (--render_tgt); the 3D target of all three arms is the smoothed dragon. "
          "Arms (1) and (3) are section 4's runs (the same seed, target and flags). Gate (as E's): the picture mesh's own perfect "
          "300k sample, read by D172's detail reader (the fast port, high-pass sigma 4K pitches), corr >= 0.25 (E: 0.28); the "
          "smallest K at 300k that passes is run.", ""]
    tbs = [(O / "assets/magC/probe_table2.txt", "4K"), (O / "assets/magC/probe_table4.txt", "4K"), (O / "assets/magC/probe_table3.txt", "4")]
    if tbs[0][0].exists():
        L += ["| K | N | high-pass s_bg (pitches) | perfect sample: share A / corr (draw 1) | corr (draw 2) | smoothed dragon's sample corr (floor) | canvas rms H (pitches) | region share |",
              "|---|---|---|---|---|---|---|---|"]
        for tb, sb in tbs:
            for x in (open(tb) if tb.exists() else []):
                w = x.split()
                if len(w) < 15:
                    L.append(f"| {w[1]} | {w[2]}k | {sb} | missing | | | | |")
                    continue
                L.append(f"| {w[1]} | {w[2]}k | {sb} | {w[4]} / {w[6]} | {w[8]} | {w[10]} | {w[12]} | {w[14]} |")
        vt = O / "assets/magC/variant.txt"
        pick = vt.read_text().split() if vt.exists() else []
        L += ["", "s_bg = 4K: the high-pass widened with the scales (they grow K times); the reader's region (the canvas within 2 "
              "s_bg of the pattern) then covers 84-100 % of the body. s_bg = 4 (the reader's default, section 1's gate) reads a "
              "smaller region (0.67) but cuts the magnified scales (their autocorrelation half-length 1.9-4.1 pitches) and reads "
              "lower, so the pick uses s_bg = 4K. What limits corr: the perfect sample's own scatter about the canvas (rms 0.35-0.40 "
              "pitch, the same for the smoothed dragon's sample) against the magnified relief's rms 0.05-0.10 pitch over the region "
              "(0.09-0.20 pitch on the flank): corr ~ A x rms H / rms h. The real relief is shallow (rms 0.07 pitch on the flank at "
              "300k) and shallower still at the chart's centre, so K = 2-3 (the user's range) reads 0.09-0.14 at 300k and 0.11-0.16 "
              "at 550k; K up to 8 was read to find where it crosses 0.25.",
              "", f"**Picked: K = {pick[1]} at 300k (s_bg {pick[2] if len(pick) > 2 else '4K'}).**" if pick else "**No K passes at 300k.**",
              "", "The first build's K = 2.5 mesh was named d173M2.5_300_dragon.obj; physmorph/sampling/orientation.py asset_name cuts a "
              "name at its first dot, so it was read without the dragon's rotation (canvas rms H 1.56 pitches, region 0.15: its corr "
              "0.56 measured the turn, not the scales). Renamed M25 and read again; no other D170-D173 mesh name carries a dot.", ""]
    L += part(("magC97", "magC99"))
    # 6. Bob with the nearest-particle colour in the loss
    L += ["## 6. Bob with --colour_kernel nearest (Rxn) against Rx and the twin", ""]
    L += part(("tex40n", "tex300n", "tex40t", "tex300t"))
    L += ["Rxt / Rxnt: Rx and Rxn rerun on the code that records each block's render share (the first runs predate it; the "
          "same flags; rerun_cmp.txt says whether the reruns are the first runs bit for bit).", ""]
    # 7. the stripes: a layout the twin cannot reach by accident
    L += ["## 7. Stripes: a colour layout the twin cannot reach by accident (the bunny, Rx + nearest colour vs the twin)", "",
          "Why: on the band scene the twin already looks nearly banded from the front, because the transport carries the "
          "source's top to the target's top (Cohen's kappa of the twin's own layout against the band paint 0.68, 0.78 on the "
          "outer layer). Stripes (appearance.py, --appearance stripes): three vertical stripes across the front view, the middle "
          "red (a third), the source's top third red (the same budget). Checked before any run (tmp/d173/d173_layout_check.py, "
          "on the band scene's twins at 40k s97 / s99 and 300k s97): the twin's layout meets the stripes at kappa 0.00-0.01 "
          "(0.02-0.03 on the outer layer; a 2 x 2 checker read -0.02 / 0.05-0.10), so the stripes the full arm shows are the "
          "render's.", ""]
    L += part(("strp40", "strp300"))
    done = {"tex40", "tex300", "cmb300", "attr97", "attr99", "magC97", "magC99", "tex40n", "tex300n", "tex40t", "tex300t",
            "strp40", "strp300"}
    for p in sorted(glob.glob(str(O / "auto/*/summary.md"))):
        if Path(p).parent.name not in done:
            L += ["---", f"<!-- {p} -->", open(p).read().strip(), ""]
    L.append(f"\nWritten {time.strftime('%Y-%m-%d %H:%M %Z')}. Code: repo_d173 = wip/d173 (the texture appearance, "
             "--colour_kernel nearest, the stripes scene, each block's render share recorded on every run; D172's "
             "--render_split, the render_x paint fix).")
    tmp = Path(out_md + f".{os.getpid()}.tmp")
    tmp.write_text("\n".join(L) + "\n")
    os.replace(tmp, out_md)


if __name__ == "__main__":
    main(sys.argv[1])
