"""gen_a6.py MESHES_FROM_QUEUE -- D145 amendment 6 (pre-registered 2026-10-15): the ablations of the tag's new parts, one
part off at a time, everything else the tag's defaults. Printed as queue lines (gen_posttag.py puts them right after
B1 seed 97). Arms (names: letters only, so band145.py reads ARM + seed):
  NRD  --no-relax_driven     D146: the layer relaxation in every step again (not only the driven half)
  NUC  --no-u_adam_carry     D146b: u's Adam moments fresh every window again
  NSS  --no-spacing_symmetric  D149: each particle's own 16 nearest for the minimum spacing again
  NFX  the floating-Gaussian fix off: only when the tag carries it, its off-flag given as POSTTAG_FX_FLAG
300k: bunny and dragon x seeds 97 / 99 / 101 (seed 97 kept); 40k: the B3 gallery's 19 meshes x seed 97 (g40)."""
import os
import sys

arms = [("NRD", "--no-relax_driven"), ("NUC", "--no-u_adam_carry"), ("NSS", "--no-spacing_symmetric")]
fx = os.environ.get("POSTTAG_FX_FLAG", "").strip()
if fx:
    arms.append(("NFX", fx))
meshes40 = []
for s in open(sys.argv[1]):
    s = s.lstrip("#").replace("HOLD-posttag ", "").split()
    if len(s) > 4 and s[4] == "d145_b3" and s[0].endswith("40k_V97") and s[1] not in meshes40:
        meshes40.append(s[1])
for seed in (97, 99, 101):
    for m in ("bunny", "dragon"):
        for a, flag in arms:
            print(f"{m}300k_{a}{seed} {m} 300000 repo_ts d145t_a6 end300 {1 if seed == 97 else 0} --seed {seed} {flag}")
for m in meshes40:
    for a, flag in arms:
        print(f"{m}40k_{a}97 {m} 40000 repo_t d145t_a6 g40 0 --seed 97 {flag}")
