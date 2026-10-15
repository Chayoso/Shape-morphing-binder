"""rgv_patch2.py REPO -- RGV's measurement copy, version 2 (applied to a copy of repo_rgv, never to the pipeline): the
dump also keeps what panels A and C need to count ALL the non-render terms as the physics' pull (the physics core,
phys_core = the released end's geometry (transport, surface proximity, end drift) + the released motion; and the
cleanup sweep, obj.cleanup = the spray cleanup + the near band, which the deliverable adds after PCGrad):

  gl_dt_tmean      the cleanup's control gradient, its mean over the steps per particle (N, 9) (the hook skipped it)
  gl_nonrend_psum  sum over the steps of the Frobenius norm of (core + cleanup) per particle (N): the exact pull of
                   everything but the render (a sum of the two channels' own sums is only its upper bound)
  gl_nonrend_tmean (core + cleanup)'s mean over the steps (N, 9): its direction, for C's cosine

Read-only like the hooks it extends (written from the dump the window already took); tests/dump_runs.py records the
new keys and checks the triangle inequality sum|core + cleanup| <= sum|core| + sum|cleanup| per particle. Applied
once; every replacement must match exactly once."""
import sys

R = sys.argv[1]


def patch(path, pairs):
    s = open(path).read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:90], s.count(old))
        s = s.replace(old, new)
    open(path, "w").write(s)
    print("patched", path)


patch(R + "/physmorph/pipeline/window/telemetry.py", [
    ("        if name in (\"gl_rend_pc\", \"gl_dt\"):\n"
     "            continue                                             # D145 B7: no linear-response rollout for these\n",
     "        if name in (\"gl_rend_pc\", \"gl_dt\"):\n"
     "            if name == \"gl_dt\":                                  # RGV v2: the cleanup's direction\n"
     "                red[\"gl_dt_tmean\"] = gpu.host(g4.mean(0))\n"
     "            continue                                             # D145 B7: no linear-response rollout for these\n"),
    ("    win.dc_buf.copy_(win.expand(leaf0).view(T, N, 3, 3))\n"
     "    resp[\"xT_base\"] = end_positions()\n",
     "    if dump.get(\"gl_phys\") is not None and dump.get(\"gl_dt\") is not None:   # RGV v2: every non-render term\n"
     "        g4 = win.expand(dump[\"gl_phys\"] + dump[\"gl_dt\"]).detach().view(T, N, 9)\n"
     "        red[\"gl_nonrend_psum\"] = gpu.host(g4.norm(dim=2).sum(0))\n"
     "        red[\"gl_nonrend_tmean\"] = gpu.host(g4.mean(0))\n"
     "    win.dc_buf.copy_(win.expand(leaf0).view(T, N, 3, 3))\n"
     "    resp[\"xT_base\"] = end_positions()\n"),
])

patch(R + "/tests/dump_runs.py", [(
    "    rows.update(cf_null=null, cf_null_keys=keys,\n",
    "    # RGV v2: the non-render pull's records, and sum|core + cleanup| <= sum|core| + sum|cleanup| per particle\n"
    "    nonrend = []\n"
    "    for f in sorted(os.listdir(g)) if os.path.isdir(g) else []:\n"
    "        zz = np.load(os.path.join(g, f))\n"
    "        if \"gl_dt_psum\" in zz.files:\n"
    "            nonrend.append(bool({\"gl_dt_tmean\", \"gl_nonrend_psum\", \"gl_nonrend_tmean\"} <= set(zz.files))\n"
    "                           and bool((zz[\"gl_nonrend_psum\"] <= zz[\"gl_phys_psum\"] + zz[\"gl_dt_psum\"] + 1e-6).all()))\n"
    "    rows.update(nonrend=nonrend)\n"
    "    rows.update(cf_null=null, cf_null_keys=keys,\n")])

patch(R + "/tests/test_dump_hooks.py", [(
    "    assert len(rows[\"cf_null\"]) >= 1 and all(v == 0.0 for v in rows[\"cf_null\"]), rows\n",
    "    assert len(rows[\"cf_null\"]) >= 1 and all(v == 0.0 for v in rows[\"cf_null\"]), rows\n"
    "    # RGV v2: where the cleanup has its own sweep, the non-render pull is recorded and bounded by its two channels\n"
    "    assert all(rows[\"nonrend\"]), rows\n")])
