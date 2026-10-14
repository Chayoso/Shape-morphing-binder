"""rgv_patch.py REPO -- the render-gradient videos' measurement copy (RGV): applied to a copy of repo_d145g (D145 B7's
hooks on deliverable-2026-10-13, 592e97a: the per-window gradient dump and the lambda = 0 counterfactual of the same
window from the same start, discarded before the normal window), never to the pipeline. It adds one record to the
counterfactual's file, the change of the committed CONTROLS that the render makes:

  dfc_render_mean, dfc_phys_mean  each window's committed dFc, mean over the driven steps (N, 9), render on / lambda = 0
  ddfc_mean                       their difference (render on - lambda = 0), mean over the driven steps (N, 9)
  ddfc_psum                       the per-particle sum over the driven steps of the difference's Frobenius norm (N)
  u_render, u_phys                the committed u of each (N; zero off the outer layer)
  u_gate, u_gate_phys             the gate g of each (u moves a layer particle by g u along its normal; the same start)

optimize_window hands the committed u and its gate out on the result when the counterfactual is on (cfg.cf_dir), so
nothing is recomputed. Read-only like the hooks it extends: no flag is added, nothing the run reads is touched; the
copy's tests/dump_runs.py gains a check that with the render weight at zero the two passes are the same (every new
difference exactly zero under the deterministic atomics) and that the records are written. Applied once; every
replacement must match exactly once."""
import sys

R = sys.argv[1]


def patch(path, pairs):
    s = open(path).read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:90], s.count(old))
        s = s.replace(old, new)
    open(path, "w").write(s)
    print("patched", path)


patch(R + "/physmorph/pipeline/window/solve.py", [(
    "    opt = WindowOptimizer(start, prm, cfg, tgt, balancer, vol0, bonds, alpha_scale, on_iter, log)\n"
    "    return opt.run(dfc_init)\n",
    "    opt = WindowOptimizer(start, prm, cfg, tgt, balancer, vol0, bonds, alpha_scale, on_iter, log)\n"
    "    res = opt.run(dfc_init)\n"
    "    if cfg.cf_dir:                  # RGV (measurement copy): the committed u and its gate, for the control difference\n"
    "        with torch.no_grad():\n"
    "            res.u_final = opt.u.detach().clone()\n"
    "            res.u_gate = (wp.to_torch(opt.win.tr.layer_ug) * opt.win.lmask).detach().clone()\n"
    "    return res\n")])

patch(R + "/physmorph/pipeline/run/runner.py", [
    ("        x_cf, cf_rec = None, None\n",
     "        x_cf, cf_rec, ctl_cf = None, None, None\n"),
    ("                        x_cf = res0.commit.x[-1].detach().clone()\n",
     "                        x_cf = res0.commit.x[-1].detach().clone()\n"
     "                        ctl_cf = _cf_controls(res0)      # RGV: the lambda = 0 pass's committed controls\n"),
    ("                                cf=repr(cf_rec), render=repr(_cf_record(res)))\n",
     "                                cf=repr(cf_rec), render=repr(_cf_record(res)), **_cf_delta(ctl_cf, res))\n"),
    ("def _cf_record(res) -> dict:\n",
     "def _cf_controls(res):\n"
     "    \"\"\"RGV: a window's committed controls, dFc (T, N, 3, 3), u and its gate (N), copied.\"\"\"\n"
     "    return res.stats[\"dfc\"].detach().clone(), res.u_final, res.u_gate\n"
     "\n"
     "\n"
     "def _cf_delta(ctl_cf, res) -> dict:\n"
     "    \"\"\"RGV: the render-induced change of the committed controls, render on minus lambda = 0, the same window from the\n"
     "    same start: each dFc's mean over the driven steps (N, 9), the difference's mean and the per-particle sum over the\n"
     "    steps of its Frobenius norm, u and the gate of each. Empty when either pass committed nothing.\"\"\"\n"
     "    if ctl_cf is None or res.commit is None:\n"
     "        return {}\n"
     "    d_r, u_r, g_r = _cf_controls(res)\n"
     "    d_0, u_0, g_0 = ctl_cf\n"
     "    T, N = d_r.shape[0], d_r.shape[1]\n"
     "    with torch.no_grad():\n"
     "        dd = (d_r - d_0).reshape(T, N, 9)\n"
     "        return dict(dfc_render_mean=_host(d_r.reshape(T, N, 9).mean(0)),\n"
     "                    dfc_phys_mean=_host(d_0.reshape(T, N, 9).mean(0)), ddfc_mean=_host(dd.mean(0)),\n"
     "                    ddfc_psum=_host(dd.norm(dim=2).sum(0)), u_render=_host(u_r), u_phys=_host(u_0),\n"
     "                    u_gate=_host(g_r), u_gate_phys=_host(g_0))\n"
     "\n"
     "\n"
     "def _cf_record(res) -> dict:\n"),
])

patch(R + "/tests/dump_runs.py", [(
    "            \"cf_grad\": same(ref, run(cf_dir=c + \"2\", cf_windows=\"1\", grad_dump=g2, dump_windows=\"0,+2\"))}\n",
    "            \"cf_grad\": same(ref, run(cf_dir=c + \"2\", cf_windows=\"1\", grad_dump=g2, dump_windows=\"0,+2\"))}\n"
    "    # RGV: the control records are written, and with the render weight at zero the counterfactual IS the window\n"
    "    # (every difference exactly zero), so what they measure with the render on is the render's alone\n"
    "    c0 = os.path.join(out_dir, \"cf_null\")\n"
    "    run(cf_dir=c0, cf_windows=\"0,1,2\", render_weight_scale=0.0)\n"
    "    null, keys = [], []\n"
    "    for f in sorted(os.listdir(c0)):\n"
    "        z = np.load(os.path.join(c0, f))\n"
    "        keys.append(sorted(z.files))\n"
    "        if \"ddfc_mean\" in z.files:\n"
    "            null.append(max(float(np.abs(z[\"ddfc_mean\"]).max()), float(np.abs(z[\"ddfc_psum\"]).max()),\n"
    "                            float(np.abs(z[\"u_render\"] - z[\"u_phys\"]).max()),\n"
    "                            float(np.abs(z[\"x_render\"] - z[\"x_phys\"]).max()),\n"
    "                            float(np.abs(z[\"dfc_render_mean\"] - z[\"dfc_phys_mean\"]).max())))\n"
    "    rows.update(cf_null=null, cf_null_keys=keys,\n"
    "                cf_ctl_keys=[sorted(np.load(os.path.join(c, f)).files) for f in sorted(os.listdir(c))]\n"
    "                if os.path.isdir(c) else [])\n")])

patch(R + "/tests/test_dump_hooks.py", [(
    "    assert rows[\"cf_grad\"] and rows[\"grad_gated_files\"] == [\"win_0000.npz\", \"win_0002.npz\"], rows\n",
    "    assert rows[\"cf_grad\"] and rows[\"grad_gated_files\"] == [\"win_0000.npz\", \"win_0002.npz\"], rows\n"
    "    # RGV: the control-difference records, and with the render weight at zero the two passes are the same\n"
    "    ctl = {\"ddfc_mean\", \"ddfc_psum\", \"dfc_render_mean\", \"dfc_phys_mean\", \"u_render\", \"u_phys\", \"u_gate\"}\n"
    "    assert any(ctl <= set(k) for k in rows[\"cf_ctl_keys\"]), rows   # (a null window writes none)\n"
    "    assert len(rows[\"cf_null\"]) >= 1 and all(v == 0.0 for v in rows[\"cf_null\"]), rows\n")])
