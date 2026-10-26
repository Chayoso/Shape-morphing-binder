"""edit_repo.py REPO -- D166: --settle_u zero|last and --settle_relax / --no-settle_relax (defaults = D162h's settle)."""
import sys
from pathlib import Path
R = Path(sys.argv[1])


def sub(rel, old, new):
    p = R / rel
    t = p.read_text()
    assert t.count(old) == 1, (rel, old[:60], t.count(old))
    p.write_text(t.replace(old, new))


sub("physmorph/pipeline/config.py",
    "    settle_windows: int = 0 ",
    "    settle_u: str = \"zero\"          # D166 A/B: u in the settle windows: zero (D162h, the default) or last = the last\n"
    "                                    #   committed window's u, held (rigid-free and clamped in each settle window, as u is)\n"
    "    settle_relax: bool = True       # D166 A/B: the outer-layer relaxation in the settle windows (on = D162h, the default)\n"
    "    settle_windows: int = 0 ")
sub("physmorph/pipeline/config.py",
    "        if self.render_weight_ref not in (",
    "        if self.settle_u not in (\"zero\", \"last\"):\n"
    "            raise ValueError(\"settle_u must be \\\"zero\\\" or \\\"last\\\"\")\n"
    "        if self.render_weight_ref not in (")
sub("scripts/pipeline_run.py",
    "    ap.add_argument(\"--placement_at_cell\",",
    "    ap.add_argument(\"--settle_u\", default=\"zero\", choices=[\"zero\", \"last\"],\n"
    "                    help=\"D166 A/B: u in the settle windows; zero (the default, D162h) or last = the last committed u held\")\n"
    "    ap.add_argument(\"--settle_relax\", action=argparse.BooleanOptionalAction, default=True,\n"
    "                    help=\"D166 A/B: the outer-layer relaxation in the settle windows (default on, D162h)\")\n"
    "    ap.add_argument(\"--placement_at_cell\",")
sub("scripts/pipeline_run.py",
    "settle_windows=args.settle_windows,",
    "settle_windows=args.settle_windows,\n                              settle_u=args.settle_u, settle_relax=args.settle_relax,")
sub("physmorph/pipeline/window/solve.py",
    "        stats.update(null_reason=self.tele.get(\"null_reason\") if self.accepted == 0 else None, E_accept=E_accept,",
    "        if getattr(cfg, \"settle_u\", \"zero\") == \"last\":     # D166: the committed u, for the settle windows (runner)\n"
    "            stats[\"u\"] = self.u.detach().clone()\n"
    "        stats.update(null_reason=self.tele.get(\"null_reason\") if self.accepted == 0 else None, E_accept=E_accept,")
sub("physmorph/pipeline/run/runner.py",
    "    dfc_prev = None\n    Fp_pre = None",
    "    dfc_prev = None\n    u_last = None                                   # D166 (settle_u last): the last kept commit's u\n    Fp_pre = None")
sub("physmorph/pipeline/run/runner.py",
    "        rec[\"frame_end\"] = len(frames)\n        converged = sel.accepted(rec, a, disp, improved)",
    "        rec[\"frame_end\"] = len(frames)\n        if \"u\" in stats:                               # D166: a kept commit's u (settle_u last)\n"
    "            u_last = stats[\"u\"]\n        converged = sel.accepted(rec, a, disp, improved)")
sub("physmorph/pipeline/run/runner.py",
    "                                  (coh_nbr, bond_rest, fragment_mask(x, prm).float()), 1.0, None, lambda *_: None)\n"
    "            with torch.no_grad():\n                commit = commit_rollout(",
    "                                  (coh_nbr, bond_rest, fragment_mask(x, prm).float()), 1.0, None, lambda *_: None)\n"
    "            if getattr(cfg, \"settle_u\", \"zero\") == \"last\" and u_last is not None:   # D166: the last committed u held\n"
    "                with torch.no_grad():\n"
    "                    opt.u.copy_(u_last)\n"
    "                opt.free_of_rigid()\n"
    "                with torch.no_grad():\n"
    "                    opt.u.clamp_(-opt.win.sp0, opt.win.sp0)\n"
    "            if not getattr(cfg, \"settle_relax\", True):        # D166: no layer relaxation in the settle windows\n"
    "                opt.win.tr.layer_frac = 0.0\n"
    "            with torch.no_grad():\n                commit = commit_rollout(")
print("edits ok")
