"""Build the high-resolution report page from the fetched per-example folders.

Reads output/report/<target>/ (produced by hr_post.sh and fetched):
  <t>_arm.txt (ARM lines + gates for render / phys), <t>_scatter.txt, <t>_loss.txt,
  <t>_grad_render.json, <t>_grad_phys.json, <t>_render.gif, <t>_phys.gif,
  <t>_{render,phys,target}_pbr_az{35,215}.png, <t>_grad_render.png, <t>_loss.png
and output/report/global.json (assessment numbers, speed, target fix) -> output/report/index.html
"""
import glob
import json
import os
import re

ROOT = r"C:\dev\Shape-morphing-binder\output\report"
targets = [os.path.basename(p) for p in sorted(glob.glob(os.path.join(ROOT, "*"))) if os.path.isdir(p)]
G = json.load(open(os.path.join(ROOT, "global.json"), encoding="utf-8")) if os.path.exists(os.path.join(ROOT, "global.json")) else {}


def rd(p):
    return open(p, encoding="utf-8", errors="replace").read() if os.path.exists(p) else ""


def parse_arm(txt):
    """two ARM lines (render first, phys second) -> dicts."""
    out = []
    for line in txt.splitlines():
        if "chamfer=" in line:
            d = dict(re.findall(r"(\w+)=([-\d.e%]+)", line))
            m = re.search(r"\(([\d.]+) min\)", line)
            d["min"] = m.group(1) if m else ""
            out.append(d)
    gates = [l for l in txt.splitlines() if "gates:" in l]
    return out, gates


def parse_scatter(txt):
    out = []
    for line in txt.splitlines():
        m = re.search(r"peak ([\d.]+) @frame \d+ -> end ([\d.]+) \| mass on thin targets: [\d.]+ -> ([\d.]+) \(target ([\d.]+)\) \| far>2sp end ([\d.]+)", line)
        if m:
            out.append(dict(peak=m.group(1), end=m.group(2), mass=m.group(3), tgt=m.group(4), far=f"{float(m.group(5))*100:.2f} %"))
    return out


def parse_loss(txt):
    out = []
    for line in txt.splitlines():
        m = re.search(r"commits=(\d+) loss ([\d.e-]+) -> ([\d.e-]+) \(x([\d.]+)\) D_vol ([\d.e-]+) -> ([\d.e-]+) \| wall ([\d.]+) min for (\d+) packets \(([\d.]+) s/commit\)", line)
        if m:
            out.append(dict(commits=m.group(1), L0=m.group(2), L1=m.group(3), ratio=m.group(4), dv0=m.group(5), dv1=m.group(6), wall=m.group(7), spc=m.group(9)))
    return out


def grad_summary(js):
    if not js or "frames" not in js:
        return None
    f = js["frames"][-1]
    return dict(surface=f["surface_share"], rx_all=f["g_render_x"]["active_all"], rx_surf=f["g_render_x"]["active_surface"],
                rx_int=f["g_render_x"]["active_interior"], rx_ms=f["g_render_x"]["mean_surface"], rx_mi=f["g_render_x"]["mean_interior"],
                vx_ms=f["g_vol_x"]["mean_surface"], vx_mi=f["g_vol_x"]["mean_interior"],
                rc_all=f["g_render_dfc"]["active_all"], rc_surf=f["g_render_dfc"]["active_surface"], rc_int=f["g_render_dfc"]["active_interior"],
                reach=[r["share_active"] for r in f["reach_by_depth"]], frames=[(x["frame"], x["g_render_x"]["active_surface"], x["g_render_dfc"]["active_all"]) for x in js["frames"]])


rows = []
for t in targets:
    d = os.path.join(ROOT, t)
    arms, gates = parse_arm(rd(os.path.join(d, f"{t}_arm.txt")))
    sc = parse_scatter(rd(os.path.join(d, f"{t}_scatter.txt")))
    lo = parse_loss(rd(os.path.join(d, f"{t}_loss.txt")))
    gj = os.path.join(d, f"{t}_grad_render.json")
    gr = grad_summary(json.load(open(gj)) if os.path.exists(gj) else None)
    gp = os.path.join(d, f"{t}_grad_phys.json")
    gph = grad_summary(json.load(open(gp)) if os.path.exists(gp) else None)
    rows.append(dict(t=t, arms=arms, gates=gates, sc=sc, lo=lo, gr=gr, gph=gph,
                     has=lambda n, d=d, t=t: os.path.exists(os.path.join(d, n.format(t=t)))))


def pct(v):
    return f"{100*v:.1f} %" if isinstance(v, (int, float)) else "–"


def sci(v):
    return f"{v:.1e}" if isinstance(v, (int, float)) else "–"


def prep(t, name):
    """Artifact size budget (64 MB per version): PBR stills -> JPEG q85 at 800 px, heatmaps
    -> PNG downscaled to 1000 px wide. Returns the published file name."""
    from PIL import Image
    p = os.path.join(ROOT, t, name)
    if not os.path.exists(p):
        return None
    if name.endswith("_pbr_az35.png") or name.endswith("_pbr_az215.png"):
        out = name[:-4] + ".jpg"
        q = os.path.join(ROOT, t, out)
        if not os.path.exists(q) or os.path.getmtime(q) < os.path.getmtime(p):
            im = Image.open(p).convert("RGB"); im.thumbnail((800, 800)); im.save(q, quality=85, optimize=True)
        return out
    if name.endswith("_grad_render.png") or name.endswith("_grad_phys.png"):
        out = name[:-4] + "_s.jpg"
        q = os.path.join(ROOT, t, out)
        if not os.path.exists(q) or os.path.getmtime(q) < os.path.getmtime(p):
            im = Image.open(p).convert("RGB"); w, h = im.size
            im = im.resize((1000, int(h * 1000 / w))); im.save(q, quality=82, optimize=True)
        return out
    return name


def fig(t, name, cap, cls=""):
    pub = prep(t, name)
    if pub is None:
        return ""
    return f'<figure class="{cls}"><img src="{t}/{pub}" alt="{cap}" loading="lazy"><figcaption>{cap}</figcaption></figure>'


css = """
:root{--bg:#f4f3ef;--panel:#fffdf9;--ink:#1f2a33;--muted:#68727c;--line:#d9d5cc;--accent:#8a4b2b;--ok:#2f7d4f;--warn:#b2741a;--bad:#a83a2c;
--sans:"IBM Plex Sans","Segoe UI",system-ui,sans-serif;--mono:"IBM Plex Mono",Consolas,monospace;color-scheme:light dark}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#15181c;--panel:#1d2126;--ink:#e8e6e0;--muted:#a3a9b0;--line:#30363d;--accent:#d99a72;--ok:#6cc08b;--warn:#d9a04a;--bad:#e07a6c}}
:root[data-theme="dark"]{--bg:#15181c;--panel:#1d2126;--ink:#e8e6e0;--muted:#a3a9b0;--line:#30363d;--accent:#d99a72;--ok:#6cc08b;--warn:#d9a04a;--bad:#e07a6c}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 var(--sans)}
main{max-width:1240px;margin:0 auto;padding:32px 24px 72px}
h1{font-size:28px;font-weight:600;letter-spacing:-.01em;margin:0 0 6px;text-wrap:balance}
h2{font-size:20px;font-weight:600;margin:40px 0 8px;border-top:2px solid var(--accent);padding-top:14px}
h3{font-size:16px;font-weight:600;margin:22px 0 6px}
.eyebrow{font:500 12px/1 var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.lede{color:var(--muted);max-width:76ch;margin:0 0 14px}
p{max-width:80ch}
table{border-collapse:collapse;font:13px/1.4 var(--mono);font-variant-numeric:tabular-nums;width:100%;margin:8px 0 14px}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{color:var(--muted);font-weight:500;border-bottom:2px solid var(--line)}
.wrap{overflow-x:auto}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}
.grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}
figure{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:6px;overflow:hidden}
figure img{display:block;width:100%;height:auto;background:#fff}
figcaption{padding:8px 12px 10px;font-size:13px;color:var(--muted)}
.pill{display:inline-block;font:500 11px/1 var(--mono);letter-spacing:.06em;text-transform:uppercase;padding:4px 8px;border-radius:3px;border:1px solid currentColor;margin-right:6px}
.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}
.note{border-left:3px solid var(--accent);padding:8px 14px;color:var(--muted);font-size:14px;max-width:84ch;background:var(--panel)}
.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:6px 18px;font:13px/1.5 var(--mono)}
.kv div{display:flex;justify-content:space-between;border-bottom:1px dotted var(--line)}
.kv span:first-child{color:var(--muted)}
code{font-family:var(--mono);font-size:.92em}
"""

H = [f'<title>PhysMorph 고해상도 보고서</title>',
     '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">',
     f'<style>{css}</style>', '<main>',
     '<div class="eyebrow">PhysMorph · sphere → 10 targets · 40k, --ppc 8, density units, kinetic recipe · hyde06 2026-09-16</div>',
     '<h1>고해상도 보고서 — 렌더 제어 vs physics-only, 기울기 도달 범위, 손실과 시간</h1>',
     f'<p class="lede">{G.get("lede", "")}</p>']

# ---- assessment -------------------------------------------------------------------------
H.append('<h2>1. 판단 — 남은 문제와 이번에 한 것</h2>')
H.append(G.get("assessment_html", ""))

# ---- summary table ------------------------------------------------------------------------
H.append('<h2>2. 10개 예제 요약 (raw state 지표, 렌더러 미사용)</h2>')
H.append('<p class="lede">각 행: 렌더 제어 arm(render_full_dt_iso_nn, λ=0.5) / physics-only 쌍(같은 코드 경로, λ=0). 시간은 arm 벽시계(hyde06 RTX 6000 Ada 1 GPU, 다른 런과 GPU 공유). "sparse"는 얇은 부위 목표점 위 입자 중 8-NN 평균 간격이 목표 간격의 2배를 넘는 비율.</p>')
H.append('<div class="wrap"><table><thead><tr><th>target</th><th>arm</th><th>chamfer</th><th>silIoU</th><th>hole</th><th>commits</th><th>min</th><th>s/commit</th><th>loss ×</th><th>sparse peak→end</th><th>thin mass / tgt</th><th>stray &gt;2sp</th><th>G4 ej.</th></tr></thead><tbody>')
for r in rows:
    for i, lab in enumerate(("render", "phys")):
        a = r["arms"][i] if len(r["arms"]) > i else {}
        s = r["sc"][i] if len(r["sc"]) > i else {}
        l = r["lo"][i] if len(r["lo"]) > i else {}
        g = r["gates"][i] if len(r["gates"]) > i else ""
        ej = "FAIL" if "G4_ejection=FAIL" in g else ("PASS" if g else "–")
        H.append(f'<tr><td>{r["t"] if i == 0 else ""}</td><td>{lab}</td><td>{a.get("chamfer","–")}</td><td>{a.get("silIoU","–")}</td><td>{a.get("hole","–")}</td>'
                 f'<td>{l.get("commits","–")}</td><td>{a.get("min","–")}</td><td>{l.get("spc","–")}</td><td>{l.get("ratio","–")}</td>'
                 f'<td>{s.get("peak","–")} → {s.get("end","–")}</td><td>{s.get("mass","–")} / {s.get("tgt","–")}</td><td>{s.get("far","–")}</td>'
                 f'<td class="{"bad" if ej=="FAIL" else "ok"}">{ej}</td></tr>')
H.append('</tbody></table></div>')

# ---- gradient table ---------------------------------------------------------------------
H.append('<h2>3. 기울기 — 표면의 몇 %가 영향을 받고, 얼마나 크며, 어디까지 닿는가</h2>')
H.append(G.get("grad_intro_html", ""))
H.append('<div class="wrap"><table><thead><tr><th>target</th><th>surface share</th><th>∂D_render/∂x active: all / surface / interior</th><th>|g_r| mean surf / int</th><th>|g_vol| mean surf / int</th><th>∂D_render/∂dFc active (adjoint): all / surface / interior</th><th>reach by depth decile (surface → core)</th></tr></thead><tbody>')
for r in rows:
    g = r["gr"]
    if not g:
        continue
    H.append(f'<tr><td>{r["t"]}</td><td>{pct(g["surface"])}</td><td>{pct(g["rx_all"])} / {pct(g["rx_surf"])} / {pct(g["rx_int"])}</td>'
             f'<td>{sci(g["rx_ms"])} / {sci(g["rx_mi"])}</td><td>{sci(g["vx_ms"])} / {sci(g["vx_mi"])}</td>'
             f'<td>{pct(g["rc_all"])} / {pct(g["rc_surf"])} / {pct(g["rc_int"])}</td><td>{" ".join(f"{v:.2f}" for v in g["reach"])}</td></tr>')
H.append('</tbody></table></div>')

# ---- per example ------------------------------------------------------------------------
H.append('<h2>4. 예제별 결과 — GIF(고해상도, 위 render / 아래 physics-only), PBR 스틸, 기울기 히트맵, 손실 곡선</h2>')
for r in rows:
    t = r["t"]
    a0 = r["arms"][0] if r["arms"] else {}
    a1 = r["arms"][1] if len(r["arms"]) > 1 else {}
    H.append(f'<h3>sphere → {t}</h3>')
    H.append('<div class="kv">' + "".join(
        f'<div><span>{k}</span><span>{v}</span></div>' for k, v in [
            ("render chamfer / silIoU", f'{a0.get("chamfer","–")} / {a0.get("silIoU","–")}'),
            ("phys chamfer / silIoU", f'{a1.get("chamfer","–")} / {a1.get("silIoU","–")}'),
            ("render min / phys min", f'{a0.get("min","–")} / {a1.get("min","–")}'),
            ("thin mass render / phys", f'{(r["sc"][0]["mass"] if r["sc"] else "–")} / {(r["sc"][1]["mass"] if len(r["sc"])>1 else "–")}')]) + '</div>')
    H.append('<div class="grid">' + fig(t, f"{t}_render.gif", f"{t} — render control (az 0.6 / 2.2)") + fig(t, f"{t}_phys.gif", f"{t} — physics-only twin") + '</div>')
    H.append('<div class="grid3">' + fig(t, f"{t}_target_pbr_az35.png", "target (PBR still, az 35°)") + fig(t, f"{t}_render_pbr_az35.png", "render control, delivered frame (az 35°)") + fig(t, f"{t}_phys_pbr_az35.png", "physics-only, delivered frame (az 35°)") + '</div>')
    H.append('<div class="grid3">' + fig(t, f"{t}_target_pbr_az215.png", "target (az 215°)") + fig(t, f"{t}_render_pbr_az215.png", "render control (az 215°)") + fig(t, f"{t}_phys_pbr_az215.png", "physics-only (az 215°)") + '</div>')
    H.append('<div class="grid">' + fig(t, f"{t}_grad_render.png", "gradient heatmaps (log10 |g|): columns ∂D_render/∂x · ∂D_vol/∂x · ∂D_render/∂dFc through the MPM adjoint; rows = commits 5 % / 35 % / delivered × two views") + fig(t, f"{t}_loss.png", "window loss and D_vol vs commit, and loss vs wall-clock (live-packet mtimes)") + '</div>')

# ---- global sections ----------------------------------------------------------------------
for key, title in (("speed_html", "5. 속도 — 프로파일과 할당 수정"), ("fix_html", "6. 타깃 이산화 수정과 mass ejection 상태"), ("summary_html", "7. 요약과 다음 단계"), ("viewer_html", "8. 3D 뷰어에서 보기")):
    if G.get(key):
        H.append(f'<h2>{title}</h2>' + G[key])
H.append('</main>')
open(os.path.join(ROOT, "index.html"), "w", encoding="utf-8").write("\n".join(H))
print("report built:", len(rows), "examples")

# ---- Markdown mirror for docs/ (tables only; images live in the artifact) ---------------
md = ["# High-resolution report (2026-09-16) — sphere → 10 targets, 40k `--ppc 8`, render vs physics-only", "",
      "Artifact (GIFs, PBR stills, heatmaps): https://claude.ai/code/artifact/3a1a66fe-6100-4ccf-9e54-21c02a1d0c8a", "",
      "Discretisation and arms: `docs/experiments.md` 2026-09-16 (batch h). Numbers are raw-state metrics; the renderer is not consumed.", "",
      "## Summary per example", "",
      "| target | arm | chamfer | silIoU | hole | commits | min | s/commit | loss × | sparse peak → end | thin mass / tgt | strays > 2 sp | G4 ejection |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    for i, lab in enumerate(("render", "phys")):
        a = r["arms"][i] if len(r["arms"]) > i else {}
        s = r["sc"][i] if len(r["sc"]) > i else {}
        l = r["lo"][i] if len(r["lo"]) > i else {}
        g = r["gates"][i] if len(r["gates"]) > i else ""
        ej = "FAIL" if "G4_ejection=FAIL" in g else ("PASS" if g else "–")
        md.append(f'| {r["t"] if i == 0 else ""} | {lab} | {a.get("chamfer","–")} | {a.get("silIoU","–")} | {a.get("hole","–")} | {l.get("commits","–")} | {a.get("min","–")} | {l.get("spc","–")} | {l.get("ratio","–")} | {s.get("peak","–")} → {s.get("end","–")} | {s.get("mass","–")} / {s.get("tgt","–")} | {s.get("far","–")} | {ej} |')
md += ["", "## Gradient reach (delivered frame of the render run)", "",
       "| target | surface share | ∂D_render/∂x active all / surface / interior | mean |g_r| surf / int | mean |g_vol| surf / int | ∂D_render/∂dFc active all / surface / interior | reach by depth decile |",
       "|---|---|---|---|---|---|---|"]
for r in rows:
    g = r["gr"]
    if g:
        md.append(f'| {r["t"]} | {pct(g["surface"])} | {pct(g["rx_all"])} / {pct(g["rx_surf"])} / {pct(g["rx_int"])} | {sci(g["rx_ms"])} / {sci(g["rx_mi"])} | {sci(g["vx_ms"])} / {sci(g["vx_mi"])} | {pct(g["rc_all"])} / {pct(g["rc_surf"])} / {pct(g["rc_int"])} | {" ".join(f"{v:.2f}" for v in g["reach"])} |')
import re as _re
strip = lambda h: _re.sub(r"<[^>]+>", "", h or "")
md += ["", "## Assessment", "", strip(G.get("assessment_html", "")), "", "## Speed", "", strip(G.get("speed_html", "")),
       "", "## Target fix and mass ejection", "", strip(G.get("fix_html", "")), "", "## Summary", "", strip(G.get("summary_html", "")),
       "", "## Viewer", "", strip(G.get("viewer_html", ""))]
open(r"C:\dev\Shape-morphing-binder\docs\highres_report.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
print("markdown written")
