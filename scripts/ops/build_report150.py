"""Build the 150k report page (v2) from the fetched per-example folders.

Reads output/report150/<target>/ (produced by hr150_post.sh and fetched):
  <t>_arm.txt (ARM line + gates), <t>_scatter.txt, <t>_census.txt, <t>_loss.txt, <t>_time.txt,
  <t>_surface.gif, <t>_particles.gif, <t>_{render,target}_pbr_az{35,215}.png, <t>_loss.png
and output/report150/global.json (assessment html blocks) -> output/report150/index.html
"""
import glob
import json
import os
import re

import sys
ROOT = sys.argv[1] if len(sys.argv) > 1 else r"C:\dev\Shape-morphing-binder\output\report150"
TITLE = sys.argv[2] if len(sys.argv) > 2 else "PhysMorph 150k 갤러리"
MD_OUT = sys.argv[3] if len(sys.argv) > 3 else r"C:\dev\Shape-morphing-binder\docs\highres150_report.md"
targets = [os.path.basename(p) for p in sorted(glob.glob(os.path.join(ROOT, "*"))) if os.path.isdir(p) and not os.path.basename(p).startswith("_")]  # _figs: shared figures
G = json.load(open(os.path.join(ROOT, "global.json"), encoding="utf-8")) if os.path.exists(os.path.join(ROOT, "global.json")) else {}


def rd(p):
    return open(p, encoding="utf-8", errors="replace").read() if os.path.exists(p) else ""


def parse_arm(txt):
    arm, gate = {}, ""
    for line in txt.splitlines():
        if "chamfer=" in line:
            arm = dict(re.findall(r"(\w+)=([-\d.e%]+)", line))
            m = re.search(r"\(([\d.]+) min\)", line)
            arm["min"] = m.group(1) if m else ""
        if "gates:" in line:
            gate = line
    return arm, gate


def parse_census(txt):
    m = re.search(r">0\.25 wu (\d+) \(([\d.]+)%\)\s+>0\.5 wu (\d+) \(([\d.]+)%\)\s+>1\.0 wu (\d+)\s+max ([\d.]+) wu", txt)
    return dict(n25=m.group(1), p25=m.group(2), n50=m.group(3), p50=m.group(4), n100=m.group(5), mx=m.group(6)) if m else {}


def parse_scatter(txt):
    m = re.search(r"peak ([\d.]+) @frame \d+ -> end ([\d.]+) \| mass on thin targets: [\d.]+ -> ([\d.]+) \(target ([\d.]+)\) \| far>2sp end ([\d.]+)", txt)
    return dict(peak=m.group(1), end=m.group(2), mass=m.group(3), tgt=m.group(4), far=f"{float(m.group(5))*100:.2f} %") if m else {}


def parse_loss(txt):
    m = re.search(r"commits=(\d+) loss ([\d.e-]+) -> ([\d.e-]+) \(x([\d.]+)\) D_vol ([\d.e-]+) -> ([\d.e-]+) \| wall ([\d.]+) min for (\d+) packets \(([\d.]+) s/commit\)", txt)
    return dict(commits=m.group(1), L0=m.group(2), L1=m.group(3), ratio=m.group(4), dv0=m.group(5), dv1=m.group(6), wall=m.group(7), spc=m.group(9)) if m else {}


def parse_qa(txt, cav_txt=""):
    """Photoreal sidecar (<t>_photoreal.mp4.components.txt): per-frame raw isosurface components, isolated
    particles, sub-cell components dropped by the deliverable rule, bridged components, interior cavities.
    A sidecar without the cavity column takes it from <t>_cavity.txt (scripts/probes/cavity_sweep.py).
    Returns the per-video QA summary."""
    cav = {}
    for line in cav_txt.splitlines():
        p = line.split()
        if p and not line.startswith("#") and p[0].isdigit():
            cav[int(p[0])] = int(p[1])
    fr = []
    for line in txt.splitlines():
        if line.startswith("#") or line.startswith("archived_frame") or not line.strip():
            continue
        p = line.split()
        if len(p) >= 3:
            fr.append((int(p[0]), int(p[1]), int(p[2]), int(p[3]) if len(p) > 3 else 0, int(p[4]) if len(p) > 4 else 0,
                       int(p[5]) if len(p) > 5 else cav.get(int(p[0]), 0)))
    if not fr:
        return {}
    raw = [f[1] for f in fr]; iso = [f[2] for f in fr]; drop = [f[3] for f in fr]; br = [f[4] for f in fr]; cv = [f[5] for f in fr]
    drawn = [r - d - c_ for r, d, c_ in zip(raw, drop, cv)]         # outer pieces only (interior cavities are not pieces)
    return dict(n=len(fr), raw_gt1=sum(1 for v in raw if v > 1), raw_max=max(raw),
                drawn_gt1=sum(1 for v in drawn if v > 1), drawn_max=max(drawn),
                bridged=sum(1 for v in br if v > 0),
                unbridged=sum(1 for d_, b_ in zip(drawn, br) if d_ > 1 and b_ < d_ - 1),
                drop_frames=sum(1 for v in drop if v > 0), drop_total=sum(drop),
                iso_max=max(iso), iso_end=iso[-1], iso_frame=fr[max(range(len(fr)), key=lambda i: iso[i])][0])


def parse_gridfrag(txt):
    """scripts/probes/grid_fragments.py sidecar: physical fragments by the grid's own criterion."""
    m = re.search(r"# frames (\d+)\s+fragments>=1cell in (\d+) frames \(max count (\d+), max size (\d+) particles = ([\d.]+) cells; ppc (\d+)[^)]*\)\s+clusters>=20 in (\d+) frames", txt)
    return dict(n=m.group(1), f1=m.group(2), f1max=m.group(3), mx=m.group(4), mxc=m.group(5), ppc=m.group(6), f20=m.group(7)) if m else {}


rows = []
for t in targets:
    d = os.path.join(ROOT, t)
    arm, gate = parse_arm(rd(os.path.join(d, f"{t}_arm.txt")))
    rows.append(dict(t=t, arm=arm, gate=gate, sc=parse_scatter(rd(os.path.join(d, f"{t}_scatter.txt"))),
                     ce=parse_census(rd(os.path.join(d, f"{t}_census.txt"))), lo=parse_loss(rd(os.path.join(d, f"{t}_loss.txt"))),
                     tm=rd(os.path.join(d, f"{t}_time.txt")).strip(),
                     frag=rd(os.path.join(d, f"{t}_frag.txt")).strip(),
                     reatt=(rd(os.path.join(d, f"{t}_reatt.txt")).split() or ["–"])[0],
                     gf=parse_gridfrag(rd(os.path.join(d, f"{t}_gridfrag.txt"))),
                     qa=parse_qa(rd(os.path.join(d, f"{t}_photoreal.mp4.components.txt")), rd(os.path.join(d, f"{t}_cavity.txt")))))


def prep(t, name):
    """Artifact size budget: PBR stills -> JPEG q85 at 800 px. Returns the published name."""
    from PIL import Image
    p = os.path.join(ROOT, t, name)
    if not os.path.exists(p):
        return None
    if name.endswith(".gif"):                       # GIF -> MP4 (26 MB -> ~1 MB), <video> in the page
        import subprocess
        out = name[:-4] + ".mp4"
        q = os.path.join(ROOT, t, out)
        if not os.path.exists(q) or os.path.getmtime(q) < os.path.getmtime(p):
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", p, "-movflags", "faststart", "-pix_fmt", "yuv420p",
                            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-crf", "24", q], check=True)
        return out
    if name.endswith("_pbr_az35.png") or name.endswith("_pbr_az215.png"):
        out = name[:-4] + ".jpg"
        q = os.path.join(ROOT, t, out)
        if not os.path.exists(q) or os.path.getmtime(q) < os.path.getmtime(p):
            im = Image.open(p).convert("RGB"); im.thumbnail((800, 800)); im.save(q, quality=85, optimize=True)
        return out
    return name


def fig(t, name, cap, cls=""):
    pub = prep(t, name)
    if pub is None:
        return ""
    if pub.endswith(".mp4"):
        return (f'<figure class="{cls}"><video src="{t}/{pub}" autoplay loop muted playsinline preload="metadata" '
                f'aria-label="{cap}"></video><figcaption>{cap}</figcaption></figure>')
    return f'<figure class="{cls}"><img src="{t}/{pub}" alt="{cap}" loading="lazy"><figcaption>{cap}</figcaption></figure>'


css = """
:root{--bg:#f3f4f1;--panel:#fbfcf8;--ink:#1c2630;--muted:#65717b;--line:#d6d9d1;--accent:#2f6f8f;--ok:#2f7d4f;--warn:#b2741a;--bad:#a83a2c;
--sans:"IBM Plex Sans","Segoe UI",system-ui,sans-serif;--mono:"IBM Plex Mono",Consolas,monospace;color-scheme:light dark}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#14181b;--panel:#1c2126;--ink:#e6e8e3;--muted:#a0a8ae;--line:#2f363c;--accent:#7fb7d3;--ok:#6cc08b;--warn:#d9a04a;--bad:#e07a6c}}
:root[data-theme="dark"]{--bg:#14181b;--panel:#1c2126;--ink:#e6e8e3;--muted:#a0a8ae;--line:#2f363c;--accent:#7fb7d3;--ok:#6cc08b;--warn:#d9a04a;--bad:#e07a6c}
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
.grid4{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px}
figure{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:6px;overflow:hidden}
figure img,figure video{display:block;width:100%;height:auto;background:#fff}
figcaption{padding:8px 12px 10px;font-size:13px;color:var(--muted)}
.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}
.note{border-left:3px solid var(--accent);padding:8px 14px;color:var(--muted);font-size:14px;max-width:84ch;background:var(--panel)}
.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:6px 18px;font:13px/1.5 var(--mono)}
.kv div{display:flex;justify-content:space-between;border-bottom:1px dotted var(--line)}
.kv span:first-child{color:var(--muted)}
code{font-family:var(--mono);font-size:.92em}
ul{max-width:84ch}
"""

H = [f'<title>{TITLE}</title>',
     '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">',
     f'<style>{css}</style>', '<main>',
     f'<div class="eyebrow">{G.get("eyebrow", "PhysMorph · sphere → 10 targets · 150k particles, --ppc 8, density units, kinetic recipe, auto domain, material re-coupling · hyde06 2026-09-16")}</div>',
     f'<h1>{G.get("h1", "150k 고해상도 갤러리 — 표면 렌더, mass ejection 조사, 속도")}</h1>',
     f'<p class="lede">{G.get("lede", "")}</p>']
H.append('<h2>1. 판단</h2>'); H.append(G.get("assessment_html", ""))
H.append(f'<h2>2. {len(rows)}개 예제 요약 (raw state 지표, 렌더러 미사용)</h2>')
H.append(f'<p class="lede">{G.get("table_lede", "arm = render_full_dt_iso_nn (λ=0.5), 150k `--ppc 8`, `--bonds` (재료 재결합, 그리드 연결성 fragment mask), `--domain auto`, archive stride 8.")} "off-target"은 마지막 프레임에서 가장 가까운 타깃 점까지의 거리가 0.5 wu를 넘는 입자 수(타깃 밖에 남은 재료; stray_census.py), 이탈 입자 수는 "fragments (grid)". 시간은 hyde06 벽시계(1 GPU, 다른 40k 런과 공유된 구간 포함).</p>')
H.append('<div class="wrap"><table><thead><tr><th>target</th><th>chamfer</th><th>silIoU</th><th>hole</th><th>commits</th><th>min</th><th>s/commit</th><th>loss ×</th><th>sparse peak→end</th><th>thin mass / tgt</th><th>fragments (grid)</th><th>off-target &gt;0.5 wu (n, %)</th><th>max off-target</th><th>G4 ej.</th></tr></thead><tbody>')
for r in rows:
    a, s, l, c = r["arm"], r["sc"], r["lo"], r["ce"]
    ej = "FAIL" if "G4_ejection=FAIL" in r["gate"] else ("PASS" if r["gate"] else "–")
    H.append(f'<tr><td>{r["t"]}</td><td>{a.get("chamfer","–")}</td><td>{a.get("silIoU","–")}</td><td>{a.get("hole","–")}</td>'
             f'<td>{l.get("commits","–")}</td><td>{a.get("min","–")}</td><td>{l.get("spc","–")}</td><td>{l.get("ratio","–")}</td>'
             f'<td>{s.get("peak","–")} → {s.get("end","–")}</td><td>{s.get("mass","–")} / {s.get("tgt","–")}</td>'
             f'<td>{r["frag"] or "–"}</td><td>{c.get("n50","–")} ({c.get("p50","–")} %)</td><td>{c.get("mx","–")} wu</td>'
             f'<td class="{"bad" if ej=="FAIL" else "ok"}">{ej}</td></tr>')
H.append('</tbody></table></div>')
H.append('<p class="note">"fragments (grid)": 마지막 window에서 그리드 연결성(occupancy를 한 셀 팽창한 뒤의 연결 성분) 기준으로 몸체와 분리된 입자 수 = 이탈 입자. "off-target"은 타깃 점군에서 0.5 wu 이상 떨어진 입자(몸체에 붙어 있어도 타깃 밖이면 셈) — 커버리지 오차이지 이탈이 아니다.</p>')
if any(r["qa"] for r in rows):
    H.append(f'<h2>2b. 프레임별 QA — 비디오에 떠다니는 조각·입자가 있는가 (photoreal 비디오, {len([r for r in rows if r["qa"]])}개)</h2>')
    H.append('<p class="lede">photoreal 비디오의 매 프레임에서 센 값. "raw 조각"은 같은 밀도장의 marching-cubes 등밀도면 연결 성분 수(몸체 = 1); "그린 조각"은 전달 규칙(부피 &lt; MPM cell 하나 dx³인 조각은 그리드가 해상하지 못하는 물질이므로 그리지 않음)을 적용한 뒤 실제로 화면에 남는 조각 수; "고립 입자"는 8-NN 거리가 중앙값의 3배를 넘는 입자 수(원시 입자 기준, 렌더러 미사용). "재부착"은 런 전체에서 안전망이 몸체로 되돌린 입자 수(0이면 안전망이 한 번도 작동하지 않았다).</p>')
    H.append('<div class="wrap"><table><thead><tr><th>target</th><th>frames</th><th>물리 조각 ≥1 cell (frames, max 크기)</th><th>raw 조각&gt;1 (frames, max)</th><th>그린 조각&gt;1 (frames, max)</th><th>그중 실로 이은 frames</th><th>잇지 못한 frames</th><th>sub-cell 제외 (frames, 조각 수)</th><th>고립 입자 max (frame)</th><th>고립 입자 end</th><th>재부착 (run)</th><th>fragments (end)</th></tr></thead><tbody>')
    for r in rows:
        q = r["qa"]
        if not q:
            continue
        g = r.get("gf", {})
        cls_d = "ok" if q["unbridged"] == 0 else ("warn" if q["unbridged"] <= 0.05 * q["n"] else "bad")
        cls_i = "ok" if q["iso_max"] == 0 else ("warn" if q["iso_max"] <= 20 else "bad")
        cls_g = ("ok" if g.get("f1") == "0" else "bad") if g else ""
        gtxt = f'{g["f1"]} ({g["mxc"]} cells)' if g else "–"
        H.append(f'<tr><td>{r["t"]}</td><td>{q["n"]}</td><td class="{cls_g}">{gtxt}</td><td>{q["raw_gt1"]} ({q["raw_max"]})</td><td>{q["drawn_gt1"]} ({q["drawn_max"]})</td><td>{q["bridged"]}</td><td class="{cls_d}">{q["unbridged"]}</td>'
                 f'<td>{q["drop_frames"]} ({q["drop_total"]})</td><td class="{cls_i}">{q["iso_max"]} ({q["iso_frame"]})</td><td>{q["iso_end"]}</td>'
                 f'<td>{r["reatt"]}</td><td>{r["frag"] or "–"}</td></tr>')
    H.append('</tbody></table></div>')
    H.append('<p class="note">"물리 조각": 렌더러를 쓰지 않고 입자 occupancy를 MPM cell 하나만큼 팽창한 연결 성분(재부착 안전망과 같은 기준)에서 몸체와 떨어진 성분 중 부피가 한 cell(ppc개 입자) 이상인 것의 프레임 수와 최대 크기 — 이것이 0이면 물리에는 떠다니는 몸이 없다. "그린 조각&gt;1"이 남아 있는 프레임은 등밀도면이 얇은 목(2입자 굵기 미만)에서 끊긴 것으로, 같은 프레임의 물리 조각이 0이면 이어진 재료의 끝이다.</p>')
H.append('<h2>3. 예제별 결과 — 표면 비디오(object, not particles), 입자 GIF, PBR 스틸, 손실 곡선</h2>')
H.append('<p class="lede">표면 비디오(isosurface): 입자 질량을 128³ 격자에 뿌리고 1.5 spacing 가우시안으로 흐린 밀도의 등밀도면(소스 bulk 밀도의 절반)을 ray-march한 것 — 물체가 하나의 연속 표면으로 보이고, 해상 가능한 밀도 아래의 고립 입자는 표면이 되지 않는다(이탈 수치는 별도 열). 스플랫 비디오는 이전 렌더(디스크 스플랫), 입자 GIF는 원시 입자.</p>')
for r in rows:
    t = r["t"]; a = r["arm"]; c = r["ce"]
    H.append(f'<h3>sphere → {t}</h3>')
    H.append('<div class="kv">' + "".join(
        f'<div><span>{k}</span><span>{v}</span></div>' for k, v in [
            ("chamfer / silIoU / hole", f'{a.get("chamfer","–")} / {a.get("silIoU","–")} / {a.get("hole","–")}'),
            ("wall (min) · s/commit", f'{a.get("min","–")} · {r["lo"].get("spc","–")}'),
            ("fragments (grid) · off-target > 0.5 wu", f'{r["frag"] or "–"} · {c.get("n50","–")} ({c.get("p50","–")} %), max {c.get("mx","–")} wu'),
            ("thin mass / tgt", f'{r["sc"].get("mass","–")} / {r["sc"].get("tgt","–")}')]) + '</div>')
    qa_path = os.path.join(ROOT, t, f"{t}_photoreal.mp4.components.txt")
    if os.path.exists(qa_path):
        qa_line = [l for l in open(qa_path, encoding="utf-8") if l.startswith("#")]
        qa_txt = qa_line[-1].lstrip("# ").strip() if qa_line else ""
        H.append('<div class="grid">' + fig(t, f"{t}_photoreal.mp4", f"{t} — photoreal (Filament PBR, IBL + sun, soft shadows; marching-cubes isosurface of the same density). Frame QA: {qa_txt}") + '</div>')
    H.append('<div class="grid">' + fig(t, f"{t}_surface.gif", f"{t} — isosurface of the particle density (two azimuths, target outline)") + fig(t, f"{t}_splat.gif", f"{t} — disk splats (the previous surface render, same frames)") + fig(t, f"{t}_particles.gif", f"{t} — raw particles (same frames)") + '</div>')
    H.append('<div class="grid4">' + fig(t, f"{t}_target_pbr_az35.png", "target (az 35°)") + fig(t, f"{t}_render_pbr_az35.png", "delivered (az 35°)") + fig(t, f"{t}_target_pbr_az215.png", "target (az 215°)") + fig(t, f"{t}_render_pbr_az215.png", "delivered (az 215°)") + '</div>')
    H.append('<div class="grid">' + fig(t, f"{t}_loss.png", "window loss and D_vol vs commit; loss vs wall-clock") + '</div>')
for key, title in (("ejection_html", "4. Mass ejection — 원인 조사와 시도한 메커니즘"), ("render_html", "5. 렌더 기울기가 물리를 바꾼다 — 인과 증명"), ("material_html", "6. 물성이 궤적을 바꾼다"), ("speed_html", "7. 속도"), ("summary_html", "8. 요약과 다음 단계"), ("viewer_html", "9. 3D 뷰어에서 보기")):
    if G.get(key):
        H.append(f'<h2>{title}</h2>' + G[key])
H.append('</main>')
open(os.path.join(ROOT, "index.html"), "w", encoding="utf-8").write("\n".join(H))
print("report built:", len(rows), "examples")
strip = lambda h: re.sub(r"<[^>]+>", "", h or "")
md = [f"# {strip(TITLE)}", "",
      f"Artifact (surface videos, PBR stills, loss curves): {G.get('artifact_url', '')}", "",
      "| target | chamfer | silIoU | hole | commits | min | s/commit | loss × | sparse peak → end | thin mass / tgt | fragments (grid) | re-attachments | off-target > 0.5 wu | max off-target | G4 ejection |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    a, s, l, c = r["arm"], r["sc"], r["lo"], r["ce"]
    ej = "FAIL" if "G4_ejection=FAIL" in r["gate"] else ("PASS" if r["gate"] else "–")
    md.append(f'| {r["t"]} | {a.get("chamfer","–")} | {a.get("silIoU","–")} | {a.get("hole","–")} | {l.get("commits","–")} | {a.get("min","–")} | {l.get("spc","–")} | {l.get("ratio","–")} | {s.get("peak","–")} → {s.get("end","–")} | {s.get("mass","–")} / {s.get("tgt","–")} | {r["frag"] or "–"} | {r["reatt"]} | {c.get("n50","–")} ({c.get("p50","–")} %) | {c.get("mx","–")} wu | {ej} |')
if any(r["qa"] for r in rows):
    md += ["", "## Frame QA (photoreal videos)", "",
           "Per frame: raw marching-cubes components of the blurred density (body = 1); drawn components after the deliverable rule (components with volume < one MPM cell dx³ are not drawn); isolated particles = 8-NN distance > 3 × median (raw particles, no renderer). Re-attachments = particles the safety net returned to the body over the run.", "",
           "| target | frames | physical fragments >= 1 cell (frames, max) | raw comps > 1 (frames, max) | drawn comps > 1 (frames, max) | bridged by filament (frames) | drawn > 1 and not bridged (frames) | sub-cell dropped (frames, comps) | isolated max (frame) | isolated end | re-attachments | fragments (end) |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        q = r["qa"]; g = r.get("gf", {})
        if q:
            gtxt = f'{g["f1"]} ({g["mxc"]} cells)' if g else "–"
            md.append(f'| {r["t"]} | {q["n"]} | {gtxt} | {q["raw_gt1"]} ({q["raw_max"]}) | {q["drawn_gt1"]} ({q["drawn_max"]}) | {q["bridged"]} | {q["unbridged"]} | {q["drop_frames"]} ({q["drop_total"]}) | {q["iso_max"]} ({q["iso_frame"]}) | {q["iso_end"]} | {r["reatt"]} | {r["frag"] or "–"} |')
for key, title in (("assessment_html", "Assessment"), ("ejection_html", "Mass ejection"), ("render_html", "Render gradient -> physics"), ("material_html", "Material -> trajectory"), ("speed_html", "Speed"), ("summary_html", "Summary"), ("viewer_html", "Viewer")):
    md += ["", f"## {title}", "", strip(G.get(key, ""))]
open(MD_OUT, "w", encoding="utf-8").write("\n".join(md) + "\n")
print("markdown written")
