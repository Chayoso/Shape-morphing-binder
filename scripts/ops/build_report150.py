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
targets = [os.path.basename(p) for p in sorted(glob.glob(os.path.join(ROOT, "*"))) if os.path.isdir(p)]
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


rows = []
for t in targets:
    d = os.path.join(ROOT, t)
    arm, gate = parse_arm(rd(os.path.join(d, f"{t}_arm.txt")))
    rows.append(dict(t=t, arm=arm, gate=gate, sc=parse_scatter(rd(os.path.join(d, f"{t}_scatter.txt"))),
                     ce=parse_census(rd(os.path.join(d, f"{t}_census.txt"))), lo=parse_loss(rd(os.path.join(d, f"{t}_loss.txt"))),
                     tm=rd(os.path.join(d, f"{t}_time.txt")).strip(),
                     frag=rd(os.path.join(d, f"{t}_frag.txt")).strip()))


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
H.append('<h2>3. 예제별 결과 — 표면 비디오(object, not particles), 입자 GIF, PBR 스틸, 손실 곡선</h2>')
H.append('<p class="lede">표면 비디오: GPU z-buffer 디스크 스플랫 → 깊이 평활 → 법선 → GGX 셰이딩, 두 방위각, 타깃 윤곽선 오버레이. 입자 GIF는 같은 프레임의 원시 입자.</p>')
for r in rows:
    t = r["t"]; a = r["arm"]; c = r["ce"]
    H.append(f'<h3>sphere → {t}</h3>')
    H.append('<div class="kv">' + "".join(
        f'<div><span>{k}</span><span>{v}</span></div>' for k, v in [
            ("chamfer / silIoU / hole", f'{a.get("chamfer","–")} / {a.get("silIoU","–")} / {a.get("hole","–")}'),
            ("wall (min) · s/commit", f'{a.get("min","–")} · {r["lo"].get("spc","–")}'),
            ("fragments (grid) · off-target > 0.5 wu", f'{r["frag"] or "–"} · {c.get("n50","–")} ({c.get("p50","–")} %), max {c.get("mx","–")} wu'),
            ("thin mass / tgt", f'{r["sc"].get("mass","–")} / {r["sc"].get("tgt","–")}')]) + '</div>')
    H.append('<div class="grid">' + fig(t, f"{t}_surface.gif", f"{t} — surface (two azimuths, target outline)") + fig(t, f"{t}_particles.gif", f"{t} — particles (same frames)") + '</div>')
    H.append('<div class="grid4">' + fig(t, f"{t}_target_pbr_az35.png", "target (az 35°)") + fig(t, f"{t}_render_pbr_az35.png", "delivered (az 35°)") + fig(t, f"{t}_target_pbr_az215.png", "target (az 215°)") + fig(t, f"{t}_render_pbr_az215.png", "delivered (az 215°)") + '</div>')
    H.append('<div class="grid">' + fig(t, f"{t}_loss.png", "window loss and D_vol vs commit; loss vs wall-clock") + '</div>')
for key, title in (("ejection_html", "4. Mass ejection — 원인 조사와 시도한 메커니즘"), ("speed_html", "5. 속도 — 150k 10분 목표"), ("summary_html", "6. 요약과 다음 단계"), ("viewer_html", "7. 3D 뷰어에서 보기")):
    if G.get(key):
        H.append(f'<h2>{title}</h2>' + G[key])
H.append('</main>')
open(os.path.join(ROOT, "index.html"), "w", encoding="utf-8").write("\n".join(H))
print("report built:", len(rows), "examples")
strip = lambda h: re.sub(r"<[^>]+>", "", h or "")
md = ["# 150k gallery report (2026-09-16) — sphere → 10 targets, 150k `--ppc 8`, render arm", "",
      f"Artifact (surface videos, PBR stills, loss curves): {G.get('artifact_url', '')}", "",
      "| target | chamfer | silIoU | hole | commits | min | s/commit | loss × | sparse peak → end | thin mass / tgt | fragments (grid) | off-target > 0.5 wu | max off-target | G4 ejection |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    a, s, l, c = r["arm"], r["sc"], r["lo"], r["ce"]
    ej = "FAIL" if "G4_ejection=FAIL" in r["gate"] else ("PASS" if r["gate"] else "–")
    md.append(f'| {r["t"]} | {a.get("chamfer","–")} | {a.get("silIoU","–")} | {a.get("hole","–")} | {l.get("commits","–")} | {a.get("min","–")} | {l.get("spc","–")} | {l.get("ratio","–")} | {s.get("peak","–")} → {s.get("end","–")} | {s.get("mass","–")} / {s.get("tgt","–")} | {r["frag"] or "–"} | {c.get("n50","–")} ({c.get("p50","–")} %) | {c.get("mx","–")} wu | {ej} |')
for key, title in (("assessment_html", "Assessment"), ("ejection_html", "Mass ejection"), ("speed_html", "Speed"), ("summary_html", "Summary"), ("viewer_html", "Viewer")):
    md += ["", f"## {title}", "", strip(G.get(key, ""))]
open(MD_OUT, "w", encoding="utf-8").write("\n".join(md) + "\n")
print("markdown written")
