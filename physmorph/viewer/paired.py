"""Atomic, localhost-only playback of a reviewed physical comparison."""
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading


PAGE = '''<!doctype html><meta charset="utf-8"><title>PhysMorph comparison</title>
<style>body{margin:0;background:#f5f5f3;color:#25282b;font:15px system-ui}
header{padding:16px 22px}main{display:flex;gap:8px;padding:0 12px}
figure{margin:0;flex:1;min-width:0}img{width:100%;object-fit:contain;background:white}
figcaption{text-align:center;padding:10px}footer{padding:14px 22px}
input{width:65vw}button{padding:6px 18px;margin-right:12px}#result{margin-top:8px}</style>
<header><b>PhysMorph · sphere → bunny</b><div id="result">Connecting…</div></header>
<main><figure><img id="physics"><figcaption>3D criterion</figcaption></figure>
<figure><img id="guided"><figcaption>Rendering guidance</figcaption></figure>
<figure><img id="target"><figcaption>Target</figcaption></figure></main>
<footer><button id="play">Pause</button><input id="scrub" type="range" min="0" value="0">
<span id="clock"></span></footer><script>
let frame=0,count=1,playing=true;
const scrub=document.querySelector('#scrub');
document.querySelector('#play').onclick=()=>{playing=!playing;document.querySelector('#play').textContent=playing?'Pause':'Play'};
scrub.oninput=()=>{frame=Number(scrub.value);playing=false;document.querySelector('#play').textContent='Play'};
async function tick(){try{
const packet=await(await fetch('/frame/'+frame,{cache:'no-store',signal:AbortSignal.timeout(8000)})).json();
const names=['physics','guided','target'];const imgs=names.map(k=>{const im=new Image();im.src='data:image/png;base64,'+packet[k];return im});
await Promise.all(imgs.map(im=>im.decode()));
names.forEach((k,i)=>document.getElementById(k).src=imgs[i].src);
count=packet.frames;scrub.max=count-1;scrub.value=packet.frame;
document.querySelector('#clock').textContent=packet.simulation_time.toFixed(3)+' s · '+packet.frame+'/'+(count-1);
document.querySelector('#result').textContent=packet.label;
if(playing)frame=(packet.frame+1)%count;
}catch(e){document.querySelector('#result').textContent='Disconnected; reconnecting…'}
setTimeout(tick,200)}tick();</script>'''.encode('utf-8')


class PairedReplay:
    def __init__(self, physics, guided, port=8776, label="Development comparison; final quality not approved"):
        physics, guided = Path(physics), Path(guided)
        manifests = [json.loads((p/"qa_manifest.json").read_text()) for p in (physics, guided)]
        if any(not str(m.get("visual_review", "")).startswith("reviewed") for m in manifests):
            raise ValueError("review every delivered frame before starting paired playback")
        if manifests[0]["frame_count"] != manifests[1]["frame_count"]:
            raise ValueError("paired trajectories must have the same frame count")
        metas = [json.loads((p/"metadata.json").read_text()) for p in (physics, guided)]
        def dt(m):
            return m.get("discretization", m.get("fixture", {}).get("discretization", {}))["dt"]
        if dt(metas[0]) != dt(metas[1]):
            raise ValueError("paired trajectories must share physical time sampling")
        arrays = [[base64.b64encode((p/name).read_bytes()).decode('ascii') for name in m["files"]]
                  for p, m in zip((physics, guided), manifests)]
        target = base64.b64encode((guided/"target.png").read_bytes()).decode('ascii')
        if (physics/"target.png").read_bytes() != (guided/"target.png").read_bytes():
            raise ValueError("paired playback requires the same target observation")
        count, timestep = len(arrays[0]), dt(metas[0])
        if any(len(a) != count for a in arrays) or count != manifests[0]["frame_count"]:
            raise ValueError("incomplete frame manifest")
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                path = self.path.split('?', 1)[0]
                if path == '/':
                    body, kind = PAGE, 'text/html; charset=utf-8'
                elif path.startswith('/frame/') and path[7:].isdigit():
                    index = int(path[7:])
                    if not 0 <= index < count:
                        self.send_error(404); return
                    body = json.dumps({'physics':arrays[0][index], 'guided':arrays[1][index],
                        'target':target, 'frame':index, 'frames':count,
                        'simulation_time':index*timestep, 'label':label}).encode()
                    kind = 'application/json'
                else:
                    self.send_error(404); return
                self.send_response(200)
                self.send_header('Content-Type', kind)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers(); self.wfile.write(body)
        self.server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown(); self.server.server_close()
