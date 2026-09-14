"""Localhost-only atomic snapshots of two independently progressing experiments."""
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

from .geometric import encode_png


PAGE = '''<!doctype html><meta charset="utf-8"><title>PhysMorph live comparison</title>
<style>body{margin:0;background:#f5f5f3;color:#25282b;font:15px system-ui}
header{padding:16px 22px}main{display:flex;gap:8px;padding:0 12px}
figure{margin:0;flex:1;min-width:0}img{width:100%;background:white}
figcaption{text-align:center;padding:10px}pre{white-space:pre-wrap;font:13px system-ui;padding:0 16px}
#connection{float:right}small{display:block;margin-top:6px;color:#60666b}</style>
<header><b>PhysMorph · sphere → bunny</b><span id="connection">Connecting…</span>
<small>실험 모니터 · 각 실행이 수락한 최신 물리 상태 · 최종 품질 검증 전</small></header>
<main><figure><img id="physics"><figcaption>3D 지도</figcaption><pre id="physics_status"></pre></figure>
<figure><img id="guided"><figcaption>Rendering → physics</figcaption><pre id="guided_status"></pre></figure>
<figure><img id="target"><figcaption>목표 bunny</figcaption><pre id="setup"></pre></figure></main>
<script>let version=-1;
async function poll(){try{
const p=await(await fetch('/snapshot',{cache:'no-store',signal:AbortSignal.timeout(8000)})).json();
if(p.version!==version){const keys=['physics','guided','target'];
const images=keys.map(k=>{const im=new Image();im.src='data:image/png;base64,'+p[k];return im});
await Promise.all(images.map(im=>im.decode()));keys.forEach((k,i)=>document.getElementById(k).src=images[i].src);
for(const k of ['physics','guided'])document.getElementById(k+'_status').textContent=p.status[k];
document.getElementById('setup').textContent=p.status.setup;version=p.version;}
document.getElementById('connection').textContent='연결됨 · '+new Date().toLocaleTimeString();
}catch(e){document.getElementById('connection').textContent='연결 재시도 중…'}setTimeout(poll,1000)}poll();</script>
'''.encode('utf-8')


class LivePairMonitor:
    def __init__(self, port=8776):
        self.lock, self.packet = threading.Lock(), None
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                path = self.path.split('?', 1)[0]
                if path == '/':
                    body, kind = PAGE, 'text/html; charset=utf-8'
                elif path in ('/snapshot', '/status'):
                    with owner.lock:
                        if owner.packet is None:
                            self.send_error(503); return
                        packet = owner.packet if path == '/snapshot' else {
                            'version': owner.packet['version'], 'status': owner.packet['status']}
                        body = json.dumps(packet, allow_nan=False).encode()
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

    def publish(self, physics, guided, target, status):
        images = {k: base64.b64encode(encode_png(v)).decode('ascii')
                  for k, v in zip(('physics', 'guided', 'target'), (physics, guided, target))}
        with self.lock:
            version = 1 if self.packet is None else self.packet['version']+1
            self.packet = {**images, 'status': dict(status), 'version': version}

    def close(self):
        self.server.shutdown(); self.server.server_close()
