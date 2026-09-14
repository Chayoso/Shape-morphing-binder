"""Localhost-only native-image monitor for the geometric control diagnostic."""
from __future__ import annotations

import io
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image


PAGE = b'''<!doctype html><html><head><meta charset="utf-8">
<title>PhysMorph geometry audit</title><style>
body{margin:0;background:#f2f2f2;color:#20262b;font:14px system-ui}
header{padding:14px 20px}b{margin-right:20px}main{display:flex;justify-content:center}
figure{margin:8px;width:48vw}img{width:100%;height:78vh;object-fit:contain;background:white}
figcaption{text-align:center;padding:6px}pre{white-space:pre-wrap;margin:10px 20px}
</style></head><body><header><b>PhysMorph: geometry diagnostic</b>
F_geom from MPM motion | fixed opacity | native CUDA render</header><main>
<figure><img id="current"><figcaption>Current physical state</figcaption></figure>
<figure><img id="target"><figcaption>Target observation</figcaption></figure></main>
<pre id="status">Connecting...</pre><script>
let version=-1;async function poll(){try{
const packet=await(await fetch('/snapshot',{cache:'no-store',signal:AbortSignal.timeout(8000)})).json();
const s=packet.status;if(s.version!==version){
const a=new Image(),b=new Image();a.src='data:image/png;base64,'+packet.current;
b.src='data:image/png;base64,'+packet.target;await Promise.all([a.decode(),b.decode()]);
document.getElementById('current').src=a.src;document.getElementById('target').src=b.src;
version=s.version;}document.getElementById('status').textContent=JSON.stringify(s,null,2);
}catch(e){document.getElementById('status').textContent='Disconnected; reconnecting...'}
setTimeout(poll,200)}poll();</script></body></html>'''


def encode_png(rgb):
    data = io.BytesIO()
    Image.fromarray((np.clip(rgb, 0, 1)*255).round().astype(np.uint8)).save(data, format="PNG")
    return data.getvalue()


class GeometryMonitor:
    """Publish only actual raster images; observation and display share GaussViews."""
    def __init__(self, port):
        self.lock = threading.Lock()
        self.current = self.target = b""
        self.status = {"version": 0, "phase": "starting"}
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                path = self.path.split("?", 1)[0]
                with owner.lock:
                    if path == "/":
                        body, kind = PAGE, "text/html; charset=utf-8"
                    elif path == "/status":
                        body, kind = json.dumps(owner.status, allow_nan=False).encode(), "application/json"
                    elif path == "/snapshot":
                        if not owner.current:
                            self.send_error(503); return
                        body = json.dumps({"status": owner.status,
                            "current": base64.b64encode(owner.current).decode("ascii"),
                            "target": base64.b64encode(owner.target).decode("ascii")}, allow_nan=False).encode()
                        kind = "application/json"
                    elif path in ("/current.png", "/target.png"):
                        body = owner.current if path == "/current.png" else owner.target
                        kind = "image/png"
                        if not body:
                            self.send_error(503); return
                    else:
                        self.send_error(404); return
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def publish(self, rgb, target, status):
        current, target_png = encode_png(rgb), encode_png(target)
        with self.lock:
            self.current, self.target = current, target_png
            self.status = {**status, "version": self.status["version"]+1}

    def update_status(self, **fields):
        with self.lock:
            self.status.update(fields)
            self.status["version"] += 1

    def close(self):
        self.server.shutdown()
        self.server.server_close()
