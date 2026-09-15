# Live viewer — decoupled monitor (2026-09-14)

The original viewer (`physmorph/viewer/server.py::LiveServer(port)`) is an HTTP server
*inside* the simulation process: it dies with the run, needs one port per run, shows
nothing between runs, cannot scrub a finished run, and the ssh tunnel is re-made by hand.
This design keeps that legacy mode intact and adds a persistent, file-backed path so one
browser tab can stay open on the side and show every run on hyde06, live or replayed.

## Architecture

```
hyde06                                                          local (Windows)
--------------------------------------------------------------  --------------------------------
sim process A  LiveServer.to_dir(ROOT/runA) ─┐                   scripts/viewer_tunnel.py
sim process B  LiveServer.to_dir(ROOT/runB) ─┤ FileHub writes    keeps  ssh -N -L 8765:localhost:8765
   …          (or --live_dir ROOT)          ─┘        │           -J hyde01 hyde06  alive, probes
                                                      ▼           /runs every 10 s, --open browser
ROOT/                                                 │                    │
  runA/ meta.json target.bin state.bin history.json   │                    ▼
        commits/000001.bin … restart.flag             │           http://127.0.0.1:8765/
  runB/ …                                             │             /            live.html  ?run=<name>
                                                      ▼             /quad        4 panes (checkboxes)
scripts/viewer_serve.py --root ROOT --port 8765  ◄────┘             /compare     L/R panels + telemetry
  (separate long-lived process, stdlib only)
```

Simulation processes never open a port; they only write files. The server reads files;
it never talks to a simulation. Either side can restart without the other noticing.

## Run-directory file protocol (`physmorph/viewer/filehub.py`)

`FileHub(run_dir, max_commits=2000, history_throttle=0.5)` has the same surface as
`Hub` (`lock`, `state`, `meta`, `target`, `history`, `restart`, `publish`, `snap`,
`hist_json`), so `LiveServer.begin_run` and its `(on_commit, on_iter)` callbacks are
unchanged. Packets are the unchanged `pack_state` bytes (protocol v2/v3).

| file | written when | content |
|---|---|---|
| `meta.json` | `hub.meta = …` (begin_run) | run metadata JSON (`arm`, `n`, `extent`, …) |
| `target.bin` | `hub.target = …` (begin_run) | target packet |
| `state.bin` | every `publish` | latest packet (iter or commit) |
| `history.json` | every publish, rewritten at most every 0.5 s; **non-iter packets always flush** | header rows, 800-row cap (same as `Hub`) |
| `commits/NNNNNN.bin` | every packet whose header `phase != "iter"` (initial state, commits) | that packet; `NNNNNN` = zero-padded `seq`; oldest deleted past `max_commits` |
| `restart.flag` | `POST /r/<name>/restart` (server) | consumed by `FileHub.poll_restart()` → sets `hub.restart` |

All writes are temp-file + `os.replace` in the same directory (atomic on local
POSIX/NTFS). Disk errors are counted in `hub.io_errors` and never raised into the run.

Decisions worth knowing:
- `seq` is per process (`LiveServer.seq`), so **use a fresh run directory per process**
  (`--live_dir` defaults the name to a timestamp). Several `begin_run` calls in one
  process (loop mode, multi-arm) share the directory: `meta.json` is overwritten and
  commit files accumulate with increasing `seq`; the packet header `run` field tells
  them apart and `live.html` reloads meta when `run` changes.
- `hub.restart` is an `Event` subclass whose `wait()` also polls `restart.flag` every
  0.5 s, so a run holding on `live.hub.restart.wait()` (no `publish()` to poll for it)
  still reacts to the viewer's restart button.
- `poll_restart()` is also called inside every `publish()` (one `os.path.exists`).
- Disk budget: a commit packet is the full state (≈4 MB for 20 k particles with 4
  children + grid diagnostics); 300 commits ≈ 1.2 GB per run; the 2000-file cap bounds
  a run at ≈8 GB. Delete old run directories under the root when done.

## Server (`scripts/viewer_serve.py`)

Stdlib `ThreadingHTTPServer`; every response is `Cache-Control: no-store`.

| endpoint | returns |
|---|---|
| `GET /`, `/quad`, `/compare` | `live.html`, `quad.html`, `compare.html` |
| `GET /runs` | `[{name, arm, mtime, age, seq, run, phase, commits, live}]`, newest first; `live` = `state.bin` modified within 120 s |
| `GET /r/<name>/meta` `/target` `/state` `/history` | the files |
| `GET /r/<name>/commits` | available commit `seq`s, ascending |
| `GET /r/<name>/commit/<seq>` | that packet |
| `POST /r/<name>/restart` | creates `restart.flag` |

`<name>` must be exactly one path component that exists directly under `--root` and
contains `meta.json` (`..`, slashes, backslashes, symlinks escaping the root → 404).
Reads validate the packet (`4 + hlen + 4·payload_floats == size`) or the JSON and retry
6 × 40 ms; a missing or still-inconsistent file answers **503**, never a crash.

### On hyde06 (AGENTS.md rule 1 launch conventions)

```bash
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
cd ~/physmorph_v2; mkdir -p /data/relcfd/chayo/physmorph_v2/output/live
cd ~/physmorph_v2; setsid nohup $PY scripts/viewer_serve.py \
    --root /data/relcfd/chayo/physmorph_v2/output/live --port 8765 \
    > output/viewer_serve.log 2>&1 < /dev/null &
```

The server binds `127.0.0.1` (tunnel only); it needs no GPU and no torch. Leave it
running; it survives every simulation process.

### Pointing a run at it

```bash
cd ~/physmorph_v2; setsid nohup env CUDA_VISIBLE_DEVICES=<n> $PY scripts/live_viewer.py \
    --live_dir /data/relcfd/chayo/physmorph_v2/output/live --run_name 20260914_bunny_gs \
    … > output/live_bunny.log 2>&1 < /dev/null &
```

`--live_dir DIR` replaces `--port`: no HTTP server is opened, packets go to
`DIR/<run_name>/` (`--run_name` default = timestamp). In code:

```python
live = LiveServer.to_dir(run_dir)          # FileHub, no port
live = LiveServer(port)                    # legacy in-process HTTP (unchanged)
live = LiveServer(port, hub=FileHub(dir))  # both at once
on_commit, on_iter = live.begin_run(name, src, tgt, prm, cfg, sigma0)
```

## Local side

```powershell
python scripts/viewer_tunnel.py --open       # tunnel + browser once /runs answers
python scripts/viewer_tunnel.py --probe      # exit 0 iff the viewer is reachable
```

`viewer_tunnel.py` (stdlib only) keeps `ssh -N -L 8765:localhost:8765 -J chayo@hyde01.dabh.io
chayo@hyde06.dabh.io` alive: restarts the child on exit with exponential backoff (2 s … 60 s),
probes `http://127.0.0.1:8765/runs` every 10 s and prints one status line per probe
(`connected … N runs, M live` / `reconnecting in Ns` / `waiting: tunnel up but /runs not
answering`). Ctrl-C terminates ssh. `--port/--remote-port/--host/--jump/--ssh` override.

## Pages

**`/` (live.html)** — same 2-D canvas splat renderer and telemetry charts as before.
- Run selector (top-left) from `/runs`, refreshed every 5 s; `●` = live, `○` = finished.
  Default = newest live run, else newest. Choosing a run reloads meta/target/history and
  sets `?run=<name>` in the URL, so a tab can be bookmarked per run.
- **Replay scrub**: the range slider spans `/r/<name>/commits`. Moving it pauses polling
  and draws `/r/<name>/commit/<seq>`; the charts are trimmed to rows with
  `seq ≤ that commit`. `play` auto-advances at 6 fps (button becomes `stop`); `live`
  resumes following `state.bin`.
- `age N s` next to the telemetry = time since the last new packet (red past 30 s), so a
  stalled run is visible; it reads `replay` while scrubbing.
- Legacy mode is automatic: with no `?run=` and `/runs` answering 404 (in-process
  `LiveServer(port)`), the page uses the root endpoints and hides the selector/scrub.

**`/quad`** — up to four panes; tick runs in the top strip (from `/runs`, refreshed every
10 s; default the four newest live runs, else the newest). `?runs=a,b,c` preselects.
`/quad?ports=8765,8766,…` keeps the old one-port-per-run mode.

**`/compare`** — left/right run selectors, two independent panels (independent cameras),
and a telemetry strip with `phase, commit, d_vol, d_render, kin, move, Jmin` from the last
row of each run's `/history` (refreshed every 2 s). `?l=<run>&r=<run>` preselects.

## Limitations

- Polling, not push: `state.bin` every 150 ms, `/runs` every 5–10 s, history every ≥400 ms.
- 2-D canvas splats (projected Gaussian ellipses), no WebGL; large runs (>100 k
  primitives) get slow to draw on the client.
- One packet per iteration on disk: iter packets overwrite `state.bin`, so replay is at
  commit granularity only (that is what `commits/` stores).
- `history.json` keeps the last 800 rows; older telemetry of a long run is only in the
  run's own log/JSON outputs.
- The server does not watch a run directory for deletion mid-request; a deleted run
  simply 503s/404s until the page picks another one.
- No authentication: keep `--bind 127.0.0.1` and reach it through the ssh tunnel.
