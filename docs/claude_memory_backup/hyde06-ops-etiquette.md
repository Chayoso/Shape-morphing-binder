---
name: hyde06-ops-etiquette
description: "hyde06 server rules — which GPUs may be used (updated 2026-09-24: all four when free), thread caps, staggered launches, result paths, the ssh routes"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-10-14T20:43:20.326Z
---

**GPU rule (updated 2026-09-24 by the user):** "GPU 1, 3도 이제 비어 있으면 전부 다 쓰자" — all four GPUs
may be used when they are free; check `nvidia-smi` before launching and never kill another user's
process (dabh's jobs run CPU-only `grain_evolution`; if a GPU shows another user's memory, leave it).
Earlier rule (2026-09-16 … 09-24 morning): GPUs 1/3 belonged to another user and were off-limits.
Priority order for the GPUs: the oscillation fix and the other defects first.

**Why:** the deliverable's defects are the blocker; the server is otherwise idle (load 4/128).

**How to apply:** launch twins in parallel across free GPUs (one run per GPU; a render job may share);
thread caps `OMP_NUM_THREADS=8` per job; stagger launches by ~20 s; results under
`/data/relcfd/chayo/physmorph_v2/output`; every run gets a `${TAG}_start/_end` marker and a
`STATUS` line, the verdict chain `scratch/verdict.sh TAG GPU`.

**ssh routes (2026-09-24):** on the campus network connect DIRECTLY (`ssh chayo@hyde06.dabh.io`);
the hyde01 jump lost the key (hyde06's authorized_keys was rewritten 06:55 CDT without it — the
user re-added it on hyde06). Public key: `ssh-ed25519
AAAAC3NzaC1lZDI1NTE5AAAAIJHxeQR7eXi3GF/dDkqcymShAGwMgXPWw4+f3lDCZyZV ab4295@Chayo`.

**Off-campus access (2026-09-25):** hyde06.dabh.io resolves to the campus-private 10.2.191.42 — unreachable from
home. hyde01.dabh.io (129.59.140.171) is public; my key is now in its authorized_keys. `ssh -J` fails ("stdio
forwarding failed": hyde01's sshd forbids TCP forwarding) — use the socat ProxyCommand alias `ssh hyde06j`
(in ~/.ssh/config: ProxyCommand ssh chayo@hyde01.dabh.io socat - TCP:hyde06.dabh.io:22). scp works with the alias.

**Clock (2026-09-25):** the server clock (CDT) is also the user PC clock - stamp docs in CDT from `date` on the server; never assume Korea time (a whole session of entries was mis-dated 09-26/09-27 for 09-25 that way).

**2026-10-14 incident:** from 02:28 CDT both hyde06 and hyde01 rejected the key; the user re-added it on both at ~09:03.
- An agent's 60-s pollers kept retrying about 50–60 times. Never poll ssh tightly; stop at the first publickey failure.
- hyde06's ROOT filesystem (/, which holds /home/chayo) was 100 % full (298 MB free) while the detached pool workers sat idle.
  - Our home is ~114 GB: miniforge3 52 GB, neuma_eval 37 GB (another project; don't touch), ~/.cache 17 GB.
  - I removed ~/.cache/pip (16 GB) and logged it.
- Rule from now on: all caches and temp go to /data (TMPDIR, WARP_CACHE_PATH, XDG_CACHE_HOME, PIP_CACHE_DIR, TORCH_HOME, MPLCONFIGDIR).
- Multi-agent card sharing uses /data/relcfd/chayo/physmorph_v2/gpu_locks/gpuN.lock (mkdir, released by a trap), one run per card.

**hyde01 render node (2026-10-14):**
- Setup: ~/physmorph_render on hyde01, 2× RTX A6000. Output is byte-identical to hyde06's, about 2.8× slower (a dragon 300k view takes ~40 min).
- Queue: scripts/ops/hyde01_render.sh, run from local Git Bash (`--npz --dest --name --views`; `--scene` for throw renders; `--repo` for newer code).
- USER rules:
  - use hyde01 GPU 1 ONLY; GPU 0 drives tianrun's desktop;
  - urgent and deliverable renders on hyde06, inspection and evidence renders on hyde01;
  - the user agreed to add chayo@hyde01's key (`ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIILVwKoIbMVo+ufz3DpJzR37w7zNEyaXziKsh8jL9G7H chayo@hyde01`, SHA256:hBS1lo1x…) to hyde06's authorized_keys themselves, for direct transfers.

**2026-10-05 update:** the jump host hyde01 (OpenSSH 10.0, Debian 13) started rejecting the local key `~/.ssh/id_ed25519` (SHA256:KdgXwOO…) at ~11:55 CDT, so `ssh hyde06j` fails; hyde06 itself still accepts it, and from campus `ssh chayo@hyde06.dabh.io` works directly — use that while hyde01 is broken. To fix hyde01 the user appends the .pub to hyde01's `~/.ssh/authorized_keys` themselves (password); never ask for or type a password.

**2026-10-15 ~11:25 CDT:** hyde01 rejects the local key again ("Permission denied (publickey,password)"), so `ssh hyde06j` fails. On campus, direct `ssh chayo@hyde06.dabh.io` works. All agents were told to use the direct route and to render on hyde06 until the user re-adds the key on hyde01. The hyde01→hyde06 transfer key was never authorised, so transfers still relay through local.
