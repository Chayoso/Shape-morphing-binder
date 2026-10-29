---
name: codex-cli-environment
description: How to invoke the local Codex CLI for adversarial reviews (PowerShell only; auth caveats)
metadata: 
  node_type: memory
  type: reference
  originSessionId: f83ab3dd-27cd-4a63-8354-f08614b6a754
  modified: 2026-09-01T16:48:47.666Z
---

Codex CLI v0.144.6 is installed (`C:\Users\ok429\AppData\Roaming\npm\codex`).

- **PowerShell only**: the Git-Bash shim is broken (node resolves a mangled anaconda-prefixed path). Invoke via the PowerShell tool.
- The user's "Codex SOL 5.6 extra high" maps to: `codex exec --cd <repo> --sandbox read-only -c model="gpt-5.6-sol" -c model_reasoning_effort="xhigh"`; pipe the prompt via stdin with `Get-Content -Raw prompt.md | codex exec ... -`. (`sol-5.6` is REJECTED — the real slug per `~/.codex/models_cache.json` is `gpt-5.6-sol`; siblings: gpt-5.6-terra/luna, gpt-5.5, gpt-5.4(-mini), gpt-5.3-codex-spark.)
- Auth is a **ChatGPT account token that expires** (observed expired 2026-09-01). `codex login` is an interactive browser flow — the user must run it themselves. When expired, the run also spuriously reports "model not supported with a ChatGPT account" for sol-5.6.
- Adversarial verification pairing the user wants: Codex (sol-5.6 xhigh) + Claude Opus subagent, both in REFUTE mode with file:line findings (see [[adversarial-verification-workflow]] in AGENTS.md rule 5).
