---
name: arm-naming
description: "User decision 2026-10-21 — never call the λ=0 twin \"render OFF / physics only\"; name the comparison arms by what they contain (dF_c only / dF_c + layer u / dF_c + layer u + render loss (full) / Chapter 3 F-control)"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-10-21T20:36:49.840Z
---

User, 2026-10-21: "Render ON/OFF가 헷갈리잖아. 표면층 제어 u, Render ON, Control only로 가던가 뭐 이렇게 가야지."

**Why:** the λ=0 twin still has the layer control u (driven by physics gradients only), so calling it "render OFF / physics only" misleads. A third arm, `--u_off --render_weight_scale 0`, exists too.

**How to apply** (figures, videos, docs, Chapter 4, paper), naming each arm by its controls and terms:
- **"dF_c only"**: no render loss, no u (`--u_off --render_weight_scale 0`).
- **"dF_c + layer u"**: λ=0, u kept. This is the twin used in all render-effect comparisons so far (P / W arms).
- **"dF_c + layer u + render loss (full)"**: the render arm (L / V).
- **"Chapter 3 F-control"**: the user's prior TVCG 2025 method (Xu et al.; our XU runs).

Measured 2026-10-21 (bunny 300k): "dF_c only" vs "full" look nearly identical. The scatter-free morph comes from the physics objective, not from the render. The render's effect is sub-pitch on rims (Boundary IoU −16 %).
