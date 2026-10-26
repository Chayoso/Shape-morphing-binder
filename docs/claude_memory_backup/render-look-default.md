---
name: render-look-default
description: "User rule 2026-10-06 night: until told otherwise, every render uses the original studio look (light-grey body on the grey gradient backdrop), not blue/matte/raw-Gaussian variants"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-10-21T19:53:20.827Z
---

From 2026-10-06 23:00 CDT until the user says otherwise: render everything in the ORIGINAL studio look — the D111/D115 4K video look (`StudioRaster` defaults: albedo (0.64, 0.69, 0.72) light grey, GGX, the three studio lights, the grey gradient backdrop, display discs with twice-averaged normals; `child_detail_morph.draw()` / `morph_4k.py`).

**Why:** the user tried blue, matte, raw-Gaussian and disc+grain variants for the pipeline figure (docs/figures/pipeline_overview_final_{gs,discgs,studio}) and decided the original look is what they want ("내가 다른 말하기 전 까지 우리 원래 render인 회색 배경에 하얀 mesh로 랜더 하자").

**How to apply:** no recolouring, no matte shading, no raw-particle splats, no white/transparent backdrop unless the user asks for that specific variant again. For figures, cut bodies from the studio render with the grey backdrop as is, or ask. Related: [[doc-only-current-gsplat]].

**Update 2026-10-21 (user, for the demo):** "배경이랑 조명도 꾸며야 할 거 같아". The user showed paper figures as references:
- a glossy black bunny on a white pedestal with a blurred HDRI;
- coloured glossy letters (red, blue, yellow, green) on a wooden table against a mottled wall;
- a matte blue dragon on wood.

For the demo and the paper video, the display surface (marching cubes of the zbs field) is rendered in Blender Cycles with these looks (DEMO agent; scripts/demo/). The plain studio look stays for analysis figures (RGV, forensics) unless the user says otherwise.
- **14:15 same day, REVERTED.** The Blender look was abandoned: the surface looked bumpy and the scene looked composited. The user: "현재 결과들을 이전 하얗고 gray 배경에서 '많이 보여주는' 거에 집중하자. demo 렌더링 중지하고 지워버려."
  - The studio look stays the default.
  - For demos, show many results (grid montages, render on/off side by side) rather than a polished environment.
  - The Blender outputs were deleted.
  - Still in git: the Blender pipeline in scripts/demo/ (commits 3a89720 and f3dc217 on settled-base). Blender 4.2.3 stays at tools/blender on hyde06.
  - The demo deliverable is output/results_2026-10-21/demo_studio/ (README_ko.md).
  - Rendering gotchas:
    - Two render_deliverable.py processes fit on one card. Three ran out of memory (warp allocation). Armadillo also failed beside bunny2dragon.
    - The scene renderer at the default dist crops a dragon: use +dist=1.6. Homer needs +dist=1.35.
