---
name: figure-style-ppt
description: "User's figure style (2026-10-23): PowerPoint look — white cards with soft drop shadows, pop-out row strips, glossy blue particle bodies, orange heat grids, red bold-italic callouts with leader line to a shadowed inset, coloured bold serif column headers"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-10-23T19:45:09.804Z
---

On 2026-10-23 the user sent two of their own slides and said "좀 더 powerpoint 스타일로 … 음영도 좀 넣어주고 … 이런게 내 스타일이야" and "강조하는 부분은 음영이랑 그림자도 넣고".

The style:
- White ground. Emphasised panels sit on white cards with a soft drop shadow.
- Rows are white strips with a shadow, and the body pops out over the strip edges.
- Column headers are bold serif, coloured per arm: grey for others, blue for the middle arm, green for ours. Row labels are italic and rotated.
- Schematic particle bodies are glossy blue spheres. Grids have thin blue lines with an orange heat fill.
- Callouts are red bold italic ("Issue: …" / "Resolve: …"). A navy leader line ends in a dot and points to a shadowed inset.
- Formulas sit centred under each panel. Colourbars are thin, labelled high/low.

**Update, later on 2026-10-23:** the user dropped the shadows: "음영 저거 없애주라 그림들에는". So: NO drop shadows or shading on figures. Keep Times, but keep the text very short (a few words per label, no sentences inside a figure). The user also asked:
- that a mechanism figure show its parts as separate zones working together (render / physics / layer);
- that it be built around one concrete example.

The user also preferred the old plain Fig 4.3 over the PPT-styled version. Do not restyle result charts.

**Why:** the plain matplotlib/TikZ figures read as flat and list-like to the user ("나열식"). This is their own visual identity across TVCG/PhysMorph-GS slides and papers.

**How to apply:**
- Every new paper or dissertation figure follows this style.
- Chapter 4 text stays Times New Roman, overriding the slide's sans font.
- Result renders keep the studio look ([[render-look-default]]). Only schematic bodies become glossy blue.
- Arm names follow [[arm-naming]].
