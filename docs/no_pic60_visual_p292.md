# P292: diagnostic PIC-on/off overview

**PIC off is not promoted.** The [full raw comparison](no_pic_full_p292.md)
improves tip occupancy and top density but reduces silhouette IoU from 0.973039
to 0.966396, below the 0.971 gate. These videos expose the resulting appearance;
they do not establish closed geometry or final free-particle rest.

## Local deliverables

Both H.264 videos are 1920 by 1080 at 60 fps, with visible diagnostic labels:

- `output/p292/no_pic60_overview/DIAGNOSTIC_baseline60_1080.mp4`
- `output/p292/no_pic60_overview/DIAGNOSTIC_no_pic60_1080.mp4`

They sample every fourth raw frame at dt=1/240, giving nominal real-time playback.
Baseline contains 161 samples, raw 0,4,...,640, through accepted commit 32;
PIC off contains 201 samples, raw 0,4,...,800, through accepted commit 40.
Their encoded durations are 2.683333 and 3.350000 seconds, respectively. The last
sample's physical times are 2.666667 and 3.333333 seconds. No frames are interpolated
or duplicated to equalize the clip lengths. Each archive's one-frame suffix is
omitted; neither accepted physical sequence contains an interior null hold.

The bounded scope is an overview plus native-pixel comparison crops, not the
earlier proposed full stride-one slow-motion export. Each displayed transition
spans four physical steps and cannot resolve the 19-to-20 or 20-to-1 boundary.
The all-raw phase audit in the full comparison supplies that evidence.

## Controlled rendering and provenance

The two simulations use N=300000, T=20, dt=1/240,
dx=0.3062907544 wu and a 36-cubed loss grid. Their numerical code hashes and
source/target arrays match; the only configuration difference is `commit_pic`.
The wrapper verifies those conditions and the raw audit's exact accepted index
lists before rendering, then checks the renderer's selected indices afterward.

The unchanged original renderer comes from the immutable
`work/gpu_refactor/render1080/repo` snapshot, with script SHA-256
`e7f47cb7a4c34ac6c952944cbfd0835e28dd4bb72303cea07f0ece87646a59c1`.
It uses azimuth 35 degrees, elevation 18 degrees, uniform satin ceramic with GGX
roughness 0.36 and synthetic studio lighting. Positions are the saved physical and
commit-corrected positions. The original current-density 8NN support, opacity
0.92, radius rule and normal filter remain unchanged. The renderer validates all
642/802 archived pin states before applying the original pinned normal/radius
latch. No new P292 support, normal-filter or material-normal variant is enabled.
No AI image generation, reconstruction mesh or additional hole cover is used.

Numerical rendering ran on hyde06 GPU2. The baseline started at
2026-09-26 23:45:55.729 UTC, and PIC off at 23:47:28.183 UTC. Both labeled
videos finished at 23:48:47.300 UTC: 171.57 seconds total. The renderer's own
reported times were 88.92 and 74.31 seconds; the difference includes process
startup and label encoding. This is one measurement, not a general benchmark.

The local folder contains `provenance.json`, both original renderer JSONs,
`run.log`, `artifact_verification.json`, `qa_sheet_mapping.json`, `visual_qa.json`,
every decoded final video frame, 16 inspection sheets, and eight native-pixel
comparison crops. Provenance includes source NPZ/JSON hashes, the full frozen
Python snapshot hashes, and per-video raw-index, physical-time, accepted-window
and phase mappings. The exact executed wrapper and its launcher are
`output/p292/render_no_pic60_overview.py` and `.sh`; local decoding/crop generation
is in `output/p292/qa_no_pic60_overview.py`. Server outputs remain under
`work/p292/no_pic60_overview/`. Prior original mixed60 videos and stills remain intact.

| Final local video | SHA-256 |
| --- | --- |
| Baseline | `c6d57847c35e54e16891ed9eb360eed915cd4c88b7c991025edc750be9c7e1a7` |
| PIC off | `6587b1d0a0aa00ace7f4b2b80a681b7894d330fefb9a5244154066ae378ae68d` |

## Visual inspection and limits

All 362 encoded frames were decoded and visually inspected through exhaustive
contact sheets. The final frame of each arm was also inspected at full resolution,
and paired unresized head/ear crops were inspected at raw indices
96,240,440,480,520,560,600,640. The object remains inside the full frame, labels
stay outside it, and no duplicate crossfade body or unrelated texture appears.
The material is uniform: this does not test a textured material's attachment.

Both arms retain soft, streaked or translucent growth boundaries around the early
head and forming ears; raw 96 and 240 show these at native resolution. Their
later ear profiles and head/shoulder shading differ. The PIC-off silhouette can
look coherent in this one view while failing the multi-view raw silhouette gate;
these images do not assign that aggregate error to an individual feature.
The late images also cannot certify absence of small particle motion or internal
voids. Remaining drift/reversals and density/coverage tradeoffs are reported from
raw particles in the linked full comparison, not estimated from these pixels.

The wrapper's provenance/indexing review passed before delivery. Final independent
artifact/report review passed: the reviewer verified both MP4 hashes and formats,
all 362 frame mappings and the wrapper provenance, and inspected early/late native
crops and the candidate's full endpoint. The exhaustive 362-frame inspection is
the renderer agent's recorded QA, not an independent repeat by the reviewer.
No defaults or physics settings were promoted.
