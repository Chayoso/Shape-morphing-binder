# D164 readings

## (R1) null windows and (R2) run length, 40k

| arm | runs | judged | committed | commit_replay nulls | share | other nulls | rescued (tag would null) | minutes (median) | s / committed window (median) |
|---|---|---|---|---|---|---|---|---|---|
| V | 57 | 2989 | 2652 | 95 | 3.18 % | {} | 0 (0.00 %) | 1.53 | 2.04 |
| W | 57 | 2794 | 2448 | 107 | 3.83 % | {} | 0 (0.00 %) | 1.43 | 2.08 |

## (R3) per mesh, the D164 arm's two seeds against the tag's three (G: D162's rule)

| mesh | arm | reading | tag 97 / 99 / 101 | D164 97 / 99 | miss | the tag's 101 alone would miss |
|---|---|---|---|---|---|---|
| A | L | (missing runs) | | | | |
| A | P | (missing runs) | | | | |
| armadilo | L | (missing runs) | | | | |
| armadilo | P | (missing runs) | | | | |
| beast | L | (missing runs) | | | | |
| beast | P | (missing runs) | | | | |
| bimba | L | (missing runs) | | | | |
| bimba | P | (missing runs) | | | | |
| bob | L | (missing runs) | | | | |
| bob | P | (missing runs) | | | | |
| bunny | L | (missing runs) | | | | |
| bunny | P | (missing runs) | | | | |
| C | L | (missing runs) | | | | |
| C | P | (missing runs) | | | | |
| cheburashka | L | (missing runs) | | | | |
| cheburashka | P | (missing runs) | | | | |
| cow | L | (missing runs) | | | | |
| cow | P | (missing runs) | | | | |
| dragon | L | (missing runs) | | | | |
| dragon | P | (missing runs) | | | | |
| fandisk | L | (missing runs) | | | | |
| fandisk | P | (missing runs) | | | | |
| heart | L | (missing runs) | | | | |
| heart | P | (missing runs) | | | | |
| homer | L | (missing runs) | | | | |
| homer | P | (missing runs) | | | | |
| maxplanck | L | (missing runs) | | | | |
| maxplanck | P | (missing runs) | | | | |
| nefertiti | L | (missing runs) | | | | |
| nefertiti | P | (missing runs) | | | | |
| ogre | L | (missing runs) | | | | |
| ogre | P | (missing runs) | | | | |
| spot | L | (missing runs) | | | | |
| spot | P | (missing runs) | | | | |
| teapot | L | (missing runs) | | | | |
| teapot | P | (missing runs) | | | | |
| V | L | (missing runs) | | | | |
| V | P | (missing runs) | | | | |

- G, arm L against the tag's V: reading misses 0 of 0 []; rest misses []; pieces misses []: G HOLDS
- G, arm P against the tag's W: reading misses 0 of 0 []; rest misses []; pieces misses []: G HOLDS
- target samples identical to the tag's run of the same mesh / seed: 0 of 0 (not: [])
- arm L: two-seed mean inside the tag's 97 / 99 / 101 range on 0 of 0 (mesh x reading) cells
- arm P: two-seed mean inside the tag's 97 / 99 / 101 range on 0 of 0 (mesh x reading) cells

## 300k: D164 seed 97 against B1 (the tag) 97 / 99 / 101

| mesh | arm | reading | tag 97 / 99 / 101 | D164 97 | inside the tag's range |
|---|---|---|---|---|---|
| bunny | L | sil_iou | 0.9884 / 0.9884 / 0.9884 | - |  |
| bunny | L | chamfer | 0.0551 / 0.05522 / 0.05507 | - |  |
| bunny | L | thin_uncovered | 0.04061 / 0.03974 / 0.03899 | - |  |
| bunny | L | yard_sil | 0.0007577 / 0.0007742 / 0.0007041 | - |  |
| bunny | L | yard_pbr | 0.0004772 / 0.000471 / 0.0004742 | - |  |
| bunny | L | display_front | 0.009182 / 0.009789 / 0.009302 | - |  |
| bunny | L | v_mean | 0.001005 / 0.0009567 / 0.0009968 | - |  |
| bunny | L | kin | 2.086e-06 / 1.658e-06 / 2.192e-06 | - |  |
| bunny | L | pieces10 | 0 / 0 / 0 | - |  |
| bunny | L | ang_last20 | 0.00377 / 0.00275 / 0.00267 | - |  |
| bunny | L | committed | 95 / 76 / 87 | - |  |
| bunny | L | judged | 102 / 80 / 93 | - |  |
| bunny | L | commit_replay | 4 / 2 / 3 | - |  |
| bunny | L | cf_tag_nulls | 0 / 0 / 0 | - |  |
| bunny | L | minutes | 11.14 / 8.843 / 10.26 | - |  |
| bunny | L | g_share | 0.3333 / 0.3333 / 0.3333 | - |  |
| bunny | L | lam_last | 0.001715 / 0.002378 / 0.001848 | - |  |
| bunny | P | sil_iou | 0.9871 / 0.987 / 0.9883 | - |  |
| bunny | P | chamfer | 0.05516 / 0.05527 / 0.05514 | - |  |
| bunny | P | thin_uncovered | 0.0385 / 0.04047 / 0.03173 | - |  |
| bunny | P | yard_sil | 0.001217 / 0.001331 / 0.0013 | - |  |
| bunny | P | yard_pbr | 0.0006283 / 0.0006504 / 0.0006459 | - |  |
| bunny | P | display_front | 0.01174 / 0.01264 / 0.01227 | - |  |
| bunny | P | v_mean | 0.001065 / 0.001246 / 0.001197 | - |  |
| bunny | P | kin | 2.265e-06 / 2.417e-06 / 2.859e-06 | - |  |
| bunny | P | pieces10 | 0 / 0 / 0 | - |  |
| bunny | P | ang_last20 | 0.00237 / 0.00296 / 0.0027 | - |  |
| bunny | P | committed | 106 / 81 / 89 | - |  |
| bunny | P | judged | 112 / 86 / 95 | - |  |
| bunny | P | commit_replay | 4 / 2 / 4 | - |  |
| bunny | P | cf_tag_nulls | 0 / 0 / 0 | - |  |
| bunny | P | minutes | 12.31 / 9.541 / 10.5 | - |  |
| bunny | P | g_share | 0 / 0 / 0 | - |  |
| bunny | P | lam_last | 0 / 0 / 0 | - |  |
| dragon | L | sil_iou | 0.9875 / 0.9891 / 0.9879 | - |  |
| dragon | L | chamfer | 0.05454 / 0.05463 / 0.05448 | - |  |
| dragon | L | thin_uncovered | 0.0326 / 0.03435 / 0.03007 | - |  |
| dragon | L | yard_sil | 0.0005281 / 0.0005795 / 0.0005386 | - |  |
| dragon | L | yard_pbr | 0.0004545 / 0.000464 / 0.0004553 | - |  |
| dragon | L | display_front | 0.008813 / 0.009906 / 0.009015 | - |  |
| dragon | L | v_mean | 0.002276 / 0.002259 / 0.002128 | - |  |
| dragon | L | kin | 1.074e-05 / 1.322e-05 / 8.648e-06 | - |  |
| dragon | L | pieces10 | 0 / 0 / 0 | - |  |
| dragon | L | ang_last20 | 0.00206 / 0.00307 / 0.00248 | - |  |
| dragon | L | committed | 164 / 165 / 186 | - |  |
| dragon | L | judged | 176 / 179 / 195 | - |  |
| dragon | L | commit_replay | 9 / 12 / 6 | - |  |
| dragon | L | cf_tag_nulls | 0 / 0 / 0 | - |  |
| dragon | L | minutes | 26.6 / 27.31 / 28.74 | - |  |
| dragon | L | g_share | 0.3333 / 0.3333 / 0.3333 | - |  |
| dragon | L | lam_last | 0.001592 / 0.001321 / 0.001238 | - |  |
| dragon | P | sil_iou | 0.9878 / 0.9892 / 0.9883 | - |  |
| dragon | P | chamfer | 0.05466 / 0.05471 / 0.05453 | - |  |
| dragon | P | thin_uncovered | 0.02917 / 0.03183 / 0.03393 | - |  |
| dragon | P | yard_sil | 0.001302 / 0.001365 / 0.001074 | - |  |
| dragon | P | yard_pbr | 0.0007019 / 0.0007296 / 0.0006277 | - |  |
| dragon | P | display_front | 0.0126 / 0.01402 / 0.01189 | - |  |
| dragon | P | v_mean | 0.002749 / 0.003044 / 0.00211 | - |  |
| dragon | P | kin | 1.382e-05 / 1.735e-05 / 7.354e-06 | - |  |
| dragon | P | pieces10 | 0 / 0 / 0 | - |  |
| dragon | P | ang_last20 | 0.00238 / 0.0026 / 0.002 | - |  |
| dragon | P | committed | 143 / 156 / 178 | - |  |
| dragon | P | judged | 151 / 168 / 193 | - |  |
| dragon | P | commit_replay | 5 / 10 / 13 | - |  |
| dragon | P | cf_tag_nulls | 0 / 0 / 0 | - |  |
| dragon | P | minutes | 23.03 / 25.62 / 28.57 | - |  |
| dragon | P | g_share | 0 / 0 / 0 | - |  |
| dragon | P | lam_last | 0 / 0 / 0 | - |  |
