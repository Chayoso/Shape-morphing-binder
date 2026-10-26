#!/bin/bash
# Chapter 4, revision 4: three composites with Times New Roman labels (ImageMagick + Pango). No Python.
set -e
W=/data/relcfd/chayo/physmorph_v2/tmp/ch4figs; cd $W
G=/data/relcfd/chayo/physmorph_v2/output/gpu
F="Times New Roman"
lab() { # lab <out> <width> <height> <pango markup> [bg] [fg]
  convert -background "${5:-white}" -fill "${6:-black}" -size ${2}x${3} -gravity center pango:"<span font=\"$F, 30\">$4</span>" -extent ${2}x${3} "$1"; }
vlab() { # vertical label: vlab <out> <long side> <short side> <markup>
  convert -background white -fill black -size ${2}x${3} -gravity center pango:"<span font=\"$F, 30\">$4</span>" -extent ${2}x${3} -rotate -90 "$1"; }

# ---- (a) image-only relief, D163 300k plaque (closeup2 stills, 1440x810 each) ----
C=$G/d163/video/p300_H_T/closeup2
for v in front_left front_right back_left back_right; do
  for a in render_target twin full ceiling; do convert $C/${a}_$v.png -resize 720x405 a_${a}_$v.png; done
  convert a_render_target_$v.png a_twin_$v.png a_full_$v.png a_ceiling_$v.png -background white -splice 6x0 +append -chop 6x0 row_$v.png
done
lab h1.png 720 70 "Target pictures (relief mesh)"; lab h2.png 720 70 "δF<sub>c</sub> + layer <i>u</i> (λ = 0 twin)"
lab h3.png 720 70 "Full method, render weight held"; lab h4.png 720 70 "Lettered mesh sampled at 300k"
convert h1.png h2.png h3.png h4.png -background white -splice 6x0 +append -chop 6x0 head_a.png
convert row_front_left.png row_front_right.png -background white -splice 0x6 -append -chop 0x6 grid_front.png
convert row_back_left.png row_back_right.png -background white -splice 0x6 -append -chop 0x6 grid_back.png
H=$(identify -format %h grid_front.png)
vlab sf.png $H 64 "SIGGRAPH (front)"; vlab sb.png $H 64 "VANDERBILT (back)"
convert sf.png grid_front.png +append part_front.png; convert sb.png grid_back.png +append part_back.png
WH=$(identify -format %w head_a.png); convert -size 64x70 xc:white head_pad.png; convert head_pad.png head_a.png +append head_a2.png
convert head_a2.png part_front.png part_back.png -background white -splice 0x10 -append -chop 0x10 -quality 92 fig_render_letters.jpg

# ---- (b) appearance placement, D160d bunny turn, 40k (end.png 2880x1080 with baked labels painted over) ----
E=$G/d160/video/bunny_turn_97_D/end.png
convert $E \
  \( $E -crop 1x45+900+0 +repage -scale 345x45! \) -geometry +0+0 -composite \
  \( $E -crop 1x45+1860+0 +repage -scale 350x45! \) -geometry +960+0 -composite \
  \( $E -crop 1x45+2820+0 +repage -scale 170x45! \) -geometry +1920+0 -composite \
  \( $E -crop 1x50+700+1030 +repage -scale 260x50! \) -geometry +0+1030 -composite colour_clean0.png
# the studio camera draws a mirror image: flop each 960-px column in place (same handedness as Figs. 4.2 / 4.7)
convert colour_clean0.png -crop 960x1080 +repage -flop +append colour_clean.png
lab c1.png 960 70 "Full method, colour weight held"; lab c2.png 960 70 "δF<sub>c</sub> + layer <i>u</i> (λ = 0 twin)"
lab c3.png 960 70 "Target paint"
convert c1.png c2.png c3.png +append head_b.png
convert head_b.png colour_clean.png -append -quality 92 fig_render_colour.jpg

# ---- (c) where the render acts: RGV panel A, rule vs held (bunny 300k, seed 97) ----
R=$G/rgv_held/compare/tag_vs_held_1.2s_5.0s.png
for t in 0 1; do oy=$((110 + t*1150));
  convert $R -crop 520x390+220+$oy +repage rule_$t.png; convert $R -crop 520x390+2140+$oy +repage held_$t.png
  convert rule_$t.png held_$t.png -background white -splice 6x0 +append -chop 6x0 rgvrow_$t.png; done
lab r1.png 520 70 "λ by the rule"; lab r2.png 520 70 "Render weight held"
convert r1.png r2.png -background white -splice 6x0 +append -chop 6x0 head_c.png
vlab t0.png 390 64 "<i>t</i> = 1.2 s"; vlab t1.png 390 64 "<i>t</i> = 5.0 s"
convert t0.png rgvrow_0.png +append rr0.png; convert t1.png rgvrow_1.png +append rr1.png
convert -size 64x70 xc:white pad_c.png; convert pad_c.png head_c.png +append head_c2.png
convert head_c2.png rr0.png rr1.png -background white -splice 0x6 -append -chop 0x6 -quality 92 fig_render_held_share.jpg
identify fig_render_letters.jpg fig_render_colour.jpg fig_render_held_share.jpg

# ---- (d) the stamp test (D165): 500k raking-light close-ups (rake84 sheet, labels cropped) + 300k print depth ----
S=stamp500_rake84.jpg; D=imprint300.png   # copies of output/results_2026-10-23/d165_stamp/{500k/s500/read/rake84/closeup_sheet.jpg,imprint_depth.png}
convert $S -crop 3840x455+0+85 +repage -resize 2880x st_r0.png
convert $S -crop 3840x455+0+625 +repage -resize 2880x st_r1.png
SH=$(identify -format %h st_r0.png)
vlab sv0.png $SH 64 "Full method"; vlab sv1.png $SH 64 "Twin + decal"
convert sv0.png st_r0.png +append st_a.png; convert sv1.png st_r1.png +append st_b.png
lab sh1.png 1440 70 "Plaque face, settled (raking light)"; lab sh2.png 1440 70 "Print in the soft slab, mirrored (500k)"
convert -size 64x70 xc:white sp.png; convert sp.png sh1.png sh2.png +append st_h.png
convert $D -crop 1108x206+234+49 +repage -resize 1420x st_d0.png
convert $D -crop 1108x206+234+334 +repage -resize 1420x st_d1.png
lab sd1.png 1420 70 "Full method: depth of the print (300k)"; lab sd2.png 1420 70 "Twin + decal: depth of the print (300k)"
convert sd1.png st_d0.png -append st_c0.png; convert sd2.png st_d1.png -append st_c1.png
convert st_c0.png st_c1.png -background white -splice 40x0 +append -chop 40x0 st_c.png
convert -size 64x10 xc:white sp2.png; convert sp2.png st_c.png -gravity north +append st_c2.png
convert st_h.png st_a.png st_b.png -background white -splice 0x6 -append -chop 0x6 st_top.png
convert st_top.png st_c2.png -background white -gravity northwest -splice 0x24 -append -chop 0x24 -quality 92 fig_render_stamp.jpg
identify fig_render_stamp.jpg
