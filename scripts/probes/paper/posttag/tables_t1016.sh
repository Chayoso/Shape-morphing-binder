#!/bin/bash
# tables_t1016.sh -- D145 post-tag: the markdown tables of the B1 / A6 / pre-tag readings (md145t.py on band145t.py's JSON;
# CPU only) into output/gpu/d145t_evidence/tables_t1016.md (compact) and tables_t1016_detail.md (min / median / max).
B=/data/relcfd/chayo/physmorph_v2; R=$B/tmp/d145r; E=$B/output/gpu/d145t_evidence
source $B/repo_t/scripts/ops/hyde06_env.sh > /dev/null 2>&1
export CUDA_VISIBLE_DEVICES=
M="$PY $R/md145t.py"
SHAPE="sil_ind::sil IoU ind,chamfer_ind::chamfer ind,yard_ext_sil:1e4:yard sil e-4,yard_ext_pbr:1e4:yard shading e-4,disp_front_1miou:100:display front 1-IoU %,disp_back_1miou:100:display back 1-IoU %,thin:100:thin %,kin_last:1e6:end kin e-6"
MOM="com_max_fine_a:1e5:CoM largest e-5 a,rotation_deg:1e3:net rotation e-3 deg,ang_net_gross_mean:1e3:ang net/gross e-3,Lstart_max::L max,Lgrid_sum::sum dL grid,rec_spacing_net_P::push net P"
VIS="vis_w3_10_side_blob::side blobs w3-10,vis_run_side_blob::side blobs run,vis_run_blob::blobs run,vis_end_side_blob::side blobs last window,pc_end_pieces::pieces end,pc_end_largest::largest piece end,pc_end_farthest_a::farthest gap end a,pc_run_farthest_a::farthest gap run a,pc_far_frames::frames with a piece >=5 a,pc_far_largest::largest piece >=5 a,apart_w3_10_max::apart w3-10 max"
COST="windows::windows,minutes::minutes,share_c_median::g_share median,lam_c_first::lambda first,gcos_w1_10::cos(g_phys g_render) w1-10,kin_peak::kin peak"
A6="yard_ext_sil:1e4:yard sil e-4,yard_ext_pbr:1e4:yard shading e-4,disp_front_1miou:100:display front %,sil_ind::sil IoU ind,thin:100:thin %,kin_last:1e6:end kin e-6,pulse_exc:100:tail excursion e-2 a,pulse_ret::tail returned share,rev_7_15::reversal 7-15 s,rev_15::reversal >15 s,move_last20_a:1e3:move last20 e-3 a,com_max_fine_a:1e5:CoM largest e-5 a,rec_spacing_net_P:1e5:push net e-5 P,vis_run_side_blob::side blobs run,vis_end_side_blob::side blobs last,windows::windows,minutes::minutes"
PRE="yard_ext_sil:1e4:yard sil e-4,yard_ext_pbr:1e4:yard shading e-4,disp_front_1miou:100:display front %,sil_ind::sil IoU ind,chamfer_ind::chamfer ind,thin:100:thin %,kin_last:1e6:end kin e-6,apart_w3_10_max::apart w3-10,com_max_fine_a:1e5:CoM largest e-5 a,rotation_deg:1e3:rotation e-3 deg,ang_net_gross_mean:1e3:ang net/gross e-3,windows::windows,minutes::minutes"
{
echo "# D145 post-tag tables (freeze-2026-10-15, fc02a1a; read $(date '+%Y-%m-%d %H:%M %Z'))"; echo
echo "Cells: median A / median B (change of A against B; for an IoU the change of 1 - IoU; * = the seed ranges do not overlap)."; echo
$M $E/tab_tx_L_vs_P.json "B1, render arm L vs its lambda = 0 twin P: shape at the end" "$SHAPE"
$M $E/tab_tx_L_vs_P.json "B1, L vs P: momentum and rest" "$MOM"
$M $E/tab_tx_L_vs_P.json "B1, L vs P: floating material (visible_apart, pieces145)" "$VIS"
$M $E/tab_tx_L_vs_P.json "B1, L vs P: cost and the render's share" "$COST"
for a in NRD NUC NSS; do $M $E/tab_tx_${a}_vs_L_s101_97_99.json "A6, $a vs the tag's L (seeds 97 / 99 / 101)" "$A6"; done
$M $E/tab_allx_L_vs_OL.json "Tag L vs pre-tag L (OL = D135, deliverable-2026-10-13)" "$PRE"
$M $E/tab_allx_P_vs_OP.json "Tag P vs pre-tag P (OP)" "$PRE"
$M $E/tab_allx_OL_vs_OP.json "Pre-tag L vs pre-tag P (the lead on the old code)" "$SHAPE"
} > $E/tables_t1016.md 2>&1
{
echo "# D145 post-tag tables, min / median / max per arm (freeze-2026-10-15; read $(date '+%Y-%m-%d %H:%M %Z'))"; echo
$M $E/tab_tx_L_vs_P.json "B1 L vs P" "$SHAPE,$MOM,$VIS,$COST" --detail
for a in NRD NUC NSS; do $M $E/tab_tx_${a}_vs_L_s101_97_99.json "A6 $a vs L" "$A6" --detail; done
$M $E/tab_allx_L_vs_OL.json "Tag L vs pre-tag L" "$PRE" --detail
$M $E/tab_allx_P_vs_OP.json "Tag P vs pre-tag P" "$PRE" --detail
} > $E/tables_t1016_detail.md 2>&1
echo tables done
