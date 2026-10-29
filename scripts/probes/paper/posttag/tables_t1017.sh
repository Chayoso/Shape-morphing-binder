#!/bin/bash
# tables_t1017.sh -- D145 post-tag reading 2: the markdown tables (CPU only) from read_t1017.sh's outputs into
# output/gpu/d145t_evidence/r2/tables_r2.md (B2, B8, B9) beside the 40k / B10 / B1-lead tables read_t1017.sh wrote.
B=/data/relcfd/chayo/physmorph_v2; R=$B/tmp/d145r; O=$B/output/gpu/d145t_evidence/r2
source $B/repo_t/scripts/ops/hyde06_env.sh > /dev/null 2>&1
export CUDA_VISIBLE_DEVICES=
M="$PY $R/md145m.py"
J() { for a in "$@"; do echo -n " $O/tab_all_${a}.json"; done; }
END="yard_ext_sil:1e4:yard sil e-4,yard_ext_pbr:1e4:yard shading e-4,disp_front_1miou:100:display front %,sil_ind::sil IoU ind,chamfer_ind::chamfer ind,thin:100:thin %,kin_last:1e6:end kin e-6"
RUN="apart_w3_10_max::apart w3-10,vis_run_side_blob::side blobs run,pc_end_largest::largest end piece,com_max_fine_a:1e5:CoM e-5 a,rotation_deg:1e3:rotation e-3 deg,kin_peak::kin peak,windows::windows,minutes::minutes"
MAT="detF_min::det F min,rec_Jx_min::J min,rec_Jx_max::J max,kin_peak::kin peak,share_c_median::g_share,lam_c_first::lambda w1"
GRAV="rec_com_y_end::CoM height change wu,rec_com_y_min::lowest wu,kin_last:1e6:end kin e-6,windows::windows,minutes::minutes,share_c_median::g_share"
S=_s101_97_99
B2T="X_vs_L$S U_vs_L$S R_vs_L$S VO_vs_L$S VJ_vs_L$S VD_vs_L$S VA_vs_L$S E_vs_L$S M_vs_L$S C_vs_L$S"
B2P="OX_vs_OL$S OU_vs_OL$S OR_vs_OL$S OVO_vs_OL$S OC_vs_OL$S"
B8T="ESL_vs_L$S EHL_vs_L$S NUL_vs_L$S ESL_vs_ESP_s97 EHL_vs_EHP_s97 NUL_vs_NUP_s97"
B8P="OESL_vs_OL$S OEHL_vs_OL$S OESL_vs_OESP_s97 OEHL_vs_OEHP_s97 ONUL_vs_ONUP_s97"
B9T="G2L_vs_L$S G2L_vs_G2P_s97 GFL_vs_G2L_s97 D0L_vs_L$S"
B9P="OG2L_vs_OL$S OG2L_vs_OG2P_s97 G2L_vs_OG2L$S"
{
echo "# D145 post-tag reading 2: B2, B8, B9 (300k; read $(date '+%Y-%m-%d %H:%M %Z'))"; echo
echo "Cells: median A / median B (change of A against B; for an IoU the change of 1 - IoU; * = the seed ranges do not overlap)."
echo "L = the tag's render arm (B1), OL = the pre-tag render arm (D135 code); O-prefixed arms are pre-tag runs."; echo
$M "B2 on the tag, each arm against L: the end" "$END" $(J $B2T)
$M "B2 on the tag: the run" "$RUN" $(J $B2T)
$M "B2 before the tag (D135 code), each arm against OL: the end" "$END" $(J $B2P)
$M "B2 before the tag: the run" "$RUN" $(J $B2P)
$M "B8 materials on the tag: the end" "$END" $(J $B8T)
$M "B8 on the tag: the run" "$RUN" $(J $B8T)
$M "B8 on the tag: the material's own measures" "$MAT" $(J $B8T)
$M "B8 before the tag: the end" "$END" $(J $B8P)
$M "B8 before the tag: the material's own measures" "$MAT" $(J $B8P)
$M "B9 external forces on the tag (floor runs read against the lifted independent sample): the end" "$END" $(J $B9T)
$M "B9 on the tag: the centre of mass, rest and cost" "$GRAV" $(J $B9T)
$M "B9 before the tag, and the tag's G2L against the pre-tag's: the end" "$END" $(J $B9P)
$M "B9 before the tag: the centre of mass, rest and cost" "$GRAV" $(J $B9P)
} > $O/tables_r2.md 2>&1
$PY $R/b9_40k.py $O/b9_40k.md > /dev/null
echo tables done
