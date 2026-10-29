#!/bin/bash
# 34_figs_v2_rgv_tnr.sh -- gpu1 queue job (agent FIGS_V2): the dissertation's RGV figure (fig_render_gradient_early/late)
# re-rendered for its two kept frames with Times New Roman text and the body flipped back; see tmp/figs_v2/rgv_job.sh.
# Exit 3 = gpu1 lacked memory (needs 14000 MiB free): queued again as 99_figs_v2_rgv_tnr_rNN.sh.
exec bash /data/relcfd/chayo/physmorph_v2/tmp/figs_v2/rgv_job.sh
