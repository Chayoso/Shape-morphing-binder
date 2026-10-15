#!/bin/bash
# B12: the first prep read B5's oriented point clouds as meshes (one face): remove those data folders, redo on the meshes.
B=/data/relcfd/chayo/physmorph_v2; R=$B/baselines/b12
echo "$(date '+%Y-%m-%d %H:%M:%S %Z') prep: the 11:36 data were made from B5's point-cloud PLYs (read as 3-vertex meshes): removed, redone on the *_ourframe meshes" >> $R/status.txt
rm -rf $B/baselines/Implicit-surf-Deformation/data/b12_*
bash $B/tmp/d145/b12/b12_prep.sh
bash $B/tmp/d145/b12/b12_radial.sh
