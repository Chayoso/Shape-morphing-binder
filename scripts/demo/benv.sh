# benv.sh -- the demo renderer's environment on hyde06: caches, Blender user / config / temp dirs under /data, $BL.
D=/data/relcfd/chayo/physmorph_v2/tmp/demo
source /data/relcfd/chayo/physmorph_v2/tmp/d146/cache_env.sh
export BLENDER_USER_RESOURCES=$D/blender_cfg BLENDER_USER_CONFIG=$D/blender_cfg/config BLENDER_USER_SCRIPTS=$D/blender_cfg/scripts BLENDER_USER_DATAFILES=$D/blender_cfg/datafiles BLENDER_USER_EXTENSIONS=$D/blender_cfg/extensions TMPDIR=$D/blender_tmp TEMP=$D/blender_tmp
BL=/data/relcfd/chayo/physmorph_v2/tools/blender/blender-4.2.3-linux-x64/blender
