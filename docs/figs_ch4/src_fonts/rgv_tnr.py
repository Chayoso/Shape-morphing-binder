"""rgv_tnr.py SCRIPT ARGS... -- FIGS_V2 (2026-10-22): runs an RGV renderer script (render_gradient_viz*.py) with its PIL
text in Times New Roman: PIL's ImageFont.truetype is redirected from DejaVuSans*.ttf to the user's Times New Roman TTFs
(tmp/fonts/). Nothing else changes."""
import runpy
import sys

sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/fonts")
import tnr  # noqa: E402
from PIL import ImageFont  # noqa: E402

_truetype = ImageFont.truetype


def truetype(font=None, size=10, *a, **k):
    if isinstance(font, str) and "DejaVuSans" in font:
        font = tnr.path(bold="Bold" in font)
    return _truetype(font, size, *a, **k)


ImageFont.truetype = truetype
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
