"""tnr.py -- Times New Roman for the dissertation's Chapter 4 figures (FIGS_V2, 2026-10-22; the user's request: every
figure text in Times New Roman, as the body text). The four TTFs in this folder are the user's own Windows copies
(times.ttf, timesbd.ttf, timesi.ttf, timesbi.ttf), copied by scp.
  import sys; sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/fonts"); import tnr
  tnr.mpl()                        # matplotlib: registered, the default family, mathtext in Times New Roman too
  tnr.pil(size, bold=False)        # a PIL ImageFont
  tnr.path(bold=False, italic=False)"""
import os

D = os.path.dirname(os.path.abspath(__file__))
FILES = {(False, False): "times.ttf", (True, False): "timesbd.ttf", (False, True): "timesi.ttf", (True, True): "timesbi.ttf"}


def path(bold=False, italic=False):
    return os.path.join(D, FILES[(bool(bold), bool(italic))])


def mpl():
    import matplotlib as m
    from matplotlib import font_manager as fm
    # only these files under the name (the server also has msttcorefonts' copy, which would otherwise win the tie)
    fm.fontManager.ttflist = [f for f in fm.fontManager.ttflist if f.name != "Times New Roman"]
    for f in FILES.values():
        fm.fontManager.addfont(os.path.join(D, f))
    m.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "custom",
                       "mathtext.rm": "Times New Roman", "mathtext.it": "Times New Roman:italic",
                       "mathtext.bf": "Times New Roman:bold", "mathtext.sf": "Times New Roman",
                       "mathtext.cal": "Times New Roman:italic", "mathtext.tt": "Times New Roman"})


def pil(size, bold=False, italic=False):
    from PIL import ImageFont
    return ImageFont.truetype(path(bold, italic), size)
