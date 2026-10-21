"""sc_sheet.py OUT_JPG TITLE FRAMES_DIR K[,K...] [COLS=4] [TILE_W=960] -- SHOWCASE: a contact sheet (CPU only) of a
render's frames (FRAMES_DIR/NNNN.jpg, NNNN the kept frame's index; a kept frame is 1/20 s), each tile labelled with its
time, the title above."""
import sys

from PIL import Image, ImageDraw, ImageFont

out, title, fdir, ks = sys.argv[1], sys.argv[2], sys.argv[3], [int(v) for v in sys.argv[4].split(",")]
cols = int(sys.argv[5]) if len(sys.argv) > 5 else 4
tw = int(sys.argv[6]) if len(sys.argv) > 6 else 960
th = tw * 9 // 16
rows = (len(ks) + cols - 1) // cols
head = 64
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 34)
    small = ImageFont.truetype("DejaVuSans.ttf", 30)
except OSError:
    font = small = ImageFont.load_default()
sheet = Image.new("RGB", (cols * tw, head + rows * th), (24, 26, 30))
dr = ImageDraw.Draw(sheet)
dr.text((16, 14), title, fill=(235, 235, 235), font=font)
for i, k in enumerate(ks):
    im = Image.open(f"{fdir}/{k:04d}.jpg").convert("RGB").resize((tw, th), Image.LANCZOS)
    x0, y0 = (i % cols) * tw, head + (i // cols) * th
    sheet.paste(im, (x0, y0))
    dr.text((x0 + 14, y0 + 10), f"t = {k / 20:.2f} s", fill=(255, 255, 255), font=small)
sheet.save(out, quality=92)
print("wrote", out)
