"""Pipeline figure — polished SIGGRAPH style."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'cm',
    'font.size': 6,
})

def crop_ws(img, pad=6, thresh=0.97):
    gray = img.mean(axis=2) if img.ndim == 3 else img
    rows = np.any(gray < thresh, axis=1)
    cols = np.any(gray < thresh, axis=0)
    if not rows.any() or not cols.any():
        return img
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return img[max(0, rmin-pad):min(img.shape[0], rmax+pad),
               max(0, cmin-pad):min(img.shape[1], cmax+pad)]

base = Path('output/morph_from_isosphere/isosphere_to_bob/morphing_sideR')
imgs = {}
for ep, key in [(0, 'src'), (20, 'mid'), (59, 'final')]:
    imgs[key] = crop_ws(mpimg.imread(str(base / f'frame_{ep:03d}.png')))

C = {
    'phys': '#1a5276', 'phys_light': '#d6eaf8',
    'render': '#922b21', 'render_light': '#f9e0dc',
    'plast': '#196f3d', 'plast_light': '#d4efdf',
    'dark': '#2c3e50', 'gray': '#666666', 'lgray': '#aaaaaa',
    'arrow': '#bdc3c7',
}

fig = plt.figure(figsize=(7.2, 4.8), facecolor='white')
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 68)
ax.axis('off')

# ================================================================
# TOP: Morphing progression
# ================================================================
top_y = 56

# Images
for cx, key, label in [(12, 'src', 'Frame 0'), (40, 'mid', 'Frame 20'), (68, 'final', 'Frame 59')]:
    ax_img = fig.add_axes([(cx-7)/100, (top_y-8)/68, 14/100, 10/68])
    ax_img.imshow(imgs[key])
    ax_img.axis('off')
    # Shadow effect
    shadow = FancyBboxPatch((-0.02, -0.02), 1.04, 1.04, transform=ax_img.transAxes,
                             boxstyle='round,pad=0.02', fc='none', ec='#ccc', lw=0.8)
    ax_img.add_patch(shadow)
    ax.text(cx, top_y - 9.5, label, ha='center', fontsize=6, color=C['gray'], fontweight='bold')

# Curved arrows
for x1, x2 in [(19, 33), (47, 61)]:
    ax.annotate('', xy=(x2, top_y - 3), xytext=(x1, top_y - 3),
                arrowprops=dict(arrowstyle='-|>', color=C['arrow'], lw=1.8,
                               mutation_scale=12, connectionstyle='arc3,rad=-0.15'))

# ================================================================
# MIDDLE: Three stages
# ================================================================
mid_y = 20
bh = 24
bw = 28
gap = 3.5

for i, (x0, col, bg, num, title, lines) in enumerate([
    (2, C['phys'], C['phys_light'], '1', 'Physics Rollout', [
        (r'MPM forward ($T$ steps)', False),
        (r'Optimize $\delta\mathbf{F}_c$', False),
        (None, False),
        (r'Inject $\partial\mathcal{L}/\partial\mathbf{F}$', True),
        (None, False),
        (r'Output: $\mathbf{x}_T,\mathbf{F}_T$', False),
    ]),
    (2 + bw + gap, C['render'], C['render_light'], '2', 'Multi-view Render', [
        (r'$\mathbf{F} \to \mathbf{R},\mathbf{S}$  (polar)', False),
        (r'$\boldsymbol{\Sigma}=\mathbf{S}\boldsymbol{\Sigma}_0\mathbf{S}^\top$', False),
        (r'3DGS  ($8$ views)', False),
        (None, False),
        (r'DT + IoU + Depth', False),
        (r'$\to \partial\mathcal{L}/\partial\mathbf{F}$', True),
    ]),
    (2 + 2*(bw + gap), C['plast'], C['plast_light'], '3', 'Chamfer Plasticity', [
        (r'NN disp. $\mathbf{d}_i$ to target', False),
        (r'KNN diffusion', False),
        (None, False),
        (r'$\delta\mathbf{F}_p$ from Jacobian', False),
        (r'$\mathbf{F}_p\!\leftarrow\!(\mathbf{I}+\eta\delta\mathbf{F}_p)\mathbf{F}_p$', True),
        (r'Rest $\to$ target', False),
    ]),
]):
    # Gradient background
    box = FancyBboxPatch((x0, mid_y), bw, bh,
                          boxstyle='round,pad=0.6', fc=bg, ec=col, lw=1.0, alpha=0.5)
    ax.add_patch(box)

    # Top accent bar
    bar = FancyBboxPatch((x0 + 0.3, mid_y + bh - 4.5), bw - 0.6, 4.5,
                          boxstyle='round,pad=0.3', fc=col, ec='none', alpha=0.12)
    ax.add_patch(bar)

    # Number badge
    badge = plt.Circle((x0 + 3, mid_y + bh - 2.5), 1.8, fc=col, ec='white', lw=1.2, zorder=5)
    ax.add_patch(badge)
    ax.text(x0 + 3, mid_y + bh - 2.5, num, ha='center', va='center',
            fontsize=8, color='white', fontweight='bold', zorder=6)

    # Title
    ax.text(x0 + bw/2 + 1.5, mid_y + bh - 2.5, title, ha='center', va='center',
            fontsize=7.5, color=col, fontweight='bold',
            path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    # Content
    cy = mid_y + bh - 7
    for text, bold in lines:
        if text is None:
            cy -= 1.2
            continue
        color = col if bold else C['gray']
        weight = 'bold' if bold else 'normal'
        ax.text(x0 + bw/2, cy, text, ha='center', va='center',
                fontsize=5.8, color=color, fontweight=weight)
        cy -= 2.5

    # Arrow to next stage
    if i < 2:
        ax_x = x0 + bw + gap/2
        ax.annotate('', xy=(ax_x + 1, mid_y + bh/2), xytext=(ax_x - 1, mid_y + bh/2),
                    arrowprops=dict(arrowstyle='-|>', color=C['arrow'], lw=2.5,
                                   mutation_scale=16))

# ================================================================
# FEEDBACK ARROW
# ================================================================
s1_cx = 2 + bw/2
s2_cx = 2 + bw + gap + bw/2
ax.annotate('',
            xy=(s1_cx, mid_y - 1.5), xytext=(s2_cx, mid_y - 1.5),
            arrowprops=dict(arrowstyle='-|>', color=C['render'], lw=1.5,
                           mutation_scale=12, connectionstyle='arc3,rad=0.25'))
ax.text((s1_cx + s2_cx)/2, mid_y - 5,
        r'$\partial\mathcal{L}/\partial\mathbf{F}$ feedback (next episode)',
        ha='center', fontsize=6, color=C['render'], style='italic',
        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=C['render'], lw=0.4, alpha=0.8))

# ================================================================
# BOTTOM: Key principles with icons
# ================================================================
bot_y = 3
ax.plot([3, 97], [bot_y + 9, bot_y + 9], color='#eee', lw=0.8)

principles = [
    (r'$\mathbf{x}$: physics only', C['phys'], 15),
    (r'$\boldsymbol{\Sigma}$: shaped by $\mathbf{F}$', C['render'], 50),
    (r'$\mathbf{F}_p$: Chamfer plasticity', C['plast'], 85),
]
for text, color, cx in principles:
    dot = plt.Circle((cx - 8, bot_y + 5), 0.8, fc=color, ec='none', alpha=0.8)
    ax.add_patch(dot)
    ax.text(cx - 6, bot_y + 5, text, va='center', fontsize=6, color=C['dark'])

out = 'output/morph_from_isosphere/isosphere_to_bob'
plt.savefig(f'{out}/fig_pipeline_v4.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(f'{out}/fig_pipeline_v4.pdf', bbox_inches='tight', facecolor='white')
print('Done')
