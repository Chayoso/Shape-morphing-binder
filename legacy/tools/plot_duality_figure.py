"""F→Gaussian duality figure for paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse, FancyBboxPatch
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'cm',
    'font.size': 7.5,
    'axes.titlesize': 9,
    'axes.titleweight': 'bold',
})

fig = plt.figure(figsize=(7.2, 3.0), facecolor='white')

ax = fig.add_axes([0.02, 0.02, 0.96, 0.88])
ax.set_xlim(-0.5, 16)
ax.set_ylim(-2.5, 4.5)
ax.set_aspect('equal')
ax.axis('off')

c_blue = '#2166ac'
c_red = '#c0392b'
c_purple = '#8e44ad'
c_gray = '#555'

# ===== Top label =====
ax.text(3.0, 4.2, r'MPM particle $\equiv$ 3D Gaussian primitive (1:1 duality)',
        ha='center', fontsize=7.5, color=c_blue,
        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=c_blue, lw=0.5, alpha=0.9))

# Duality dotted box
dbox = FancyBboxPatch((0.0, -0.3), 6.0, 3.0,
                       boxstyle='round,pad=0.15', fc='none', ec=c_blue, lw=0.6, ls=':', alpha=0.3)
ax.add_patch(dbox)

# ===== Section 1: Rest config (isotropic) =====
cx1 = 1.0

rng = np.random.default_rng(42)
for _ in range(14):
    dx, dy = rng.normal(0, 0.5, 2)
    r = 0.18 + rng.uniform(-0.02, 0.02)
    c = plt.Circle((cx1 + dx, 1.2 + dy), r, fill=True, fc='#c4ddf0', ec=c_blue,
                    lw=0.6, alpha=0.7, zorder=2)
    ax.add_patch(c)

ax.text(cx1, 3.2, r'$\mathbf{F} = \mathbf{I}$', ha='center', fontsize=9,
        color=c_blue, fontweight='bold')
ax.text(cx1, -1.1, r'$\boldsymbol{\Sigma}_0 = \sigma_0^2\,\mathbf{I}$',
        ha='center', fontsize=7, color=c_gray)
ax.text(cx1, -1.6, 'Isotropic', ha='center', fontsize=6.5, color=c_gray, style='italic')

# ===== Arrow 1: MPM deformation =====
ax.annotate('', xy=(3.8, 1.2), xytext=(2.5, 1.2),
            arrowprops=dict(arrowstyle='-|>', color=c_gray, lw=1.5, mutation_scale=12))
ax.text(3.05, 1.5, 'MPM', ha='center', fontsize=7, color=c_gray, fontweight='bold')
ax.text(3.15, 0.5, r'$\mathbf{F}=\mathbf{R}\mathbf{S}$', ha='center', fontsize=7, color='#888')

# ===== Section 2: Deformed (anisotropic) =====
cx2 = 5.0

stretches = [
    (0.50, 0.18, 25), (0.40, 0.20, -15), (0.55, 0.15, 40),
    (0.35, 0.22, -5), (0.45, 0.17, 50), (0.38, 0.20, -35),
    (0.48, 0.16, 15), (0.42, 0.19, -20), (0.52, 0.14, 30),
    (0.36, 0.21, 10), (0.44, 0.18, -40), (0.50, 0.15, 55),
    (0.46, 0.17, 5), (0.39, 0.21, -25),
]
for i, (sx, sy, angle) in enumerate(stretches):
    dx, dy = rng.normal(0, 0.5, 2)
    e = Ellipse((cx2 + dx, 1.2 + dy), sx, sy, angle=angle,
                fill=True, fc='#f5cdc1', ec=c_red, lw=0.6, alpha=0.7, zorder=2)
    ax.add_patch(e)

ax.text(cx2, 3.2, r'$\mathbf{F} \neq \mathbf{I}$', ha='center', fontsize=9,
        color=c_red, fontweight='bold')
ax.text(cx2, -1.1, r'$\boldsymbol{\Sigma} = \mathbf{S}\,\boldsymbol{\Sigma}_0\,\mathbf{S}^\top$',
        ha='center', fontsize=7, color=c_gray)
ax.text(cx2, -1.6, 'Anisotropic', ha='center', fontsize=6.5, color=c_gray, style='italic')

# ===== Arrow 2: 3DGS render =====
ax.annotate('', xy=(8.3, 1.2), xytext=(6.8, 1.2),
            arrowprops=dict(arrowstyle='-|>', color=c_gray, lw=1.5, mutation_scale=12))
ax.text(7.55, 1.9, '3DGS', ha='center', fontsize=7, color=c_gray, fontweight='bold')

# ===== Section 3: Rendered image =====
cx3 = 9.5
rect = FancyBboxPatch((cx3 - 0.9, 0.2), 1.8, 2.0,
                       boxstyle='round,pad=0.12', fc='#f5f5f5', ec='#888', lw=0.8)
ax.add_patch(rect)
ax.text(cx3, 1.35, r'$\alpha$, $d$', ha='center', fontsize=9, color=c_gray, fontweight='bold')
ax.text(cx3, 0.65, 'silhouette\n+ depth', ha='center', fontsize=5.5, color='#999', linespacing=1.2)

# ===== Feedback arrow: render gradient (lower) =====
ax.annotate('', xy=(5.0, -2.0), xytext=(11.0, -2.0),
            arrowprops=dict(arrowstyle='-|>', color=c_red, lw=1.3, mutation_scale=11))
ax.text(8.0, -2.5, r'$\partial \mathcal{L} / \partial \mathbf{F}$  (render gradient)',
        ha='center', fontsize=7.5, color=c_red, fontweight='bold')

# ===== Section 4: Gradient comparison table =====
cx4 = 13.8

ax.text(cx4, 2.8, 'Gradient flow comparison', ha='center', fontsize=8, fontweight='bold', color=c_gray)

# Table layout — evenly spaced columns, centered at y~1.2
col0 = cx4 - 2.0   # row labels
col1 = cx4          # stretch
col2 = cx4 + 2.0    # rotation
row_h = 2.2
row1 = 1.4
row2 = 0.6

# Headers
ax.text(col0, row_h, '', ha='center', fontsize=6.5)
ax.text(col1, row_h, r'Stretch $\nabla_{\mathbf{s}}\mathcal{L}$', ha='center', fontsize=6.5,
        color=c_gray, fontweight='bold')
ax.text(col2, row_h, r'Rotation $\nabla_{\mathbf{q}}\mathcal{L}$', ha='center', fontsize=6.5,
        color=c_gray, fontweight='bold')

# Lines
lx0, lx1 = cx4 - 2.8, cx4 + 2.8
ax.plot([lx0, lx1], [1.9, 1.9], color='#ccc', lw=0.5)
ax.plot([lx0, lx1], [1.0, 1.0], color='#ccc', lw=0.5)
ax.plot([lx0, lx1], [0.2, 0.2], color='#ccc', lw=0.5)

# Row 1: isotropic
ax.text(col0, row1, r'$\mathbf{F}=\mathbf{I}$', ha='center', fontsize=7.5, color=c_blue)
ax.text(col1, row1, r'$\neq \mathbf{0}$', ha='center', fontsize=8.5, color='#27ae60', fontweight='bold')
ax.text(col2, row1, r'$= \mathbf{0}$', ha='center', fontsize=8.5, color='#e74c3c', fontweight='bold')

# Row 2: anisotropic
ax.text(col0, row2, r'$\mathbf{F}\neq\mathbf{I}$', ha='center', fontsize=7.5, color=c_red)
ax.text(col1, row2, r'$\neq \mathbf{0}$', ha='center', fontsize=8.5, color='#27ae60', fontweight='bold')
ax.text(col2, row2, r'$\neq \mathbf{0}$', ha='center', fontsize=8.5, color='#27ae60', fontweight='bold')

# Key insight
ax.text(cx4, -0.2, r'$\mathbf{F}$-derived covariance enables', ha='center', fontsize=6.5, color=c_gray)
ax.text(cx4, -0.7, 'rotation gradient channel', ha='center', fontsize=7, color=c_red,
        fontweight='bold', style='italic')

out = 'output/morph_from_isosphere/isosphere_to_bob'
plt.savefig(f'{out}/fig_duality_v3.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(f'{out}/fig_duality_v3.pdf', bbox_inches='tight', facecolor='white')
print('Done')
