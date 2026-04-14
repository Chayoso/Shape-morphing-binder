"""Convergence acceleration figure — two graphs + renders."""
import numpy as np
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patheffects as pe
from PIL import Image

po = json.load(open('output/ablation_bob_ppc5/physics_only/losses.json'))
fi = json.load(open('output/ablation_bob_ppc5/F_injection_no_plasticity/losses.json'))

eps30 = np.arange(30)
phys_po = np.array([po[i]['loss_physics'] for i in range(30)])
phys_fi = np.array([fi[i]['loss_physics'] for i in range(30)])
alpha_po = np.array([po[i]['alpha_mse'] for i in range(30)])
alpha_fi = np.array([fi[i]['alpha_mse'] for i in range(30)])

def crop(img, pad=2):
    gray = img.mean(axis=2)
    rows = np.any(gray < 0.97, axis=1)
    cols = np.any(gray < 0.97, axis=0)
    if not rows.any():
        return img
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return img[max(0, rmin-pad):rmax+pad, max(0, cmin-pad):cmax+pad]

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'cm',
    'font.size': 7,
})

out = Path('figs')
c_po = '#7f8c8d'
c_fi = '#2980b9'

fig = plt.figure(figsize=(7.2, 4.5), facecolor='white')

# Top left: physics loss
ax_phys = fig.add_axes([0.06, 0.58, 0.28, 0.32])
ax_phys.set_facecolor('#fafafa')
ax_phys.plot(eps30, phys_po, color=c_po, ls='--', lw=1.3, label='Physics-only',
             path_effects=[pe.Stroke(linewidth=2.2, foreground='white'), pe.Normal()])
ax_phys.plot(eps30, phys_fi, color=c_fi, lw=1.8, label='Ours',
             path_effects=[pe.Stroke(linewidth=2.8, foreground='white'), pe.Normal()])
ax_phys.axvspan(10, 19, alpha=0.08, color=c_fi, zorder=0)
for ep in [10, 15, 19]:
    ax_phys.axvline(x=ep, color=c_fi, ls=':', lw=0.4, alpha=0.3)
ax_phys.annotate('40% faster',
                 xy=(15, phys_fi[15]), xytext=(22, 6000),
                 fontsize=6.5, color=c_fi, fontweight='bold',
                 arrowprops=dict(arrowstyle='-|>', color=c_fi, lw=0.7,
                                connectionstyle='arc3,rad=-0.2'),
                 bbox=dict(boxstyle='round,pad=0.12', fc='white', ec=c_fi, lw=0.3, alpha=0.9))
ax_phys.set_xlabel('Frame')
ax_phys.set_ylabel(r'$\mathcal{L}_{\mathrm{phys}}$ ($\downarrow$)')
ax_phys.set_title('(a) Physics convergence', fontsize=7.5, fontweight='bold', pad=8)
ax_phys.set_xlim(0, 29)
ax_phys.legend(loc='upper right', framealpha=0.95, edgecolor='none', fontsize=5.5)
ax_phys.grid(True, alpha=0.12)
ax_phys.spines['top'].set_visible(False)
ax_phys.spines['right'].set_visible(False)

# Bottom left: alpha MSE
ax_alpha = fig.add_axes([0.06, 0.10, 0.28, 0.32])
ax_alpha.set_facecolor('#fafafa')
ax_alpha.plot(eps30, alpha_po, color=c_po, ls='--', lw=1.3, label='Physics-only',
              path_effects=[pe.Stroke(linewidth=2.2, foreground='white'), pe.Normal()])
ax_alpha.plot(eps30, alpha_fi, color=c_fi, lw=1.8, label='Ours',
              path_effects=[pe.Stroke(linewidth=2.8, foreground='white'), pe.Normal()])
ax_alpha.axvspan(10, 19, alpha=0.08, color=c_fi, zorder=0)
for ep in [10, 15, 19]:
    ax_alpha.axvline(x=ep, color=c_fi, ls=':', lw=0.4, alpha=0.3)
# Best markers
best_fi_ep = np.argmin(alpha_fi)
ax_alpha.scatter([best_fi_ep], [alpha_fi[best_fi_ep]], color='white', s=20, zorder=6,
                  edgecolors=c_fi, linewidths=1.0)
ax_alpha.annotate(f'8% lower',
                  xy=(best_fi_ep, alpha_fi[best_fi_ep]),
                  xytext=(5, 0.05),
                  fontsize=6, color=c_fi, fontweight='bold',
                  arrowprops=dict(arrowstyle='-|>', color=c_fi, lw=0.7,
                                 connectionstyle='arc3,rad=0.3'),
                  bbox=dict(boxstyle='round,pad=0.12', fc='white', ec=c_fi, lw=0.3, alpha=0.9))
ax_alpha.set_xlabel('Frame')
ax_alpha.set_ylabel(r'Alpha MSE ($\downarrow$)')
ax_alpha.set_title('(b) Silhouette error', fontsize=7.5, fontweight='bold', pad=8)
ax_alpha.set_xlim(0, 29)
ax_alpha.set_ylim(0, 0.36)
ax_alpha.legend(loc='upper right', framealpha=0.95, edgecolor='none', fontsize=5.5)
ax_alpha.grid(True, alpha=0.12)
ax_alpha.spines['top'].set_visible(False)
ax_alpha.spines['right'].set_visible(False)

# Right: 2x3 render grid
eps_show = [10, 15, 19]
methods = [
    ('physics_only', 'Physics-only', c_po),
    ('F_injection_no_plasticity', 'Ours', c_fi),
]

render_left = 0.40
render_w = 0.19
render_h = 0.33
render_hgap = 0.01
render_vgap = 0.03

for row, (method, label, c) in enumerate(methods):
    y = 0.47 - row * (render_h + render_vgap)
    for col, ep in enumerate(eps_show):
        x = render_left + col * (render_w + render_hgap)
        ax = fig.add_axes([x, y, render_w, render_h])
        img_raw = mpimg.imread(str(out / f'conv_{method}_ep{ep:03d}.png'))
        img = crop(img_raw)
        pil = Image.fromarray((img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8))
        pil = pil.resize((240, 160), Image.LANCZOS)
        img = np.array(pil).astype(np.float32) / 255
        ax.imshow(img)
        ax.axis('off')

        if row == 0:
            ax.set_title(f'Frame {ep}', fontsize=7, pad=6,
                          fontweight='bold', color='#2c3e50')

    fig.text(render_left - 0.015, y + render_h / 2, label,
             ha='right', va='center', fontsize=6.5, rotation=90,
             color=c, fontweight='bold', style='italic')

fig.text(0.70, 0.87,
         r'Isosphere $\rightarrow$ Duck (Bob)   $\cdot$   32$^3$ grid   $\cdot$   113K particles',
         ha='center', fontsize=8, fontweight='bold', color='#2c3e50')
fig.text(0.70, 0.835, '(c) Visual comparison at highlighted frames',
         ha='center', fontsize=6.5, color='#666')

plt.savefig(str(out / 'fig_convergence_comparison.png'), dpi=300,
            bbox_inches='tight', facecolor='white')
plt.savefig(str(out / 'fig_convergence_comparison.pdf'),
            bbox_inches='tight', facecolor='white')
print('Done')
