"""Generate polished SIGGRAPH-style 1x3 loss figure for paper."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.collections import LineCollection

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'cm',
    'font.size': 8,
    'axes.labelsize': 8.5,
    'axes.titlesize': 9.5,
    'axes.titleweight': 'bold',
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'axes.linewidth': 0.5,
    'grid.linewidth': 0.25,
    'lines.linewidth': 1.8,
    'figure.dpi': 300,
})

losses = json.load(open('output/morph_from_isosphere/isosphere_to_bob/losses.json'))[:60]
eps = np.arange(len(losses))

physics = np.array([l['loss_physics'] for l in losses])
alpha = np.array([l['alpha_mse'] for l in losses])
j_min = np.array([l['J_min'] for l in losses])
j_max = np.array([l['J_max'] for l in losses])

best_ep = np.argmin(alpha)
best_val = alpha[best_ep]

# Palette
c_alpha   = '#d62728'
c_phys    = '#1f77b4'
c_jmin    = '#2ca02c'
c_jmax    = '#ff7f0e'
c_chamfer = '#9467bd'
bg_alpha  = '#fdedec'
bg_phys   = '#eaf2f8'
bg_jmin   = '#eafaf1'
bg_jmax   = '#fef5e7'

fig = plt.figure(figsize=(7.2, 2.4), facecolor='white')
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.55,
                       left=0.08, right=0.96, bottom=0.20, top=0.85)

# ===================== (a) Silhouette error =====================
ax = fig.add_subplot(gs[0])
ax.set_facecolor('#fafafa')

# Gradient fill
for i in range(len(eps)-1):
    ax.fill_between(eps[i:i+2], alpha[i:i+2], alpha=0.06 + 0.08*(1 - i/len(eps)),
                    color=c_alpha, linewidth=0)
ax.plot(eps, alpha, color=c_alpha, zorder=3,
        path_effects=[pe.Stroke(linewidth=2.8, foreground='white'), pe.Normal()])

# Chamfer region
ax.axvspan(20, 59, alpha=0.04, color=c_chamfer, zorder=0)
ax.axvline(x=20, color=c_chamfer, ls='--', lw=0.8, alpha=0.5)
ax.annotate('Chamfer plasticity\nactivated', xy=(20, 0.18), xytext=(38, 0.28),
            fontsize=6, color=c_chamfer, ha='center', style='italic',
            arrowprops=dict(arrowstyle='-|>', color=c_chamfer, lw=0.8,
                           connectionstyle='arc3,rad=-0.2'))

# Best point
ax.scatter([best_ep], [best_val], color='white', s=40, zorder=6, edgecolors=c_alpha, linewidths=1.5)
ax.annotate(f'{best_val:.3f}', xy=(best_ep, best_val),
            xytext=(best_ep-12, best_val+0.06), fontsize=7, color=c_alpha,
            fontweight='bold',
            arrowprops=dict(arrowstyle='-|>', color=c_alpha, lw=0.8),
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=c_alpha, lw=0.5, alpha=0.9))

ax.set_xlabel('Frame')
ax.set_ylabel(r'Alpha MSE ($\downarrow$)')
ax.set_title('(a)  Silhouette error')
ax.set_xlim(0, 59); ax.set_ylim(0, 0.40)
ax.grid(True, alpha=0.12, color='#ccc')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# ===================== (b) Physics convergence =====================
ax = fig.add_subplot(gs[1])
ax.set_facecolor('#fafafa')

for i in range(len(eps)-1):
    ax.fill_between(eps[i:i+2], physics[i:i+2], alpha=0.05 + 0.08*(1 - i/len(eps)),
                    color=c_phys, linewidth=0)
ax.plot(eps, physics, color=c_phys, zorder=3,
        path_effects=[pe.Stroke(linewidth=2.8, foreground='white'), pe.Normal()])

reduction = (1 - physics[-1] / physics[0]) * 100
ax.annotate(f'{reduction:.0f}% reduction',
            xy=(52, physics[-1]), xytext=(25, 15000),
            fontsize=7, color=c_phys, fontweight='bold',
            arrowprops=dict(arrowstyle='-|>', color=c_phys, lw=0.8,
                           connectionstyle='arc3,rad=0.2'),
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=c_phys, lw=0.5, alpha=0.9))

ax.set_xlabel('Frame')
ax.set_ylabel(r'Physics loss ($\downarrow$)')
ax.set_title('(b)  Physics convergence')
ax.set_xlim(0, 59)
ax.grid(True, alpha=0.12, color='#ccc')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# ===================== (c) Deformation range =====================
ax = fig.add_subplot(gs[2])
ax.set_facecolor('#fafafa')

ax.fill_between(eps, j_min, 1.0, alpha=0.10, color=c_jmin, linewidth=0)
ax.fill_between(eps, 1.0, j_max, alpha=0.10, color=c_jmax, linewidth=0)
ax.plot(eps, j_min, color=c_jmin, label=r'$J_{\min}$', zorder=3,
        path_effects=[pe.Stroke(linewidth=2.8, foreground='white'), pe.Normal()])
ax.plot(eps, j_max, color=c_jmax, label=r'$J_{\max}$', zorder=3,
        path_effects=[pe.Stroke(linewidth=2.8, foreground='white'), pe.Normal()])
ax.axhline(y=1.0, color='#bbb', ls='-', lw=0.5, zorder=1)

# Annotate spread
ax.annotate('', xy=(55, j_min[-1]), xytext=(55, j_max[-1]),
            arrowprops=dict(arrowstyle='<->', color='#666', lw=0.8))
spread = j_max[-1] - j_min[-1]
ax.text(56.5, (j_min[-1]+j_max[-1])/2, f'{spread:.2f}',
        fontsize=6, color='#555', ha='left', va='center')

ax.set_xlabel('Frame')
ax.set_ylabel(r'$\det(\mathbf{F})$')
ax.set_title('(c)  Deformation range')
ax.set_xlim(0, 59)
leg = ax.legend(loc='center left', framealpha=0.95, edgecolor='#ddd',
                fancybox=True, shadow=False)
leg.get_frame().set_linewidth(0.5)
ax.grid(True, alpha=0.12, color='#ccc')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

out = 'output/morph_from_isosphere/isosphere_to_bob'
plt.savefig(f'{out}/loss_figure_fancy.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(f'{out}/loss_figure_fancy.pdf', bbox_inches='tight', facecolor='white')
print('Done')
