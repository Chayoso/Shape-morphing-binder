"""Ablation comparison figure — PPC5 results."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'cm',
    'font.size': 8,
    'axes.labelsize': 8.5,
    'axes.titlesize': 9,
    'axes.titleweight': 'bold',
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'axes.linewidth': 0.6,
    'lines.linewidth': 1.6,
})

po = json.load(open('output/ablation_bob_ppc5/physics_only/losses.json'))
fi = json.load(open('output/ablation_bob_ppc5/F_injection_no_plasticity/losses.json'))
# fm = json.load(open('output/ablation_bob_ppc5/full_method/losses.json'))

eps = np.arange(60)
alpha_po = np.array([l['alpha_mse'] for l in po])
alpha_fi = np.array([l['alpha_mse'] for l in fi])
# alpha_fm = np.array([l['alpha_mse'] for l in fm])
phys_po = np.array([l['loss_physics'] for l in po])
phys_fi = np.array([l['loss_physics'] for l in fi])
# phys_fm = np.array([l['loss_physics'] for l in fm])

c_po = '#7f8c8d'
c_fi = '#2980b9'
c_fm = '#c0392b'

fig = plt.figure(figsize=(7.2, 2.6), facecolor='white')
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.35,
                       left=0.08, right=0.96, bottom=0.18, top=0.85)

# (a) Alpha MSE comparison
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor('#fafafa')

ax1.plot(eps, alpha_po, color=c_po, ls='--', lw=1.2, label='Physics-only',
         path_effects=[pe.Stroke(linewidth=2.4, foreground='white'), pe.Normal()])
ax1.plot(eps, alpha_fi, color=c_fi, lw=2.0, label='+ Control-space inj. (ours)',
         path_effects=[pe.Stroke(linewidth=3.0, foreground='white'), pe.Normal()])

# Best markers
for alphas, col, name in [(alpha_po, c_po, 'po'), (alpha_fi, c_fi, 'fi')]:
    best_ep = np.argmin(alphas)
    best_val = alphas[best_ep]
    ax1.scatter([best_ep], [best_val], color='white', s=25, zorder=6,
                edgecolors=col, linewidths=1.2)


ax1.set_xlabel('Frame')
ax1.set_ylabel(r'Alpha MSE ($\downarrow$)')
ax1.set_title('(a) Silhouette error')
ax1.set_xlim(0, 30)
ax1.set_ylim(0, 0.36)
ax1.legend(loc='upper right', framealpha=0.95, edgecolor='none', fontsize=6.5)
ax1.grid(True, alpha=0.12)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# (b) Physics loss comparison
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor('#fafafa')

ax2.plot(eps, phys_po, color=c_po, ls='--', lw=1.2, label='Physics-only',
         path_effects=[pe.Stroke(linewidth=2.4, foreground='white'), pe.Normal()])
ax2.plot(eps, phys_fi, color=c_fi, lw=2.0, label='+ Control-space inj. (ours)',
         path_effects=[pe.Stroke(linewidth=3.0, foreground='white'), pe.Normal()])

ax2.set_xlabel('Frame')
ax2.set_ylabel(r'Physics loss ($\downarrow$)')
ax2.set_title('(b) Physics convergence')
ax2.set_xlim(0, 30)
ax2.legend(loc='upper right', framealpha=0.95, edgecolor='none', fontsize=6.5)
ax2.grid(True, alpha=0.12)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Summary annotation
best_po = alpha_po.min()
best_fi = alpha_fi.min()
# best_fm removed
improvement = (best_po - best_fi) / best_po * 100
ax1.annotate(f'{improvement:.0f}% lower',
             xy=(17, best_fi), xytext=(8, 0.05),
             fontsize=7.5, color=c_fi, fontweight='bold',
             arrowprops=dict(arrowstyle='-|>', color=c_fi, lw=0.8,
                            connectionstyle='arc3,rad=0.2'),
             bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=c_fi, lw=0.5, alpha=0.95))

out = 'output/morph_from_isosphere/isosphere_to_bob'
plt.savefig(f'{out}/fig_ablation.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(f'{out}/fig_ablation.pdf', bbox_inches='tight', facecolor='white')
print('Done')
