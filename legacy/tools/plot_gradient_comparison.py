"""Gradient flow comparison figure: isotropic vs anisotropic."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'cm',
    'font.size': 8,
    'axes.labelsize': 8.5,
    'axes.titlesize': 9,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'axes.linewidth': 0.6,
    'lines.linewidth': 1.6,
})

# Data from compare_gradients.py (Spot)
eps = [0, 1, 3, 5, 10, 15, 20, 30, 40, 59]
iso_scale =  [0.011639, 0.011622, 0.011623, 0.010485, 0.010254, 0.011471, 0.012580, 0.013933, 0.013190, 0.015011]
iso_rot =    [0.0,      0.0,      0.0,      0.0,      0.0,      0.0,      0.0,      0.0,      0.0,      0.0]
ani_scale =  [0.011639, 0.011623, 0.011623, 0.010489, 0.010260, 0.011478, 0.012607, 0.013965, 0.013233, 0.015036]
ani_rot =    [0.000001, 0.000003, 0.000006, 0.000009, 0.000014, 0.000023, 0.000029, 0.000028, 0.000024, 0.000031]

fig = plt.figure(figsize=(7.2, 2.4), facecolor='white')
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.35,
                       left=0.08, right=0.96, bottom=0.20, top=0.85)

c_scale = '#2166ac'
c_rot_aniso = '#c0392b'
c_rot_iso = '#cccccc'

# (a) Scale gradient: both similar
ax1 = fig.add_subplot(gs[0])
ax1.plot(eps, iso_scale, 'o-', color=c_scale, markersize=4, label='Isotropic', alpha=0.5, ls='--')
ax1.plot(eps, ani_scale, 's-', color=c_scale, markersize=4, label='Anisotropic (ours)')
ax1.set_xlabel('Frame')
ax1.set_ylabel(r'$\sum \|\nabla_{\mathbf{s}} L\|$')
ax1.set_title(r'(a) Scale gradient $\nabla_{\mathbf{s}} L$')
ax1.legend(framealpha=0.9, edgecolor='none')
ax1.grid(True, alpha=0.15)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# (b) Rotation gradient: zero vs nonzero
ax2 = fig.add_subplot(gs[1])

# Bar chart: iso = 0 (gray), aniso = nonzero (red)
x_pos = np.arange(len(eps))
width = 0.35
bars_iso = ax2.bar(x_pos - width/2, iso_rot, width, color=c_rot_iso, label='Isotropic (= 0)', edgecolor='#999', linewidth=0.5)
bars_ani = ax2.bar(x_pos + width/2, ani_rot, width, color=c_rot_aniso, label='Anisotropic (ours)', edgecolor=c_rot_aniso, linewidth=0.5, alpha=0.8)

ax2.set_xlabel('Frame')
ax2.set_ylabel(r'$\sum \|\nabla_{\mathbf{q}} L\|$')
ax2.set_title(r'(b) Rotation gradient $\nabla_{\mathbf{q}} L$')
ax2.set_xticks(x_pos)
ax2.set_xticklabels([str(e) for e in eps])
ax2.legend(framealpha=0.9, edgecolor='none')
ax2.grid(True, alpha=0.15, axis='y')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Annotation
ax2.annotate('Isotropic:\nrotation gradient\nis exactly zero',
             xy=(0, 0), xytext=(3, 0.000025),
             fontsize=6.5, color='#666', style='italic',
             arrowprops=dict(arrowstyle='->', color='#999', lw=0.7))

out = 'output/morph_from_isosphere/isosphere_to_bob'
plt.savefig(f'{out}/fig_gradient_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(f'{out}/fig_gradient_comparison.pdf', bbox_inches='tight', facecolor='white')
print('Done')
