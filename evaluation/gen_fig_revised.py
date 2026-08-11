"""
Regenerate the paper's tier figure for the revised narrative.

Left  : DEAP 4-class macro F1 across the four tiers, all three architectures --
        the protocol effect, and the fact that the architectures track each other.
Right : the OpenBCI session-confound controls. The previous version of this
        figure showed OpenBCI holding above 0.98 under trial-aware evaluation,
        framed as a success; the controls show why that number is not evidence
        of emotion decoding.

Usage: python gen_fig_revised.py
Writes conference_paper/fig_tier_comparison.png
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'outputs', 'all_results.json')
CONTROLS = os.path.join(HERE, 'outputs', 'openbci_session_control.json')
OUT = os.path.join(HERE, '..', 'conference_paper', 'fig_tier_comparison.png')

TIERS = ['Tier0', 'Tier1', 'Tier2', 'Tier3']
TIER_LBL = ['T0\nrandom', 'T1\nleaky', 'T2\ntrial-aware', 'T3\nLOSO']
COLORS = {'MLP': '#4C72B0', 'GAT': '#C44E52', 'LightGAT': '#55A868'}

R = json.load(open(RESULTS))
C = json.load(open(CONTROLS))

# Take the architecture list from the results rather than hardcoding it, so the
# figure matches whatever the sweep actually evaluated.
MODELS = list(R['DEAP']['Tier0'].keys())

plt.rcParams.update({'font.size': 8, 'axes.grid': True,
                     'grid.alpha': 0.3, 'grid.linewidth': 0.5})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.15))

# ── Left: DEAP tier degradation ──────────────────────────────────────────────
x = np.arange(len(TIERS))
for m in MODELS:
    means = [R['DEAP'][t][m]['f1_mean'] for t in TIERS]
    stds = [R['DEAP'][t][m].get('f1_std', 0) for t in TIERS]
    ax1.errorbar(x, means, yerr=stds, marker='o', ms=4, capsize=2.5,
                 lw=1.4, color=COLORS[m], label=m)

ax1.axhline(0.25, ls='--', lw=1, color='grey')
ax1.text(3.05, 0.265, 'chance', fontsize=6.5, color='grey', ha='right')
ax1.set_xticks(x)
ax1.set_xticklabels(TIER_LBL)
ax1.set_ylabel('Macro F1')
ax1.set_title(f"DEAP 4-class (n={R.get('_meta', {}).get('deap_subjects', 32)} subjects)",
              fontsize=9)
ax1.set_ylim(0, 0.85)
ax1.legend(fontsize=6.5, loc='upper right', framealpha=0.9)

# annotate the dominant transition
t1 = R['DEAP']['Tier1']['GAT']['f1_mean']
t2 = R['DEAP']['Tier2']['GAT']['f1_mean']
ax1.annotate('', xy=(2.0, t2 + 0.01), xytext=(2.0, t1 - 0.01),
             arrowprops=dict(arrowstyle='<->', color='black', lw=0.9))
ax1.text(2.12, (t1 + t2) / 2, f'trial leakage\n{t2 - t1:+.3f}',
         fontsize=6.5, ha='left', va='center')

# ── Right: OpenBCI session-confound controls (all binary, chance = 0.50) ─────
pa = C['probe_A_recording_order']
pairs = {e['pair']: e for e in C['probe_C_pairwise']['pairs']}


def pair_f1(name):
    for k, e in pairs.items():
        if set(k.split(' vs ')) == set(name):
            return e['MLP']['mean'], e['gap_days']
    raise KeyError(name)


hc, hc_gap = pair_f1(('happy', 'calm'))
cs, cs_gap = pair_f1(('calm', 'sad'))

labels = ['order within\nsession',
          'sham labels\n(control)',
          f'happy vs calm\n({hc_gap:.0f} d apart)',
          f'calm vs sad\n({cs_gap:.0f} d apart)']
vals = [pa['MLP_across_sessions']['mean'], C['probe_B_sham_labels']['mean'], hc, cs]
errs = [pa['MLP_across_sessions']['std'], C['probe_B_sham_labels']['std'], 0, 0]
cols = ['#C44E52', '#999999', '#4C72B0', '#4C72B0']

xb = np.arange(len(vals))
ax2.bar(xb, vals, yerr=errs, capsize=3, color=cols, width=0.62)
ax2.axhline(0.5, ls='--', lw=1, color='grey')
ax2.text(3.45, 0.515, 'chance', fontsize=6.5, color='grey', ha='right')
ax2.set_xticks(xb)
ax2.set_xticklabels(labels, fontsize=6.3)
ax2.set_ylabel('Macro F1')
ax2.set_ylim(0, 1.08)
ax2.set_title('OpenBCI controls (binary, chance = 0.50)', fontsize=9)
for i, v in enumerate(vals):
    ax2.text(i, v + 0.03, f'{v:.3f}', ha='center', fontsize=6.5)

plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches='tight')
print(f'Saved {os.path.normpath(OUT)}')
print(f'  left  : DEAP tiers, {len(MODELS)} models')
print(f'  right : controls  order={vals[0]:.3f}  sham={vals[1]:.3f}  '
      f'happy/calm={vals[2]:.3f}  calm/sad={vals[3]:.3f}')
