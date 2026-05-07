"""Generate paper figures from all_results.json for ICSTCC 2026 paper."""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    'font.size': 9,
    'font.family': 'serif',
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
})

OUT_DIR = '../conference_paper'

with open('outputs/all_results.json', 'r') as f:
    data = json.load(f)

# --- Extract means ---
# 4-class DEAP
deap4 = {
    'MLP': [data['DEAP']['Tier0']['MLP']['f1_mean'],
            data['DEAP']['Tier1']['MLP']['f1_mean'],
            data['DEAP']['Tier2']['MLP']['f1_mean'],
            data['DEAP']['Tier3']['MLP']['f1_mean']],
    'GAT': [data['DEAP']['Tier0']['GAT']['f1_mean'],
            data['DEAP']['Tier1']['GAT']['f1_mean'],
            data['DEAP']['Tier2']['GAT']['f1_mean'],
            data['DEAP']['Tier3']['GAT']['f1_mean']],
}

# OpenBCI 4-class
openbci = {
    'MLP': [data['OpenBCI']['Tier0']['MLP']['f1_mean'],
            data['OpenBCI']['Tier1']['MLP']['f1_mean'],
            data['OpenBCI']['Tier2']['MLP']['f1_mean']],
    'GAT': [data['OpenBCI']['Tier0']['GAT']['f1_mean'],
            data['OpenBCI']['Tier1']['GAT']['f1_mean'],
            data['OpenBCI']['Tier2']['GAT']['f1_mean']],
}

# Binary DEAP
binary_aro = {
    'MLP': [data['DEAP_binary']['arousal']['Tier0']['MLP']['f1_mean'],
            data['DEAP_binary']['arousal']['Tier1']['MLP']['f1_mean'],
            data['DEAP_binary']['arousal']['Tier2']['MLP']['f1_mean'],
            data['DEAP_binary']['arousal']['Tier3']['MLP']['f1_mean']],
    'GAT': [data['DEAP_binary']['arousal']['Tier0']['GAT']['f1_mean'],
            data['DEAP_binary']['arousal']['Tier1']['GAT']['f1_mean'],
            data['DEAP_binary']['arousal']['Tier2']['GAT']['f1_mean'],
            data['DEAP_binary']['arousal']['Tier3']['GAT']['f1_mean']],
}
binary_val = {
    'MLP': [data['DEAP_binary']['valence']['Tier0']['MLP']['f1_mean'],
            data['DEAP_binary']['valence']['Tier1']['MLP']['f1_mean'],
            data['DEAP_binary']['valence']['Tier2']['MLP']['f1_mean'],
            data['DEAP_binary']['valence']['Tier3']['MLP']['f1_mean']],
    'GAT': [data['DEAP_binary']['valence']['Tier0']['GAT']['f1_mean'],
            data['DEAP_binary']['valence']['Tier1']['GAT']['f1_mean'],
            data['DEAP_binary']['valence']['Tier2']['GAT']['f1_mean'],
            data['DEAP_binary']['valence']['Tier3']['GAT']['f1_mean']],
}

tiers_4 = ['T0', 'T1', 'T2', 'T3']
tiers_3 = ['T0', 'T1', 'T2']

# ============================================================
# FIGURE 1: Progressive Degradation (GAT only, two panels)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.0))

# Left: DEAP 4-class GAT
colors_gat = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
bars1 = ax1.bar(tiers_4, deap4['GAT'], color=colors_gat, width=0.5, edgecolor='black', linewidth=0.5)
ax1.axhline(y=0.25, color='gray', linestyle='--', linewidth=1, label='Chance (0.25)')
ax1.set_ylabel('Macro F1')
ax1.set_xlabel('Evaluation Tier')
ax1.set_title('DEAP: GAT 4-Class Degradation')
ax1.set_ylim(0, 0.75)
ax1.legend(loc='upper right', framealpha=0.9)
for bar in bars1:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., h + 0.01, f'{h:.3f}',
             ha='center', va='bottom', fontsize=7.5)

# Right: OpenBCI 4-class GAT
colors_ob = ['#2196F3', '#4CAF50', '#FF9800']
bars2 = ax2.bar(tiers_3, openbci['GAT'], color=colors_ob, width=0.5, edgecolor='black', linewidth=0.5)
ax2.axhline(y=0.25, color='gray', linestyle='--', linewidth=1, label='Chance (0.25)')
ax2.set_ylabel('Macro F1')
ax2.set_xlabel('Evaluation Tier')
ax2.set_title('OpenBCI: GAT 4-Class (n=1)')
ax2.set_ylim(0, 1.1)
ax2.legend(loc='lower right', framealpha=0.9)
for bar in bars2:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., h + 0.01, f'{h:.3f}',
             ha='center', va='bottom', fontsize=7.5)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig_tier_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved fig_tier_comparison.png")

# ============================================================
# FIGURE 2: Model Comparison (MLP vs GAT, two panels)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.0))

x = np.arange(len(tiers_4))
w = 0.35

# Left: DEAP 4-class MLP vs GAT
bars_mlp = ax1.bar(x - w/2, deap4['MLP'], w, label='MLP', color='#1976D2', edgecolor='black', linewidth=0.5)
bars_gat = ax1.bar(x + w/2, deap4['GAT'], w, label='GAT', color='#FF8F00', edgecolor='black', linewidth=0.5)
ax1.axhline(y=0.25, color='gray', linestyle='--', linewidth=1, label='Chance')
ax1.set_xticks(x)
ax1.set_xticklabels(tiers_4)
ax1.set_ylabel('Macro F1')
ax1.set_xlabel('Evaluation Tier')
ax1.set_title('DEAP: 4-Class (32ch, 26 feat/ch)')
ax1.set_ylim(0, 0.85)
ax1.legend(loc='upper right', framealpha=0.9)

# Right: DEAP Binary (Arousal + Valence, MLP vs GAT)
# Show arousal and valence as separate groups
x2 = np.arange(len(tiers_4))
w2 = 0.18

bars_mlp_a = ax2.bar(x2 - 1.5*w2, binary_aro['MLP'], w2, label='MLP-Aro', color='#1976D2', edgecolor='black', linewidth=0.4)
bars_gat_a = ax2.bar(x2 - 0.5*w2, binary_aro['GAT'], w2, label='GAT-Aro', color='#FF8F00', edgecolor='black', linewidth=0.4)
bars_mlp_v = ax2.bar(x2 + 0.5*w2, binary_val['MLP'], w2, label='MLP-Val', color='#64B5F6', edgecolor='black', linewidth=0.4)
bars_gat_v = ax2.bar(x2 + 1.5*w2, binary_val['GAT'], w2, label='GAT-Val', color='#FFB74D', edgecolor='black', linewidth=0.4)
ax2.axhline(y=0.50, color='gray', linestyle='--', linewidth=1, label='Chance')
ax2.set_xticks(x2)
ax2.set_xticklabels(tiers_4)
ax2.set_ylabel('Macro F1')
ax2.set_xlabel('Evaluation Tier')
ax2.set_title('DEAP: Binary Arousal & Valence')
ax2.set_ylim(0, 0.95)
ax2.legend(loc='upper right', framealpha=0.9, fontsize=6.5, ncol=2)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved fig_model_comparison.png")

print("Done! Figures saved to conference_paper/")
