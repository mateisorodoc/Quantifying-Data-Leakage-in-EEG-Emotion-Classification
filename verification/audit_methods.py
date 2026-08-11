"""
Audit the paper's METHODOLOGICAL claims against what the code actually does.
Numbers can be right while the description of how they were produced is wrong,
so each claim here is checked against the source, not against results.
"""
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
EV  = _os.path.join(_REPO, 'evaluation')
OUT = _os.path.join(_REPO, 'evaluation', 'outputs')
TEX = _os.path.join(_REPO, 'paper', 'main.tex')
import re, os, sys, inspect
import numpy as np


src = open(os.path.join(EV, 'run_evaluation_unified.py'), encoding='utf-8').read()
feat = open(os.path.join(EV, 'feature_extraction.py'), encoding='utf-8').read()
tex = open(TEX, encoding='utf-8').read()

res = []
def claim(name, ok, detail=''):
    res.append((ok, name, detail))

# 1. "test fold is used only once, for the reported predictions"
#    -> predictions must come from test_dl, and val_dl must come from training data
tm = src[src.find('def train_model'):src.find('def create_gat')]
claim('early stopping uses a split of TRAINING data',
      'make_val_split' in tm and 'X_tr[val_i]' in tm,
      'val_dl built from X_tr[val_i]')
claim('final predictions come from the TEST loader',
      re.search(r'for Xb, _ in test_dl', tm) is not None)
claim('test set never feeds early stopping',
      'EEGDataset(X_te' not in tm.split('test_dl =')[0],
      'no X_te dataset before test_dl definition')

# 2. "patience=20"
claim('patience default is 20', "'patience', 20" in tm or "hparams.get('patience', 20)" in tm)

# 3. "15% validation split"
claim('VAL_FRACTION is 0.15', 'VAL_FRACTION = 0.15' in src)

# 4. "grouped by trial at Tier 2, by subject at Tier 3"
claim('T2 passes trial groups to val split',
      'groups_tr=trials[tr_idx]' in src)
claim('T3 passes subject groups to val split',
      'G_tr = all_subs_arr[train_mask]' in src and 'groups_tr=G_tr' in src)

# 5. "Z-score normalization fit on training data only"
nf = src[src.find('def normalize_features'):src.find('def train_model')]
claim('scaler fit on train only',
      'scaler.fit_transform(X_tr_flat)' in nf and 'scaler.transform(X_te_flat)' in nf)

# 6. Tier definitions
claim('T0 is a random 80/20 split of pooled subjects',
      'train_test_split(' in src and 'test_size=0.2' in src)
claim('T1 uses StratifiedKFold without groups',
      re.search(r'skf = StratifiedKFold\(n_splits=N_FOLDS.*\n.*sub_scores', src) is not None
      or 'skf.split(X, Y)' in src)
claim('T2 uses StratifiedGroupKFold grouped by trial',
      'sgkf.split(X, Y, groups=trials)' in src)
claim('T3 trains on N-1 subjects (LOSO)',
      'test_mask = all_subs_arr == int(test_sub)' in src and 'train_mask = ~test_mask' in src)
claim('N_FOLDS is 10', re.search(r'^N_FOLDS = 10', src, re.M) is not None)
claim('32 subjects configured',
      re.search(r'SUBJECTS = \[f.\{i:02d\}. for i in range\(1, 33\)\]', src) is not None)

# 7. "class-weighted cross-entropy", "AdamW", "cosine annealing"
claim('AdamW optimizer', 'optim.AdamW' in tm)
claim('cosine annealing scheduler', 'CosineAnnealingLR' in tm)
claim('class-weighted cross-entropy', 'CrossEntropyLoss(weight=weights' in tm)
claim('class weights from TRAINING labels only', 'np.bincount(Y_tr' in tm)

# 8. Feature description: 26 per channel, 5 bands, 2 s @ 50% overlap
claim('5 frequency bands defined', len(re.findall(r'\(\s*\d+\s*,\s*\d+\s*\)', feat[feat.find('BANDS'):feat.find('BANDS')+200])) >= 5
      or 'N_BANDS' in feat)
claim('DEAP window 256 samples (2 s @128 Hz)', 'DEAP_WINDOW = 256' in feat)
claim('DEAP step 128 (50% overlap)', 'DEAP_STEP = 128' in feat)
claim('OpenBCI window 250 (2 s @125 Hz)', 'OPENBCI_WINDOW = 250' in feat)
claim('OpenBCI step 125 (50% overlap)', 'OPENBCI_STEP = 125' in feat)

# 9. "two architectures" only
claim('MODELS is exactly (MLP, GAT)', "MODELS = ('MLP', 'GAT')" in src)
claim('LightGAT excluded from MODELS', "'LightGAT'" not in src.split('MODELS = ')[1].split('\n')[0])

# 10. binary Tier3 skipped
claim('RUN_BINARY_T3 is False', 'RUN_BINARY_T3 = False' in src)

# 11. statistics: paired t-test on subject vectors, Bonferroni
tc = src[src.find('def tier_comparison_stats'):src.find('# ═══', src.find('def tier_comparison_stats'))]
claim('paired t-test used', 'ttest_rel' in tc)
claim("Cohen's d_z uses sample SD (ddof=1)", "diff.std(ddof=1)" in tc)
claim('Bonferroni correction applied', 'p_raw * n_comparisons' in tc)
claim('CI uses t critical value', 'scipy_stats.t.ppf(0.975' in tc)
claim('sample unit recorded as subject', "'unit': 'subject'" in tc)

# 12. reproducibility
claim('global seed 42', re.search(r'^SEED = 42', src, re.M) is not None)
claim('cudnn deterministic', 'cudnn.deterministic = True' in src)
claim('per-model seeding in train_model', 'torch.manual_seed(seed)' in tm)

# 13. Paper text consistency with code constants
claim('paper says patience=20', 'patience=20' in tex)
claim('paper says 15% validation split', "15\\%" in tex and 'validation split' in tex)
claim('paper says 10-fold', '10-fold' in tex)
claim('paper says 31 training subjects at T3', 'training on 31 subjects' in tex)
claim('paper says 2 s windows / 50% overlap', "2\\,s" in tex and "50\\%" in tex)
claim('paper says 26 features per channel', '26 features per channel' in tex)
claim('paper reports 832 / 416 dims',
      '832 for DEAP' in tex and '416 for OpenBCI' in tex)
# 32 ch x 26 = 832 ; 16 x 26 = 416
claim('832 = 32 x 26 arithmetic', 32*26 == 832)
claim('416 = 16 x 26 arithmetic', 16*26 == 416)

# 14. Claims the paper makes about what it does NOT do
claim('paper does not claim EEGNet was run',
      'We do not evaluate raw-signal models' in tex)
claim('code indeed has no EEGNet', 'EEGNet' not in src and 'eegnet' not in src.lower())

print('=' * 78)
print('METHODOLOGY AUDIT: paper claims vs. source code')
print('=' * 78)
bad = 0
for ok, name, detail in res:
    if not ok:
        print(f'  FAIL  {name}  {detail}')
        bad += 1
print(f'\n{len(res)-bad}/{len(res)} methodological claims verified against code')
print('=' * 78)
sys.exit(1 if bad else 0)
