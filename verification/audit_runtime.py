"""
Runtime proof that the evaluation is leak-free.

Code reading can miss a path that only triggers at runtime, so this instruments
the real train_model and asserts, on actual DEAP data, that:
  1. the early-stopping validation batches are a subset of the training rows and
     share no row with the test fold;
  2. group integrity holds (no trial/subject straddles sub-train and validation);
  3. predictions are returned for the test fold, not the validation split;
  4. the scaler is fit without ever seeing test rows.
"""
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
EV  = _os.path.join(_REPO, 'evaluation')
OUT = _os.path.join(_REPO, 'evaluation', 'outputs')
TEX = _os.path.join(_REPO, 'paper', 'main.tex')
import sys, os
sys.path.insert(0, EV)
import numpy as np
import torch
import run_evaluation_unified as R

fails = []
def ck(name, ok, detail=''):
    print(('  OK   ' if ok else '  FAIL ') + name + (f'   {detail}' if detail else ''))
    if not ok: fails.append(name)

# ── Load one real subject ---------------------------------------------------
data = R.load_deap_4class()
sid = sorted(data)[0]
X, Y, trials = data[sid]['X'], data[sid]['Y'], data[sid]['trials']
print(f'Subject s{sid}: X={X.shape}, {len(np.unique(trials))} trials\n')

from sklearn.model_selection import StratifiedGroupKFold
sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=R.SEED)
tr_idx, te_idx = next(iter(sgkf.split(X, Y, groups=trials)))

# ── 1. Trial-aware split integrity (Tier 2 definition) ---------------------
print('TIER 2 SPLIT INTEGRITY')
ck('no trial spans train and test',
   not (set(trials[tr_idx]) & set(trials[te_idx])),
   f'{len(set(trials[tr_idx]))} train trials, {len(set(trials[te_idx]))} test trials')
ck('no row index shared', not (set(tr_idx.tolist()) & set(te_idx.tolist())))

# ── 2. Normalization fit on train only -------------------------------------
print('\nNORMALIZATION')
Xtr_n, Xte_n = R.normalize_features(X[tr_idx], X[te_idx], 32, X.shape[2])
# Refit using ONLY train rows and confirm the test transform is identical,
# i.e. no test statistics leaked into the scaler.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(X[tr_idx].reshape(-1, X.shape[2]))
manual = sc.transform(X[te_idx].reshape(-1, X.shape[2])).reshape(-1, 32, X.shape[2])
ck('test transform uses train-only statistics',
   np.allclose(Xte_n, manual.astype(np.float32), atol=1e-5))

# ── 3. Instrument train_model to capture what it actually trains/validates --
print('\nEARLY STOPPING DATA PROVENANCE (instrumented live run)')
seen = {'val': [], 'train': [], 'test': []}
orig_ds = R.EEGDataset
class SpyDataset(orig_ds):
    def __init__(self, Xa, Ya):
        super().__init__(Xa, Ya)
        seen.setdefault('_all', []).append(np.asarray(Xa))
R.EEGDataset = SpyDataset

hp = dict(R.DEFAULT_HPARAMS); hp['epochs'] = 3; hp['patience'] = 2
model = R.create_gat(32, X.shape[2], hp, n_classes=4)
preds, epochs_ran = R.train_model(model, Xtr_n, Y[tr_idx], Xte_n, Y[te_idx],
                                  hp, R.SEED, groups_tr=trials[tr_idx])
R.EEGDataset = orig_ds

# train_model constructs val_dl FIRST, then train_dl, then test_dl.
arrs = seen['_all']
ck('three datasets constructed (val, sub-train, test)', len(arrs) == 3, f'got {len(arrs)}')
val, sub_tr, test = arrs[0], arrs[1], arrs[2]

def rowset(A):
    return set(map(lambda r: hash(r.tobytes()), A.astype(np.float32)))

s_sub, s_val, s_test = rowset(sub_tr), rowset(val), rowset(test)
ck('validation shares NO row with test fold', not (s_val & s_test),
   f'|val|={len(s_val)} |test|={len(s_test)} overlap={len(s_val & s_test)}')
ck('sub-train shares NO row with test fold', not (s_sub & s_test),
   f'overlap={len(s_sub & s_test)}')
ck('validation disjoint from sub-train', not (s_sub & s_val),
   f'overlap={len(s_sub & s_val)}')
ck('sub-train + val reconstruct the training set',
   len(s_sub | s_val) == len(rowset(Xtr_n)),
   f'{len(s_sub | s_val)} vs {len(rowset(Xtr_n))}')
ck('test dataset equals the held-out fold', s_test == rowset(Xte_n))
ck('val fraction near 15%',
   0.10 <= len(val)/ (len(sub_tr)+len(val)) <= 0.20,
   f'{len(val)/(len(sub_tr)+len(val)):.3f}')

# ── 4. Predictions correspond to the TEST fold -----------------------------
print('\nPREDICTION TARGET')
ck('one prediction per test row', len(preds) == len(te_idx),
   f'{len(preds)} preds vs {len(te_idx)} test rows')
ck('prediction count != validation size (not scoring val)',
   len(preds) != len(val))

# ── 5. Group integrity of the validation split -----------------------------
print('\nVALIDATION SPLIT GROUP INTEGRITY')
sub_i, val_i = R.make_val_split(Y[tr_idx], trials[tr_idx], R.SEED)
ck('val split returned', sub_i is not None)
if sub_i is not None:
    g_sub, g_val = set(trials[tr_idx][sub_i]), set(trials[tr_idx][val_i])
    ck('no trial straddles sub-train and validation', not (g_sub & g_val),
       f'{len(g_sub)} vs {len(g_val)} trials, overlap={len(g_sub & g_val)}')
    ck('all classes present in sub-train',
       len(np.unique(Y[tr_idx][sub_i])) == len(np.unique(Y[tr_idx])))

# LOSO-style: groups are subjects
print('\nLOSO VALIDATION SPLIT (subject-grouped)')
subj = np.repeat(np.arange(31), 300)
Yl = np.random.RandomState(0).randint(0, 4, len(subj))
s2, v2 = R.make_val_split(Yl, subj, 42)
ck('LOSO val split returned', s2 is not None)
if s2 is not None:
    ck('no subject straddles sub-train and validation',
       not (set(subj[s2]) & set(subj[v2])),
       f'{len(set(subj[s2]))} train subj, {len(set(subj[v2]))} val subj')

print('\n' + '=' * 78)
print(f'{"ALL RUNTIME LEAK CHECKS PASSED" if not fails else "FAILURES: " + ", ".join(fails)}')
print('=' * 78)
sys.exit(1 if fails else 0)
