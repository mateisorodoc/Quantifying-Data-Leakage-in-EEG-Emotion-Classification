"""
═══════════════════════════════════════════════════════════════════════════════
DeepGAT: UNIFIED FOUR-TIER EVALUATION (MLP + GAT)
═══════════════════════════════════════════════════════════════════════════════
TIER 0: Random split (no awareness of trials/subjects) — leakage baseline
TIER 1: Within-subject CV (StratifiedKFold) — moderate leakage
TIER 2: Trial-aware CV (StratifiedGroupKFold by trial) — proper evaluation
TIER 3: Cross-subject LOSO (train N-1, test 1) — generalization test

Models: MLP (baseline) + DeepGAT (graph attention)
Classification: 4-class quadrant (HAHV, HALV, LAHV, LALV) for both datasets
Features: 26/channel (BP+DE+PLV+Coherence+PAC+Temporal)
Channels: 32 (DEAP), 16 (OpenBCI)
Normalization: Per-fold StandardScaler (fit on train only)
═══════════════════════════════════════════════════════════════════════════════
"""

import os, warnings, glob, time, json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from collections import Counter
from scipy import stats as scipy_stats

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    StratifiedGroupKFold, StratifiedKFold, LeaveOneGroupOut, train_test_split
)
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

# ─── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEAP_FEAT_DIR = os.path.join(PROJECT_ROOT, '..', 'data', 'DEAP', 'output', 'features_v6')
OPENBCI_BASE = os.path.join(PROJECT_ROOT, '..', 'data', 'recordings_clean')
OPENBCI_FEAT = os.path.join(OPENBCI_BASE, 'features_v5', 'openbci_all.npz')
OUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Constants ────────────────────────────────────────────────────────────────
DEAP_N_CH = 32
OPENBCI_N_CH = 16
N_CLASSES = 4
N_FOLDS = 10
SUBJECTS = [f'{i:02d}' for i in range(1, 21)]

# Class labels (same mapping for both datasets)
CLASS_NAMES = ['HAHV', 'HALV', 'LAHV', 'LALV']  # happy, stressed, calm, sad

# ─── Default Hyperparameters ──────────────────────────────────────────────────
DEFAULT_HPARAMS = {
    'backbone_dims': [64, 64, 64],
    'num_heads': 4,
    'attn_dropout': 0.05,
    'head_dense': 128,
    'head_dropout': 0.3,
    'lr': 5e-4,
    'weight_decay': 1e-4,
    'batch_size': 512,
    'epochs': 200,
    'patience': 20,
    'label_smoothing': 0.0,
    'input_noise_std': 0.0,
}


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=4,
                 attn_dropout=0.05, residual=True):
        super().__init__()
        self.H, self.d = num_heads, out_features
        self.residual = residual
        self.W = nn.Linear(in_features, num_heads * out_features, bias=False)
        self.a_src = nn.Parameter(torch.empty(num_heads, out_features))
        self.a_dst = nn.Parameter(torch.empty(num_heads, out_features))
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))
        self.leaky = nn.LeakyReLU(0.2)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.bn = nn.BatchNorm1d(out_features)
        if residual:
            self.res_proj = (nn.Linear(in_features, out_features, bias=False)
                            if in_features != out_features else nn.Identity())

    def forward(self, x, return_attn=False):
        B, N, _ = x.shape
        h = self.W(x).view(B, N, self.H, self.d)
        e_src = (h * self.a_src).sum(-1)
        e_dst = (h * self.a_dst).sum(-1)
        e = self.leaky(e_src.unsqueeze(2) + e_dst.unsqueeze(1))
        alpha = self.attn_drop(F.softmax(e, dim=2))
        out = torch.einsum('bqkh, bkhd -> bqhd', alpha, h).mean(dim=2)
        out = self.bn(out.reshape(B * N, self.d)).reshape(B, N, self.d)
        out = F.elu(out)
        if self.residual:
            out = out + self.res_proj(x)
        if return_attn:
            return out, alpha.permute(0, 3, 1, 2)
        return out


class DeepGAT(nn.Module):
    """Graph Attention Network for 4-class EEG emotion classification.
    
    Fully-connected graph: each EEG channel (node) attends to all others.
    The model learns salient connectivity patterns via attention weights.
    """
    def __init__(self, n_ch=32, in_feats=26, n_classes=4,
                 backbone_dims=None, dense=128, num_heads=4,
                 attn_dropout=0.05, head_dropout=0.3):
        super().__init__()
        if backbone_dims is None:
            backbone_dims = [64, 64, 64]
        d0 = backbone_dims[0]
        self.input_proj = nn.Linear(in_feats, d0)
        self.ch_embed = nn.Parameter(torch.randn(1, n_ch, d0) * 0.02)
        dims = [d0] + backbone_dims
        self.backbone = nn.ModuleList([
            GATLayer(dims[i], dims[i+1], num_heads, attn_dropout)
            for i in range(len(backbone_dims))
        ])
        self.classifier = nn.Sequential(
            nn.Linear(backbone_dims[-1], dense), nn.LayerNorm(dense),
            nn.GELU(), nn.Dropout(head_dropout), nn.Linear(dense, n_classes))

    def forward(self, x, return_attn=False):
        x = F.gelu(self.input_proj(x)) + self.ch_embed
        attns = []
        for layer in self.backbone:
            if return_attn:
                x, aw = layer(x, return_attn=True)
                attns.append(aw)
            else:
                x = layer(x)
        out = self.classifier(x.mean(dim=1))
        if return_attn:
            return out, attns
        return out


class MLP(nn.Module):
    """MLP baseline with comparable parameter count to GAT (~115K params).
    
    Flattens (N, C, 26) → (N, C×26) then passes through dense layers.
    """
    def __init__(self, n_ch=32, in_feats=26, n_classes=4, dropout=0.3):
        super().__init__()
        in_dim = n_ch * in_feats
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x.reshape(x.shape[0], -1))


class EEGDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).long()
    def __len__(self): return len(self.Y)
    def __getitem__(self, i): return self.X[i], self.Y[i]


# ═══════════════════════════════════════════════════════════════════════════════
# NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_features(X_train, X_test, n_ch, n_feats):
    """Z-score normalization fit ONLY on training data.
    
    Flattens (N, C, F) → (N*C, F) for fitting, normalizes per-feature
    across all channels, then reshapes back.
    """
    scaler = StandardScaler()
    X_tr_flat = X_train.reshape(-1, n_feats)
    X_te_flat = X_test.reshape(-1, n_feats)
    X_tr_norm = scaler.fit_transform(X_tr_flat).reshape(-1, n_ch, n_feats)
    X_te_norm = scaler.transform(X_te_flat).reshape(-1, n_ch, n_feats)
    return X_tr_norm.astype(np.float32), X_te_norm.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(model, X_tr, Y_tr, X_te, Y_te, hparams=None, seed=42):
    """Train any model (GAT or MLP) for 4-class classification.
    
    Returns predictions on test set.
    """
    if hparams is None:
        hparams = DEFAULT_HPARAMS

    torch.manual_seed(seed)
    # Infer n_classes from the model's final layer to support both 2-class and 4-class
    n_classes = next(reversed(list(model.parameters()))).shape[0]

    noise_std = hparams.get('input_noise_std', 0.0)
    label_smooth = hparams.get('label_smoothing', 0.0)

    counts = np.bincount(Y_tr, minlength=n_classes).astype(float)
    weights = torch.tensor(counts.sum() / (n_classes * counts + 1e-6),
                          dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smooth)

    optimizer = optim.AdamW(model.parameters(),
                           lr=hparams.get('lr', 5e-4),
                           weight_decay=hparams.get('weight_decay', 1e-4))
    epochs = hparams.get('epochs', 200)
    patience = hparams.get('patience', 20)
    batch_size = hparams.get('batch_size', 512)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_dl = DataLoader(EEGDataset(X_tr, Y_tr), batch_size=batch_size,
                          shuffle=True, drop_last=len(X_tr) > batch_size)
    val_dl = DataLoader(EEGDataset(X_te, Y_te), batch_size=batch_size, shuffle=False)
    best_loss, wait, best_state = float('inf'), 0, None

    for epoch in range(epochs):
        model.train()
        for Xb, Yb in train_dl:
            Xb, Yb = Xb.to(device), Yb.to(device)
            if noise_std > 0:
                Xb = Xb + torch.randn_like(Xb) * noise_std
            optimizer.zero_grad()
            loss = criterion(model(Xb), Yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        v_loss_sum, v_n = 0.0, 0
        with torch.no_grad():
            for Xb, Yb in val_dl:
                Xb, Yb = Xb.to(device), Yb.to(device)
                v_loss_sum += criterion(model(Xb), Yb).item() * len(Xb)
                v_n += len(Xb)
        v_loss = v_loss_sum / v_n

        if v_loss < best_loss:
            best_loss = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for Xb, _ in val_dl:
            all_preds.append(model(Xb.to(device)).argmax(dim=1).cpu().numpy())
    return np.concatenate(all_preds)


def create_gat(n_ch, n_feats, hparams, n_classes=None):
    return DeepGAT(
        n_ch=n_ch, in_feats=n_feats, n_classes=n_classes or N_CLASSES,
        backbone_dims=hparams.get('backbone_dims', [64, 64, 64]),
        num_heads=hparams.get('num_heads', 4),
        attn_dropout=hparams.get('attn_dropout', 0.05),
        dense=hparams.get('head_dense', 128),
        head_dropout=hparams.get('head_dropout', 0.3),
    ).to(device)


def create_mlp(n_ch, n_feats, hparams, n_classes=None):
    return MLP(
        n_ch=n_ch, in_feats=n_feats, n_classes=n_classes or N_CLASSES,
        dropout=hparams.get('head_dropout', 0.3),
    ).to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_deap_4class():
    """Load DEAP features with 4-class quadrant labels (per-subject median).
    
    Returns dict[subject_id] -> {'X': (N, 32, 26), 'Y': (N,), 'trials': (N,)}
    where Y ∈ {0: HAHV, 1: HALV, 2: LAHV, 3: LALV}
    """
    deap_data = {}
    for sub in SUBJECTS:
        fp = os.path.join(DEAP_FEAT_DIR, f's{sub}.npz')
        if not os.path.exists(fp):
            continue
        d = np.load(fp)
        X = d['features']  # (N, 32, 26)
        labels = d['labels']  # (N, 2): arousal, valence
        trials = d['trials']  # (N,)

        # Per-subject median threshold for balanced quadrants
        aro_median = np.median(labels[:, 0])
        val_median = np.median(labels[:, 1])

        high_aro = labels[:, 0] > aro_median
        high_val = labels[:, 1] > val_median

        # 4-class quadrant assignment
        Y = np.zeros(len(labels), dtype=np.int64)
        Y[high_aro & high_val] = 0   # HAHV (happy)
        Y[high_aro & ~high_val] = 1  # HALV (stressed)
        Y[~high_aro & high_val] = 2  # LAHV (calm)
        Y[~high_aro & ~high_val] = 3 # LALV (sad)

        deap_data[sub] = {'X': X, 'Y': Y, 'trials': trials}

    return deap_data


def load_deap_binary():
    """Load DEAP features with binary labels (per-subject median threshold).
    
    Returns dict[subject_id] -> {'X': (N, 32, 26), 'Y_aro': (N,), 'Y_val': (N,), 'trials': (N,)}
    where Y_aro, Y_val ∈ {0: low, 1: high}
    """
    deap_data = {}
    for sub in SUBJECTS:
        fp = os.path.join(DEAP_FEAT_DIR, f's{sub}.npz')
        if not os.path.exists(fp):
            continue
        d = np.load(fp)
        X = d['features']   # (N, 32, 26)
        labels = d['labels'] # (N, 2): arousal, valence
        trials = d['trials'] # (N,)

        # Per-subject median threshold
        aro_median = np.median(labels[:, 0])
        val_median = np.median(labels[:, 1])

        Y_aro = (labels[:, 0] > aro_median).astype(np.int64)
        Y_val = (labels[:, 1] > val_median).astype(np.int64)

        deap_data[sub] = {'X': X, 'Y_aro': Y_aro, 'Y_val': Y_val, 'trials': trials}

    return deap_data


def load_openbci():
    """Load OpenBCI features (4-class: calm=0, happy=1, sad=2, stressed=3).
    
    Remaps to match DEAP quadrant convention:
      happy(1) → HAHV(0), stressed(3) → HALV(1), calm(0) → LAHV(2), sad(2) → LALV(3)
    """
    if not os.path.exists(OPENBCI_FEAT):
        raise FileNotFoundError(f'OpenBCI features not found: {OPENBCI_FEAT}\n'
                               f'Run: python evaluation/feature_extraction.py --dataset openbci --force')
    d = np.load(OPENBCI_FEAT)
    X = d['X']       # (N, 16, 26)
    Y_raw = d['Y']   # (N,) with calm=0, happy=1, sad=2, stressed=3
    groups = d['groups']  # (N,) trial IDs

    # Remap to quadrant convention: HAHV=0, HALV=1, LAHV=2, LALV=3
    remap = {1: 0, 3: 1, 0: 2, 2: 3}  # happy→HAHV, stressed→HALV, calm→LAHV, sad→LALV
    Y = np.array([remap[y] for y in Y_raw], dtype=np.int64)

    return X, Y, groups


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_fold(X_tr, Y_tr, X_te, Y_te, n_ch, n_feats, hparams, seed, n_classes=None):
    """Run MLP + GAT on one fold, return predictions and metrics."""
    nc = n_classes or N_CLASSES
    X_tr_n, X_te_n = normalize_features(X_tr, X_te, n_ch, n_feats)

    results = {}
    for model_name, create_fn in [('MLP', create_mlp), ('GAT', create_gat)]:
        model = create_fn(n_ch, n_feats, hparams, n_classes=nc)
        preds = train_model(model, X_tr_n, Y_tr, X_te_n, Y_te, hparams, seed)
        f1 = f1_score(Y_te, preds, average='macro', zero_division=0)
        cm = confusion_matrix(Y_te, preds, labels=list(range(nc)))
        per_class_f1 = f1_score(Y_te, preds, average=None, labels=list(range(nc)), zero_division=0)
        results[model_name] = {
            'preds': preds, 'f1': f1, 'cm': cm, 'per_class_f1': per_class_f1
        }

    return results


def bootstrap_ci(scores, n_boot=1000, ci=0.95):
    """Compute bootstrap confidence interval."""
    scores = np.array(scores)
    boot_means = []
    for _ in range(n_boot):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        boot_means.append(np.mean(sample))
    boot_means = sorted(boot_means)
    lo = boot_means[int((1 - ci) / 2 * n_boot)]
    hi = boot_means[int((1 + ci) / 2 * n_boot)]
    return lo, hi


def tier_comparison_stats(scores_a, scores_b, name_a, name_b, n_comparisons=6):
    """Paired t-test + Cohen's d between two tiers, with Bonferroni correction."""
    t_stat, p_raw = scipy_stats.ttest_rel(scores_a, scores_b)
    p_corrected = min(p_raw * n_comparisons, 1.0)
    diff = np.array(scores_a) - np.array(scores_b)
    cohens_d = diff.mean() / (diff.std() + 1e-10)
    return {
        'comparison': f'{name_a} vs {name_b}',
        't_stat': float(t_stat),
        'p_raw': float(p_raw),
        'p_corrected': float(p_corrected),
        'cohens_d': float(cohens_d),
        'significant_bonferroni': p_corrected < 0.05,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Load hyperparameters
    hparams_file = os.path.join(OUT_DIR, 'best_hparams.json')
    if os.path.exists(hparams_file):
        with open(hparams_file) as f:
            hparams = json.load(f)
        print(f'Loaded tuned hyperparameters from {hparams_file}')
    else:
        hparams = DEFAULT_HPARAMS
        print('Using default hyperparameters')

    print(f'\n{"="*70}')
    print(f'UNIFIED FOUR-TIER EVALUATION (MLP + GAT)')
    print(f'{"="*70}')
    print(f'Device: {device}')
    print(f'Classification: 4-class quadrant ({", ".join(CLASS_NAMES)})')
    print(f'Metric: Macro F1 (chance = 25%)')
    print(f'Folds: {N_FOLDS} | Epochs: {hparams["epochs"]} | Patience: {hparams["patience"]}')
    print(f'Backbone: {hparams["backbone_dims"]} | Heads: {hparams["num_heads"]}')
    print(f'LR: {hparams["lr"]} | Batch: {hparams["batch_size"]}')
    print(f'Seed: {SEED} | Deterministic: True')
    print()

    # Report parameter counts
    _gat = create_gat(DEAP_N_CH, 26, hparams)
    _mlp = create_mlp(DEAP_N_CH, 26, hparams)
    print(f'Parameter counts (DEAP, 32ch):')
    print(f'  GAT: {count_parameters(_gat):,}')
    print(f'  MLP: {count_parameters(_mlp):,}')
    del _gat, _mlp

    # Load data
    print('\nLoading DEAP features (4-class quadrant, per-subject median)...')
    deap_data = load_deap_4class()
    n_feats = list(deap_data.values())[0]['X'].shape[2]
    total_windows = sum(len(d['X']) for d in deap_data.values())
    print(f'  Loaded: {len(deap_data)} subjects, {total_windows:,} windows')
    print(f'  Shape: ({DEAP_N_CH}, {n_feats}) per window')
    # Show class balance for first subject
    first_sub = list(deap_data.values())[0]
    print(f'  Example class dist (s01): {dict(zip(*np.unique(first_sub["Y"], return_counts=True)))}')

    print('\nLoading OpenBCI features...')
    openbci_X, openbci_Y, openbci_groups = load_openbci()
    ob_n_feats = openbci_X.shape[2]
    print(f'  Windows: {len(openbci_X):,} | Trials: {len(np.unique(openbci_groups))}')
    print(f'  Shape: ({OPENBCI_N_CH}, {ob_n_feats}) per window')
    print(f'  Class dist: {dict(zip(*np.unique(openbci_Y, return_counts=True)))}')
    print(flush=True)

    # Results storage
    all_results = {
        'config': {
            'n_folds': N_FOLDS, 'n_classes': N_CLASSES,
            'deap_n_ch': DEAP_N_CH, 'openbci_n_ch': OPENBCI_N_CH,
            'n_feats': n_feats, 'seed': SEED, 'hparams': hparams,
            'class_names': CLASS_NAMES,
        },
        'DEAP': {}, 'OpenBCI': {}, 'statistics': {}
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # DEAP TIER 0: RANDOM SPLIT (leakage from overlapping windows)
    # ═══════════════════════════════════════════════════════════════════════════

    print('=' * 70)
    print('DEAP TIER 0: RANDOM SPLIT (50% window overlap → data leakage)')
    print('=' * 70)
    t0 = time.time()

    all_X_deap = np.concatenate([d['X'] for d in deap_data.values()])
    all_Y_deap = np.concatenate([d['Y'] for d in deap_data.values()])

    tier0_scores = {'MLP': [], 'GAT': []}
    tier0_cms = {'MLP': [], 'GAT': []}

    # 5 random 80/20 splits for stability
    for split_i in range(5):
        tr_idx, te_idx = train_test_split(
            np.arange(len(all_X_deap)), test_size=0.2,
            stratify=all_Y_deap, random_state=SEED + split_i)
        
        res = evaluate_fold(all_X_deap[tr_idx], all_Y_deap[tr_idx],
                           all_X_deap[te_idx], all_Y_deap[te_idx],
                           DEAP_N_CH, n_feats, hparams, SEED + split_i)
        for m in ('MLP', 'GAT'):
            tier0_scores[m].append(res[m]['f1'])
            tier0_cms[m].append(res[m]['cm'].tolist())
        print(f'  Split {split_i+1}: MLP={res["MLP"]["f1"]:.4f}, GAT={res["GAT"]["f1"]:.4f}')

    all_results['DEAP']['Tier0'] = {
        m: {'f1_scores': tier0_scores[m],
            'f1_mean': float(np.mean(tier0_scores[m])),
            'f1_std': float(np.std(tier0_scores[m])),
            'confusion_matrices': tier0_cms[m]}
        for m in ('MLP', 'GAT')
    }
    print(f'\n-- TIER 0 SUMMARY -- [{time.time()-t0:.0f}s]')
    for m in ('MLP', 'GAT'):
        print(f'  {m}: F1 = {np.mean(tier0_scores[m]):.4f} +/- {np.std(tier0_scores[m]):.4f}')
    print()

    # ═══════════════════════════════════════════════════════════════════════════
    # DEAP TIER 1: WITHIN-SUBJECT CV (StratifiedKFold, windows from same trial can leak)
    # ═══════════════════════════════════════════════════════════════════════════

    print('=' * 70)
    print('DEAP TIER 1: WITHIN-SUBJECT CV (StratifiedKFold)')
    print('=' * 70)
    t0 = time.time()

    tier1_per_subject = {'MLP': [], 'GAT': []}

    for sub_id, sub_data in tqdm(list(deap_data.items()), desc='Tier1'):
        X, Y, trials = sub_data['X'], sub_data['Y'], sub_data['trials']

        if len(np.unique(Y)) < 2:
            continue

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        sub_scores = {'MLP': [], 'GAT': []}

        for fold, (tr_idx, te_idx) in enumerate(skf.split(X, Y)):
            res = evaluate_fold(X[tr_idx], Y[tr_idx], X[te_idx], Y[te_idx],
                               DEAP_N_CH, n_feats, hparams, SEED + fold)
            for m in ('MLP', 'GAT'):
                sub_scores[m].append(res[m]['f1'])

        for m in ('MLP', 'GAT'):
            tier1_per_subject[m].append(np.mean(sub_scores[m]))

    all_results['DEAP']['Tier1'] = {
        m: {'per_subject': tier1_per_subject[m],
            'f1_mean': float(np.mean(tier1_per_subject[m])),
            'f1_std': float(np.std(tier1_per_subject[m])),
            'ci_95': list(bootstrap_ci(tier1_per_subject[m]))}
        for m in ('MLP', 'GAT')
    }
    print(f'\n-- TIER 1 SUMMARY -- [{time.time()-t0:.0f}s]')
    for m in ('MLP', 'GAT'):
        ci = bootstrap_ci(tier1_per_subject[m])
        print(f'  {m}: F1 = {np.mean(tier1_per_subject[m]):.4f} +/- {np.std(tier1_per_subject[m]):.4f}  '
              f'95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]')
    print()

    # ═══════════════════════════════════════════════════════════════════════════
    # DEAP TIER 2: TRIAL-AWARE WITHIN-SUBJECT (StratifiedGroupKFold)
    # ═══════════════════════════════════════════════════════════════════════════

    print('=' * 70)
    print('DEAP TIER 2: TRIAL-AWARE WITHIN-SUBJECT (StratifiedGroupKFold)')
    print('=' * 70)
    t0 = time.time()

    tier2_per_subject = {'MLP': [], 'GAT': []}

    for sub_id, sub_data in tqdm(list(deap_data.items()), desc='Tier2'):
        X, Y, trials = sub_data['X'], sub_data['Y'], sub_data['trials']

        n_unique_trials = len(np.unique(trials))
        actual_folds = min(N_FOLDS, n_unique_trials)
        if len(np.unique(Y)) < 2 or actual_folds < 2:
            continue

        sgkf = StratifiedGroupKFold(n_splits=actual_folds, shuffle=True, random_state=SEED)
        sub_scores = {'MLP': [], 'GAT': []}

        for fold, (tr_idx, te_idx) in enumerate(sgkf.split(X, Y, groups=trials)):
            res = evaluate_fold(X[tr_idx], Y[tr_idx], X[te_idx], Y[te_idx],
                               DEAP_N_CH, n_feats, hparams, SEED + fold)
            for m in ('MLP', 'GAT'):
                sub_scores[m].append(res[m]['f1'])

        for m in ('MLP', 'GAT'):
            tier2_per_subject[m].append(np.mean(sub_scores[m]))

    all_results['DEAP']['Tier2'] = {
        m: {'per_subject': tier2_per_subject[m],
            'f1_mean': float(np.mean(tier2_per_subject[m])),
            'f1_std': float(np.std(tier2_per_subject[m])),
            'ci_95': list(bootstrap_ci(tier2_per_subject[m]))}
        for m in ('MLP', 'GAT')
    }
    print(f'\n-- TIER 2 SUMMARY -- [{time.time()-t0:.0f}s]')
    for m in ('MLP', 'GAT'):
        ci = bootstrap_ci(tier2_per_subject[m])
        print(f'  {m}: F1 = {np.mean(tier2_per_subject[m]):.4f} +/- {np.std(tier2_per_subject[m]):.4f}  '
              f'95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]')
    print()

    # ═══════════════════════════════════════════════════════════════════════════
    # DEAP TIER 3: CROSS-SUBJECT LOSO
    # ═══════════════════════════════════════════════════════════════════════════

    print('=' * 70)
    print('DEAP TIER 3: CROSS-SUBJECT LOSO (train N-1, test 1)')
    print('=' * 70)
    t0 = time.time()

    all_subs_arr = np.concatenate([
        np.full(len(d['X']), int(sid)) for sid, d in deap_data.items()
    ])

    tier3_per_subject = {'MLP': [], 'GAT': []}

    for test_sub in tqdm(sorted(deap_data.keys()), desc='LOSO'):
        test_mask = all_subs_arr == int(test_sub)
        train_mask = ~test_mask
        X_tr, X_te = all_X_deap[train_mask], all_X_deap[test_mask]
        Y_tr, Y_te = all_Y_deap[train_mask], all_Y_deap[test_mask]

        res = evaluate_fold(X_tr, Y_tr, X_te, Y_te,
                           DEAP_N_CH, n_feats, hparams, SEED)
        for m in ('MLP', 'GAT'):
            tier3_per_subject[m].append(res[m]['f1'])
        print(f'  s{test_sub}: MLP={res["MLP"]["f1"]:.4f}, GAT={res["GAT"]["f1"]:.4f}')

    all_results['DEAP']['Tier3'] = {
        m: {'per_subject': tier3_per_subject[m],
            'f1_mean': float(np.mean(tier3_per_subject[m])),
            'f1_std': float(np.std(tier3_per_subject[m])),
            'ci_95': list(bootstrap_ci(tier3_per_subject[m]))}
        for m in ('MLP', 'GAT')
    }
    print(f'\n-- TIER 3 SUMMARY (LOSO) -- [{time.time()-t0:.0f}s]')
    for m in ('MLP', 'GAT'):
        ci = bootstrap_ci(tier3_per_subject[m])
        print(f'  {m}: F1 = {np.mean(tier3_per_subject[m]):.4f} +/- {np.std(tier3_per_subject[m]):.4f}  '
              f'95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]')
    print()

    # ═══════════════════════════════════════════════════════════════════════════
    # OPENBCI: TIER 0 + TIER 1 + TIER 2
    # ═══════════════════════════════════════════════════════════════════════════

    print('=' * 70)
    print('OpenBCI 4-CLASS EVALUATION (Tier 0 + Tier 1 + Tier 2)')
    print('=' * 70)
    t0 = time.time()

    # Tier 0: Random split
    print('\n-- OpenBCI Tier 0: Random Split --')
    ob_tier0 = {'MLP': [], 'GAT': []}
    for split_i in range(5):
        tr_idx, te_idx = train_test_split(
            np.arange(len(openbci_X)), test_size=0.2,
            stratify=openbci_Y, random_state=SEED + split_i)
        res = evaluate_fold(openbci_X[tr_idx], openbci_Y[tr_idx],
                           openbci_X[te_idx], openbci_Y[te_idx],
                           OPENBCI_N_CH, ob_n_feats, hparams, SEED + split_i)
        for m in ('MLP', 'GAT'):
            ob_tier0[m].append(res[m]['f1'])
        print(f'  Split {split_i+1}: MLP={res["MLP"]["f1"]:.4f}, GAT={res["GAT"]["f1"]:.4f}')

    # Tier 1: StratifiedKFold (leaky)
    print('\n-- OpenBCI Tier 1: StratifiedKFold --')
    skf_ob = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    ob_tier1 = {'MLP': [], 'GAT': []}
    for fold, (tr_idx, te_idx) in enumerate(skf_ob.split(openbci_X, openbci_Y)):
        res = evaluate_fold(openbci_X[tr_idx], openbci_Y[tr_idx],
                           openbci_X[te_idx], openbci_Y[te_idx],
                           OPENBCI_N_CH, ob_n_feats, hparams, SEED + fold)
        for m in ('MLP', 'GAT'):
            ob_tier1[m].append(res[m]['f1'])
        print(f'  Fold {fold+1:2d}: MLP={res["MLP"]["f1"]:.4f}, GAT={res["GAT"]["f1"]:.4f}')

    # Tier 2: Trial-aware (StratifiedGroupKFold)
    print('\n-- OpenBCI Tier 2: Trial-Aware --')
    sgkf_ob = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    ob_tier2 = {'MLP': [], 'GAT': []}
    for fold, (tr_idx, te_idx) in enumerate(
        sgkf_ob.split(openbci_X, openbci_Y, groups=openbci_groups)):
        res = evaluate_fold(openbci_X[tr_idx], openbci_Y[tr_idx],
                           openbci_X[te_idx], openbci_Y[te_idx],
                           OPENBCI_N_CH, ob_n_feats, hparams, SEED + fold)
        for m in ('MLP', 'GAT'):
            ob_tier2[m].append(res[m]['f1'])
        print(f'  Fold {fold+1:2d}: MLP={res["MLP"]["f1"]:.4f}, GAT={res["GAT"]["f1"]:.4f}')

    all_results['OpenBCI'] = {
        'Tier0': {m: {'f1_scores': ob_tier0[m],
                      'f1_mean': float(np.mean(ob_tier0[m])),
                      'f1_std': float(np.std(ob_tier0[m]))}
                  for m in ('MLP', 'GAT')},
        'Tier1': {m: {'f1_scores': ob_tier1[m],
                      'f1_mean': float(np.mean(ob_tier1[m])),
                      'f1_std': float(np.std(ob_tier1[m]))}
                  for m in ('MLP', 'GAT')},
        'Tier2': {m: {'f1_scores': ob_tier2[m],
                      'f1_mean': float(np.mean(ob_tier2[m])),
                      'f1_std': float(np.std(ob_tier2[m]))}
                  for m in ('MLP', 'GAT')},
    }

    print(f'\n-- OpenBCI SUMMARY -- [{time.time()-t0:.0f}s]')
    for tier_name, tier_data in [('Tier0', ob_tier0), ('Tier1', ob_tier1), ('Tier2', ob_tier2)]:
        for m in ('MLP', 'GAT'):
            print(f'  {tier_name} {m}: F1 = {np.mean(tier_data[m]):.4f} +/- {np.std(tier_data[m]):.4f}')
    print()

    # ═══════════════════════════════════════════════════════════════════════════
    # STATISTICAL COMPARISONS (Bonferroni-corrected)
    # ═══════════════════════════════════════════════════════════════════════════

    print('=' * 70)
    print('STATISTICAL COMPARISONS (paired t-tests, Bonferroni-corrected)')
    print('=' * 70)

    # Compare tiers for GAT on DEAP (using per-subject means)
    deap_tier_scores = {
        'Tier1': tier1_per_subject['GAT'],
        'Tier2': tier2_per_subject['GAT'],
        'Tier3': tier3_per_subject['GAT'],
    }
    
    tier_pairs = [('Tier1', 'Tier2'), ('Tier1', 'Tier3'), ('Tier2', 'Tier3')]
    n_comp = len(tier_pairs)
    stats_results = []
    
    for ta, tb in tier_pairs:
        stat = tier_comparison_stats(
            deap_tier_scores[ta], deap_tier_scores[tb], ta, tb, n_comp)
        stats_results.append(stat)
        sig = '***' if stat['p_corrected'] < 0.001 else ('**' if stat['p_corrected'] < 0.01 else ('*' if stat['p_corrected'] < 0.05 else 'ns'))
        print(f'  {ta} vs {tb}: t={stat["t_stat"]:.3f}, p_raw={stat["p_raw"]:.4f}, '
              f'p_corr={stat["p_corrected"]:.4f}, d={stat["cohens_d"]:.2f} [{sig}]')

    # Compare MLP vs GAT within each tier
    print('\n  MLP vs GAT (per tier):')
    model_comparisons = []
    for tier_name, mlp_scores, gat_scores in [
        ('Tier1', tier1_per_subject['MLP'], tier1_per_subject['GAT']),
        ('Tier2', tier2_per_subject['MLP'], tier2_per_subject['GAT']),
        ('Tier3', tier3_per_subject['MLP'], tier3_per_subject['GAT']),
    ]:
        stat = tier_comparison_stats(gat_scores, mlp_scores, f'GAT_{tier_name}', f'MLP_{tier_name}', 3)
        model_comparisons.append(stat)
        sig = '***' if stat['p_corrected'] < 0.001 else ('**' if stat['p_corrected'] < 0.01 else ('*' if stat['p_corrected'] < 0.05 else 'ns'))
        print(f'    {tier_name}: GAT-MLP diff = {np.mean(gat_scores)-np.mean(mlp_scores):.4f}, '
              f'd={stat["cohens_d"]:.2f} [{sig}]')

    all_results['statistics'] = {
        'tier_comparisons': stats_results,
        'model_comparisons': model_comparisons,
    }
    print()

    # ═══════════════════════════════════════════════════════════════════════════
    # XAI: ATTENTION ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════

    print('=' * 70)
    print('XAI: GAT ATTENTION ANALYSIS')
    print('=' * 70)

    # Train a GAT on all DEAP data (80/20 split) for attention extraction
    n = len(all_X_deap)
    idx = np.random.permutation(n)
    split = int(0.8 * n)
    X_tr_xai, X_te_xai = all_X_deap[idx[:split]], all_X_deap[idx[split:]]
    Y_tr_xai, Y_te_xai = all_Y_deap[idx[:split]], all_Y_deap[idx[split:]]

    X_tr_xai_n, X_te_xai_n = normalize_features(X_tr_xai, X_te_xai, DEAP_N_CH, n_feats)

    print('Training GAT for attention extraction...')
    xai_model = create_gat(DEAP_N_CH, n_feats, hparams)
    _ = train_model(xai_model, X_tr_xai_n, Y_tr_xai, X_te_xai_n, Y_te_xai, hparams, SEED)

    # Extract attention
    n_xai_samples = min(1000, len(X_te_xai_n))
    x_input = torch.from_numpy(X_te_xai_n[:n_xai_samples]).float().to(device)
    xai_model.eval()
    with torch.no_grad():
        _, attns = xai_model(x_input, return_attn=True)

    attn_maps = [aw.mean(dim=(0, 1)).cpu().numpy() for aw in attns]
    final_attn = attn_maps[-1]
    ch_importance = final_attn.sum(axis=0)

    CHANNEL_NAMES_DEAP = [
        'Fp1','AF3','F3','F7','FC5','FC1','C3','T7',
        'CP5','CP1','P3','P7','PO3','O1','Oz','Pz',
        'Fp2','AF4','F4','F8','FC6','FC2','C4','T8',
        'CP6','CP2','P4','P8','PO4','O2','O9','O10'
    ]

    order = np.argsort(ch_importance)[::-1]
    print(f'\nTop-5 attended channels:')
    for i in range(5):
        print(f'  {i+1}. {CHANNEL_NAMES_DEAP[order[i]]} (attn={ch_importance[order[i]]:.4f})')

    all_results['attention'] = {
        'channel_importance': ch_importance.tolist(),
        'channel_names': CHANNEL_NAMES_DEAP,
        'attention_matrix': final_attn.tolist(),
    }
    print()

    # ═══════════════════════════════════════════════════════════════════════════
    # DEAP BINARY CLASSIFICATION (Arousal + Valence, 2-class each)
    # ═══════════════════════════════════════════════════════════════════════════

    print('=' * 70)
    print('DEAP BINARY CLASSIFICATION (per-subject median, chance = 50%)')
    print('=' * 70)
    t0 = time.time()

    deap_binary = load_deap_binary()
    all_results['DEAP_binary'] = {}

    for dim_name, y_key in [('arousal', 'Y_aro'), ('valence', 'Y_val')]:
        print(f'\n  --- {dim_name.upper()} ---')

        # Pool all data for this dimension
        all_X_bin = np.concatenate([d['X'] for d in deap_binary.values()])
        all_Y_bin = np.concatenate([d[y_key] for d in deap_binary.values()])

        dim_results = {}

        # Binary Tier 0: Random split
        print(f'  Tier 0 (random split):')
        bin_t0 = {'MLP': [], 'GAT': []}
        for split_i in range(5):
            tr_idx, te_idx = train_test_split(
                np.arange(len(all_X_bin)), test_size=0.2,
                stratify=all_Y_bin, random_state=SEED + split_i)
            res = evaluate_fold(all_X_bin[tr_idx], all_Y_bin[tr_idx],
                               all_X_bin[te_idx], all_Y_bin[te_idx],
                               DEAP_N_CH, n_feats, hparams, SEED + split_i, n_classes=2)
            for m in ('MLP', 'GAT'):
                bin_t0[m].append(res[m]['f1'])
        for m in ('MLP', 'GAT'):
            print(f'    {m}: {np.mean(bin_t0[m]):.4f} +/- {np.std(bin_t0[m]):.4f}')
        dim_results['Tier0'] = {
            m: {'f1_scores': bin_t0[m], 'f1_mean': float(np.mean(bin_t0[m])),
                'f1_std': float(np.std(bin_t0[m]))}
            for m in ('MLP', 'GAT')
        }

        # Binary Tier 1: Within-subject StratifiedKFold
        print(f'  Tier 1 (within-subject, leaky):')
        bin_t1 = {'MLP': [], 'GAT': []}
        for sub_id, sub_data in deap_binary.items():
            X, Y = sub_data['X'], sub_data[y_key]
            if len(np.unique(Y)) < 2:
                continue
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
            sub_scores = {'MLP': [], 'GAT': []}
            for fold, (tr_idx, te_idx) in enumerate(skf.split(X, Y)):
                res = evaluate_fold(X[tr_idx], Y[tr_idx], X[te_idx], Y[te_idx],
                                   DEAP_N_CH, n_feats, hparams, SEED + fold, n_classes=2)
                for m in ('MLP', 'GAT'):
                    sub_scores[m].append(res[m]['f1'])
            for m in ('MLP', 'GAT'):
                bin_t1[m].append(np.mean(sub_scores[m]))
        for m in ('MLP', 'GAT'):
            print(f'    {m}: {np.mean(bin_t1[m]):.4f} +/- {np.std(bin_t1[m]):.4f}')
        dim_results['Tier1'] = {
            m: {'per_subject': bin_t1[m], 'f1_mean': float(np.mean(bin_t1[m])),
                'f1_std': float(np.std(bin_t1[m])), 'ci_95': list(bootstrap_ci(bin_t1[m]))}
            for m in ('MLP', 'GAT')
        }

        # Binary Tier 2: Trial-aware within-subject
        print(f'  Tier 2 (trial-aware):')
        bin_t2 = {'MLP': [], 'GAT': []}
        for sub_id, sub_data in deap_binary.items():
            X, Y, trials = sub_data['X'], sub_data[y_key], sub_data['trials']
            n_unique = len(np.unique(trials))
            actual_folds = min(N_FOLDS, n_unique)
            if len(np.unique(Y)) < 2 or actual_folds < 2:
                continue
            sgkf = StratifiedGroupKFold(n_splits=actual_folds, shuffle=True, random_state=SEED)
            sub_scores = {'MLP': [], 'GAT': []}
            for fold, (tr_idx, te_idx) in enumerate(sgkf.split(X, Y, groups=trials)):
                res = evaluate_fold(X[tr_idx], Y[tr_idx], X[te_idx], Y[te_idx],
                                   DEAP_N_CH, n_feats, hparams, SEED + fold, n_classes=2)
                for m in ('MLP', 'GAT'):
                    sub_scores[m].append(res[m]['f1'])
            for m in ('MLP', 'GAT'):
                bin_t2[m].append(np.mean(sub_scores[m]))
        for m in ('MLP', 'GAT'):
            print(f'    {m}: {np.mean(bin_t2[m]):.4f} +/- {np.std(bin_t2[m]):.4f}')
        dim_results['Tier2'] = {
            m: {'per_subject': bin_t2[m], 'f1_mean': float(np.mean(bin_t2[m])),
                'f1_std': float(np.std(bin_t2[m])), 'ci_95': list(bootstrap_ci(bin_t2[m]))}
            for m in ('MLP', 'GAT')
        }

        # Binary Tier 3: LOSO
        print(f'  Tier 3 (LOSO):')
        all_subs_bin = np.concatenate([
            np.full(len(d['X']), int(sid)) for sid, d in deap_binary.items()
        ])
        bin_t3 = {'MLP': [], 'GAT': []}
        for test_sub in sorted(deap_binary.keys()):
            test_mask = all_subs_bin == int(test_sub)
            train_mask = ~test_mask
            Y_tr_bin = all_Y_bin[train_mask]
            Y_te_bin = all_Y_bin[test_mask]
            res = evaluate_fold(all_X_bin[train_mask], Y_tr_bin,
                               all_X_bin[test_mask], Y_te_bin,
                               DEAP_N_CH, n_feats, hparams, SEED, n_classes=2)
            for m in ('MLP', 'GAT'):
                bin_t3[m].append(res[m]['f1'])
        for m in ('MLP', 'GAT'):
            print(f'    {m}: {np.mean(bin_t3[m]):.4f} +/- {np.std(bin_t3[m]):.4f}')
        dim_results['Tier3'] = {
            m: {'per_subject': bin_t3[m], 'f1_mean': float(np.mean(bin_t3[m])),
                'f1_std': float(np.std(bin_t3[m])), 'ci_95': list(bootstrap_ci(bin_t3[m]))}
            for m in ('MLP', 'GAT')
        }

        all_results['DEAP_binary'][dim_name] = dim_results

    print(f'\n-- BINARY SUMMARY -- [{time.time()-t0:.0f}s]')
    for dim in ('arousal', 'valence'):
        print(f'  {dim.upper()}:')
        for tier in ('Tier0', 'Tier1', 'Tier2', 'Tier3'):
            td = all_results['DEAP_binary'][dim][tier]
            print(f'    {tier}: MLP={td["MLP"]["f1_mean"]:.4f}, GAT={td["GAT"]["f1_mean"]:.4f}')
    print()

    # ═══════════════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════════════════

    out_path = os.path.join(OUT_DIR, 'all_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'Results saved to: {out_path}')

    # ═══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════

    print(f'\n{"="*70}')
    print('FINAL RESULTS')
    print('=' * 70)
    print(f'\n  DEAP (20 subjects, 4-class quadrant, Macro F1)')
    print(f'  {"─"*55}')
    print(f'  {"Tier":<12} {"MLP":<20} {"GAT":<20}')
    print(f'  {"─"*55}')
    for tier_name in ('Tier0', 'Tier1', 'Tier2', 'Tier3'):
        td = all_results['DEAP'][tier_name]
        mlp_str = f'{td["MLP"]["f1_mean"]:.4f} +/- {td["MLP"].get("f1_std", 0):.4f}'
        gat_str = f'{td["GAT"]["f1_mean"]:.4f} +/- {td["GAT"].get("f1_std", 0):.4f}'
        print(f'  {tier_name:<12} {mlp_str:<20} {gat_str:<20}')

    print(f'\n  OpenBCI (1 subject, 4-class quadrant, Macro F1)')
    print(f'  {"─"*55}')
    print(f'  {"Tier":<12} {"MLP":<20} {"GAT":<20}')
    print(f'  {"─"*55}')
    for tier_name in ('Tier0', 'Tier1', 'Tier2'):
        td = all_results['OpenBCI'][tier_name]
        mlp_str = f'{td["MLP"]["f1_mean"]:.4f} +/- {td["MLP"]["f1_std"]:.4f}'
        gat_str = f'{td["GAT"]["f1_mean"]:.4f} +/- {td["GAT"]["f1_std"]:.4f}'
        print(f'  {tier_name:<12} {mlp_str:<20} {gat_str:<20}')

    print(f'\n  DEAP Binary (20 subjects, per-subject median, Macro F1)')
    print(f'  {"─"*55}')
    for dim in ('arousal', 'valence'):
        print(f'  {dim.upper()}:')
        print(f'  {"Tier":<12} {"MLP":<20} {"GAT":<20}')
        for tier_name in ('Tier0', 'Tier1', 'Tier2', 'Tier3'):
            td = all_results['DEAP_binary'][dim][tier_name]
            mlp_str = f'{td["MLP"]["f1_mean"]:.4f} +/- {td["MLP"].get("f1_std", 0):.4f}'
            gat_str = f'{td["GAT"]["f1_mean"]:.4f} +/- {td["GAT"].get("f1_std", 0):.4f}'
            print(f'  {tier_name:<12} {mlp_str:<20} {gat_str:<20}')

    print(f'\n{"="*70}')
    print(f'DONE — all results in: {OUT_DIR}')
    print(f'{"="*70}')
