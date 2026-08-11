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
import psutil
from collections import Counter
from scipy import stats as scipy_stats

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    StratifiedGroupKFold, StratifiedKFold, LeaveOneGroupOut, train_test_split,
    GroupShuffleSplit, StratifiedShuffleSplit
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
CKPT_DIR = os.path.join(OUT_DIR, 'ckpt')
os.makedirs(CKPT_DIR, exist_ok=True)
PROGRESS_FILE = os.path.join(OUT_DIR, 'progress.json')


# ═══════════════════════════════════════════════════════════════════════════════
# CHECKPOINTING + PROGRESS
# ═══════════════════════════════════════════════════════════════════════════════
# A full sweep runs for many hours. Every unit of work (one Tier-0 split, one
# subject of T1/T2, one LOSO fold, ...) is written to outputs/ckpt/ as soon as it
# finishes, so an interrupted run resumes exactly where it stopped instead of
# starting over. Delete outputs/ckpt/ to force a clean re-run.

def _np_safe(o):
    """Convert numpy scalar/array types so json.dump never fails on them."""
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f'Object of type {type(o).__name__} is not JSON serializable')


def log(msg=''):
    """Print and flush immediately -- a buffered log is useless for monitoring."""
    print(msg, flush=True)


# Rough relative cost of one unit, used only for the ETA estimate.
# Approximate wall-clock SECONDS per unit on an RTX 3080, used only to weight
# the ETA. Calibrated against measured runs rather than guessed from data size:
# an earlier version scaled these by window count, which rated a Tier-3 LOSO
# unit at roughly the cost of one Tier-2 subject when it is really ~10x more
# expensive (73K-window pooled training vs ~2.4K per-subject), making the ETA
# collapse before the most expensive tier had even started.
#   *_t0 / *_t3 units are one (split-or-subject, model) pair.
#   *_t1 / *_t2 units are one subject, covering all folds and models.
UNIT_COST = {
    'deap_t0': 600,   # measured: MLP ~500s, GAT ~710s on the pooled corpus
    'deap_t1': 161,   # measured: 7734s / 32 subjects, scaled to 2 models
    'deap_t2': 70,    # measured: 61-82s per subject
    'deap_t3': 726,   # deap_t0 scaled by train size (73,150 / 60,416)
    'openbci': 100,   # small corpus, 2 models per fold
    'xai': 700,       # one GAT on the pooled corpus
    'bin_t0': 600, 'bin_t1': 161, 'bin_t2': 70, 'bin_t3': 726,
}

_PROGRESS = {
    'started': None, 'total_cost': 0, 'done_cost': 0,
    'computed_cost': 0,          # excludes units replayed from checkpoints
    'units_total': 0, 'units_done': 0, 'phase': 'init',
    'resumed_units': 0, 'recent': [],
}


def progress_init(plan):
    """plan: list of (unit_kind, count) covering the whole sweep."""
    _PROGRESS['started'] = time.time()
    _PROGRESS['total_cost'] = sum(UNIT_COST[k] * n for k, n in plan)
    _PROGRESS['units_total'] = sum(n for _, n in plan)


def progress_write(phase, note=''):
    el = time.time() - _PROGRESS['started']
    frac = _PROGRESS['done_cost'] / max(_PROGRESS['total_cost'], 1)
    # Throughput must be measured over work this process actually computed.
    # Units replayed from checkpoints land in done_cost instantly, so including
    # them would make the rate (and therefore the ETA) wildly optimistic for the
    # rest of a resumed run -- it never washes out, because the resumed cost
    # stays in the numerator while elapsed only counts time since restart.
    computed = _PROGRESS['computed_cost']
    remaining = max(_PROGRESS['total_cost'] - _PROGRESS['done_cost'], 0)
    eta = (remaining * el / computed) if computed > 0 else None
    state = {
        'phase': phase,
        'note': note,
        'units_done': _PROGRESS['units_done'],
        'units_total': _PROGRESS['units_total'],
        'resumed_from_checkpoint': _PROGRESS['resumed_units'],
        'percent_complete': round(100 * frac, 2),
        'units_computed_this_session': _PROGRESS['units_done'] - _PROGRESS['resumed_units'],
        'elapsed_hours': round(el / 3600, 2),
        'eta_hours_remaining': round(eta / 3600, 2) if eta else None,
        'eta_finish_local': (time.strftime('%Y-%m-%d %H:%M',
                                           time.localtime(time.time() + eta))
                             if eta else None),
        'updated': time.strftime('%Y-%m-%d %H:%M:%S'),
        'recent': _PROGRESS['recent'][-12:],
    }
    tmp = PROGRESS_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2, default=_np_safe)
    os.replace(tmp, PROGRESS_FILE)   # atomic: readers never see a partial file
    return state


def unit(kind, key, fn, phase=''):
    """Run fn() unless its checkpoint already exists; record progress either way."""
    fp = os.path.join(CKPT_DIR, f'{key}.json')
    if os.path.exists(fp):
        try:
            with open(fp) as f:
                val = json.load(f)
            _PROGRESS['units_done'] += 1
            _PROGRESS['done_cost'] += UNIT_COST[kind]
            _PROGRESS['resumed_units'] += 1
            return val
        except (json.JSONDecodeError, OSError):
            log(f'  [ckpt] {key} unreadable, recomputing')

    t0 = time.time()
    val = fn()
    dt = time.time() - t0

    tmp = fp + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(val, f, default=_np_safe)
    os.replace(tmp, fp)

    _PROGRESS['units_done'] += 1
    _PROGRESS['done_cost'] += UNIT_COST[kind]
    _PROGRESS['computed_cost'] += UNIT_COST[kind]
    _PROGRESS['recent'].append(f'{key} ({dt:.0f}s)')
    st = progress_write(phase or kind, key)
    ep = val.get('epochs_ran') if isinstance(val, dict) else None
    ep_str = f' | {ep} epochs ({dt/ep:.2f}s/ep)' if ep else ''
    log(f'  [{st["percent_complete"]:5.1f}%] {key} done in {dt:.0f}s{ep_str} | '
        f'unit {st["units_done"]}/{st["units_total"]} | '
        f'elapsed {st["elapsed_hours"]:.2f}h'
        + (f' | ETA {st["eta_hours_remaining"]:.1f}h (~{st["eta_finish_local"]})'
           if st['eta_hours_remaining'] is not None else ''))
    return val

# ─── Constants ────────────────────────────────────────────────────────────────
DEAP_N_CH = 32
OPENBCI_N_CH = 16
N_CLASSES = 4
N_FOLDS = 10
SUBJECTS = [f'{i:02d}' for i in range(1, 33)]  # full DEAP corpus (s01-s32)

# Class labels (same mapping for both datasets)
CLASS_NAMES = ['HAHV', 'HALV', 'LAHV', 'LALV']  # happy, stressed, calm, sad

# Architectures evaluated. LightGAT (a 16K-param single-layer capacity control)
# was dropped from this run: it accounted for a third of total compute while
# consistently tracking the full GAT to within ~1 F1 point, and the LOSO tiers
# made the full three-architecture sweep too expensive to finish in time.
# Re-add 'LightGAT' here (and to MODEL_FACTORIES) to restore it.
MODELS = ('MLP', 'GAT')

# Binary LOSO accounted for ~half the total sweep runtime for the paper's least
# load-bearing result; binary is reported at Tiers 0-2. See the comment at the
# binary Tier 3 block for the full rationale.
RUN_BINARY_T3 = False

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

VAL_FRACTION = 0.15   # of the TRAINING set, held out for early stopping


def make_val_split(Y_tr, groups_tr, seed):
    """Indices of a validation split carved out of the TRAINING data.

    Early stopping must never see the test fold, so the stopping signal comes
    from data the model is allowed to look at. Where the tier defines a grouping
    (trials at T2, subjects at T3), the split respects it, so the early-stopping
    signal reflects the same kind of generalization the tier is testing rather
    than a within-group optimistic estimate.

    Returns (sub_train_idx, val_idx), or (None, None) if the training set is too
    small or too imbalanced to split safely -- in which case the caller trains
    for a fixed number of epochs instead of early stopping.
    """
    n = len(Y_tr)
    if n < 50:
        return None, None
    try:
        if groups_tr is not None and len(np.unique(groups_tr)) >= 5:
            splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION,
                                         random_state=seed)
            tr_i, va_i = next(splitter.split(np.zeros(n), Y_tr, groups=groups_tr))
        else:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=VAL_FRACTION,
                                              random_state=seed)
            tr_i, va_i = next(splitter.split(np.zeros(n), Y_tr))
    except ValueError:
        return None, None
    # both sides must contain every class, or the weighted loss is undefined
    if len(np.unique(Y_tr[tr_i])) < len(np.unique(Y_tr)) or len(va_i) < 10:
        return None, None
    return tr_i, va_i


def train_model(model, X_tr, Y_tr, X_te, Y_te, hparams=None, seed=42,
                groups_tr=None):
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

    # Early stopping is driven by a validation split carved out of the TRAINING
    # data. The test fold is used only once, for the final prediction below.
    sub_i, val_i = make_val_split(Y_tr, groups_tr, seed)
    if sub_i is None:
        X_fit, Y_fit = X_tr, Y_tr
        val_dl = None            # too small to split: train a fixed budget
    else:
        X_fit, Y_fit = X_tr[sub_i], Y_tr[sub_i]
        val_dl = DataLoader(EEGDataset(X_tr[val_i], Y_tr[val_i]),
                            batch_size=batch_size, shuffle=False)

    train_dl = DataLoader(EEGDataset(X_fit, Y_fit), batch_size=batch_size,
                          shuffle=True, drop_last=len(X_fit) > batch_size)
    test_dl = DataLoader(EEGDataset(X_te, Y_te), batch_size=batch_size, shuffle=False)
    best_loss, wait, best_state, epochs_ran = float('inf'), 0, None, 0

    for epoch in range(epochs):
        epochs_ran = epoch + 1
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

        if val_dl is None:
            continue          # no validation split available: run the full budget

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

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for Xb, _ in test_dl:          # the test fold, used exactly once
            all_preds.append(model(Xb.to(device)).argmax(dim=1).cpu().numpy())
    return np.concatenate(all_preds), epochs_ran


def create_gat(n_ch, n_feats, hparams, n_classes=None):
    return DeepGAT(
        n_ch=n_ch, in_feats=n_feats, n_classes=n_classes or N_CLASSES,
        backbone_dims=hparams.get('backbone_dims', [64, 64, 64]),
        num_heads=hparams.get('num_heads', 4),
        attn_dropout=hparams.get('attn_dropout', 0.05),
        dense=hparams.get('head_dense', 128),
        head_dropout=hparams.get('head_dropout', 0.3),
    ).to(device)


def create_light_gat(n_ch, n_feats, hparams, n_classes=None):
    """Single-layer GAT with reduced model dimension to avoid over-smoothing."""
    return DeepGAT(
        n_ch=n_ch, in_feats=n_feats, n_classes=n_classes or N_CLASSES,
        backbone_dims=[48],  # Single layer, d=48
        num_heads=4,
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

_ALL_FACTORIES = {'MLP': create_mlp, 'GAT': create_gat, 'LightGAT': create_light_gat}
MODEL_FACTORIES = [(m, _ALL_FACTORIES[m]) for m in MODELS]


def evaluate_one_model(model_name, X_tr_n, Y_tr, X_te_n, Y_te, n_ch, n_feats,
                       hparams, seed, n_classes=None, groups_tr=None):
    """Train + evaluate a single architecture on already-normalized data."""
    nc = n_classes or N_CLASSES
    create_fn = dict(MODEL_FACTORIES)[model_name]
    model = create_fn(n_ch, n_feats, hparams, n_classes=nc)
    preds, epochs_ran = train_model(model, X_tr_n, Y_tr, X_te_n, Y_te, hparams, seed,
                                    groups_tr=groups_tr)
    f1 = f1_score(Y_te, preds, average='macro', zero_division=0)
    cm = confusion_matrix(Y_te, preds, labels=list(range(nc)))
    per_class_f1 = f1_score(Y_te, preds, average=None, labels=list(range(nc)), zero_division=0)
    return {'preds': preds, 'f1': f1, 'cm': cm, 'per_class_f1': per_class_f1,
            'epochs_ran': epochs_ran}


def evaluate_fold(X_tr, Y_tr, X_te, Y_te, n_ch, n_feats, hparams, seed,
                  n_classes=None, groups_tr=None):
    """Run every architecture in MODELS on one fold; return metrics per model."""
    X_tr_n, X_te_n = normalize_features(X_tr, X_te, n_ch, n_feats)
    return {
        model_name: evaluate_one_model(model_name, X_tr_n, Y_tr, X_te_n, Y_te,
                                       n_ch, n_feats, hparams, seed, n_classes,
                                       groups_tr=groups_tr)
        for model_name, _ in MODEL_FACTORIES
    }


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
    """Paired t-test + Cohen's d_z between two paired score vectors.

    The sample unit is whatever the caller pairs on -- for every comparison
    reported in the paper this is the SUBJECT (one macro-F1 per subject,
    itself already averaged over that subject's folds). n is returned
    explicitly so the reported effect sizes can be read unambiguously.
    """
    scores_a, scores_b = np.asarray(scores_a, float), np.asarray(scores_b, float)
    t_stat, p_raw = scipy_stats.ttest_rel(scores_a, scores_b)
    p_corrected = min(p_raw * n_comparisons, 1.0)
    diff = scores_a - scores_b
    n = len(diff)
    # Cohen's d_z for paired samples (sample SD, ddof=1)
    cohens_d = diff.mean() / (diff.std(ddof=1) + 1e-12)
    # Parametric 95% CI of the mean paired difference
    sem = diff.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    tcrit = scipy_stats.t.ppf(0.975, n - 1) if n > 1 else 0.0
    return {
        'comparison': f'{name_a} vs {name_b}',
        'n_pairs': int(n),
        'unit': 'subject',
        'mean_diff': float(diff.mean()),
        'diff_ci_95': [float(diff.mean() - tcrit * sem), float(diff.mean() + tcrit * sem)],
        't_stat': float(t_stat),
        'p_raw': float(p_raw),
        'p_corrected': float(p_corrected),
        'cohens_d': float(cohens_d),
        'significant_bonferroni': bool(p_corrected < 0.05),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def assert_sole_instance():
    """Refuse to start if another instance of this script is already running.

    Two concurrent processes share the same checkpoint directory and
    progress.json, with no locking: they can each independently satisfy a
    given checkpoint key and unconditionally overwrite one another's result
    for it, and progress.json reflects whichever process wrote most recently
    rather than the process actually being watched. This has already happened
    once during this project (a stale pre-fix process kept running silently
    after a corrected version was launched, contaminating the checkpoint
    directory with results from the old code for ~11 minutes) -- so this is a
    real failure mode, not a hypothetical one.
    """
    my_pid = os.getpid()
    my_name = os.path.basename(__file__)  # 'run_evaluation_unified.py'
    others = []
    for p in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        if p.info['pid'] == my_pid:
            continue
        name = (p.info['name'] or '').lower()
        if 'python' not in name:
            continue        # only python interpreters can be a real instance
        try:
            cmdline = p.info['cmdline'] or []
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        # Match only an actual script argument, e.g. ".../run_evaluation_unified.py"
        # as its own argv element -- not a substring anywhere on the command line,
        # which false-positives on any command that merely mentions the filename
        # (including this function's own test/diagnostic invocations).
        if any(isinstance(a, str) and os.path.basename(a) == my_name for a in cmdline):
            others.append(p.info['pid'])
    if others:
        age_min = [round((time.time() - psutil.Process(pid).create_time()) / 60, 1)
                  for pid in others if psutil.pid_exists(pid)]
        raise RuntimeError(
            f'Another instance of run_evaluation_unified.py is already running '
            f'(PID(s): {others}, running for {age_min} min). Refusing to start a '
            f'second instance: both would write to the same checkpoint directory '
            f'and progress.json with no coordination, silently corrupting results '
            f'or reporting stale progress. Stop the other process first if it is '
            f'stale, or wait for it to finish.')


if __name__ == '__main__':
    assert_sole_instance()
    SCRIPT_START = time.time()
    PHASE_TIMES = {}  # wall-clock seconds per evaluation phase (for cost reporting)

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
    _lgat = create_light_gat(DEAP_N_CH, 26, hparams)
    print(f'Parameter counts (DEAP, 32ch):')
    print(f'  GAT (3-layer): {count_parameters(_gat):,}')
    print(f'  MLP: {count_parameters(_mlp):,}')
    del _gat, _mlp, _lgat

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

    n_sub = len(deap_data)
    progress_init([
        ('deap_t0', 5 * len(MODELS)), ('deap_t1', n_sub), ('deap_t2', n_sub), ('deap_t3', n_sub * len(MODELS)),
        ('openbci', 25), ('xai', 1),
        ('bin_t0', 2 * 5 * len(MODELS)), ('bin_t1', 2 * n_sub),
        ('bin_t2', 2 * n_sub),
        ('bin_t3', 2 * n_sub * len(MODELS) if RUN_BINARY_T3 else 0),
    ])
    n_ckpt = len(glob.glob(os.path.join(CKPT_DIR, '*.json')))
    log(f'  Work units planned: {_PROGRESS["units_total"]} '
        f'| checkpoints already on disk: {n_ckpt}')
    if n_ckpt:
        log(f'  Resuming -- completed units will be loaded from {CKPT_DIR}')

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

    tier0_scores = {m: [] for m in MODELS}
    tier0_cms = {m: [] for m in MODELS}

    # 5 random 80/20 splits for stability. Checkpointed per (split, model) --
    # not per split -- because a single GAT training here can run 15-20+ min
    # on the pooled ~60K-window corpus, and per-split granularity left too
    # long a blind spot between checkpoints during exactly the slowest tier.
    for split_i in range(5):
        tr_idx, te_idx = train_test_split(
            np.arange(len(all_X_deap)), test_size=0.2,
            stratify=all_Y_deap, random_state=SEED + split_i)
        X_tr_n, X_te_n = normalize_features(
            all_X_deap[tr_idx], all_X_deap[te_idx], DEAP_N_CH, n_feats)
        Y_tr, Y_te = all_Y_deap[tr_idx], all_Y_deap[te_idx]

        split_out = {}
        for model_name, _ in MODEL_FACTORIES:
            def _run(model_name=model_name, split_i=split_i,
                     X_tr_n=X_tr_n, Y_tr=Y_tr, X_te_n=X_te_n, Y_te=Y_te):
                r = evaluate_one_model(model_name, X_tr_n, Y_tr, X_te_n, Y_te,
                                       DEAP_N_CH, n_feats, hparams, SEED + split_i)
                return {'f1': r['f1'], 'cm': r['cm'].tolist(), 'epochs_ran': r['epochs_ran']}

            split_out[model_name] = unit(
                'deap_t0', f'deap4_t0_split{split_i}_{model_name}', _run,
                'DEAP 4-class Tier0')

        for m in MODELS:
            tier0_scores[m].append(split_out[m]['f1'])
            tier0_cms[m].append(split_out[m]['cm'])
        log('  Split %d: ' % (split_i + 1)
            + ', '.join(f'{m}={split_out[m]["f1"]:.4f}' for m in MODELS))

    all_results['DEAP']['Tier0'] = {
        m: {'f1_scores': tier0_scores[m],
            'f1_mean': float(np.mean(tier0_scores[m])),
            'f1_std': float(np.std(tier0_scores[m])),
            'confusion_matrices': tier0_cms[m]}
        for m in MODELS
    }
    PHASE_TIMES['DEAP_4class_Tier0'] = time.time() - t0
    print(f'\n-- TIER 0 SUMMARY -- [{time.time()-t0:.0f}s]')
    for m in MODELS:
        print(f'  {m}: F1 = {np.mean(tier0_scores[m]):.4f} +/- {np.std(tier0_scores[m]):.4f}')
    print()

    # ═══════════════════════════════════════════════════════════════════════════
    # DEAP TIER 1: WITHIN-SUBJECT CV (StratifiedKFold, windows from same trial can leak)
    # ═══════════════════════════════════════════════════════════════════════════

    print('=' * 70)
    print('DEAP TIER 1: WITHIN-SUBJECT CV (StratifiedKFold)')
    print('=' * 70)
    t0 = time.time()

    tier1_per_subject = {m: [] for m in MODELS}

    for sub_id, sub_data in list(deap_data.items()):
        X, Y, trials = sub_data['X'], sub_data['Y'], sub_data['trials']

        if len(np.unique(Y)) < 2:
            continue

        def _run(X=X, Y=Y):
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
            sub_scores = {m: [] for m in MODELS}
            for fold, (tr_idx, te_idx) in enumerate(skf.split(X, Y)):
                res = evaluate_fold(X[tr_idx], Y[tr_idx], X[te_idx], Y[te_idx],
                                   DEAP_N_CH, n_feats, hparams, SEED + fold)
                for m in MODELS:
                    sub_scores[m].append(res[m]['f1'])
            return {m: float(np.mean(sub_scores[m])) for m in MODELS}

        out = unit('deap_t1', f'deap4_t1_s{sub_id}', _run, 'DEAP 4-class Tier1')
        for m in MODELS:
            tier1_per_subject[m].append(out[m])

    all_results['DEAP']['Tier1'] = {
        m: {'per_subject': tier1_per_subject[m],
            'f1_mean': float(np.mean(tier1_per_subject[m])),
            'f1_std': float(np.std(tier1_per_subject[m])),
            'ci_95': list(bootstrap_ci(tier1_per_subject[m]))}
        for m in MODELS
    }
    PHASE_TIMES['DEAP_4class_Tier1'] = time.time() - t0
    print(f'\n-- TIER 1 SUMMARY -- [{time.time()-t0:.0f}s]')
    for m in MODELS:
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

    tier2_per_subject = {m: [] for m in MODELS}

    for sub_id, sub_data in list(deap_data.items()):
        X, Y, trials = sub_data['X'], sub_data['Y'], sub_data['trials']

        n_unique_trials = len(np.unique(trials))
        actual_folds = min(N_FOLDS, n_unique_trials)
        if len(np.unique(Y)) < 2 or actual_folds < 2:
            continue

        def _run(X=X, Y=Y, trials=trials, actual_folds=actual_folds):
            sgkf = StratifiedGroupKFold(n_splits=actual_folds, shuffle=True,
                                        random_state=SEED)
            sub_scores = {m: [] for m in MODELS}
            for fold, (tr_idx, te_idx) in enumerate(sgkf.split(X, Y, groups=trials)):
                res = evaluate_fold(X[tr_idx], Y[tr_idx], X[te_idx], Y[te_idx],
                                   DEAP_N_CH, n_feats, hparams, SEED + fold,
                                   groups_tr=trials[tr_idx])
                for m in MODELS:
                    sub_scores[m].append(res[m]['f1'])
            return {m: float(np.mean(sub_scores[m])) for m in MODELS}

        out = unit('deap_t2', f'deap4_t2_s{sub_id}', _run, 'DEAP 4-class Tier2')
        for m in MODELS:
            tier2_per_subject[m].append(out[m])

    all_results['DEAP']['Tier2'] = {
        m: {'per_subject': tier2_per_subject[m],
            'f1_mean': float(np.mean(tier2_per_subject[m])),
            'f1_std': float(np.std(tier2_per_subject[m])),
            'ci_95': list(bootstrap_ci(tier2_per_subject[m]))}
        for m in MODELS
    }
    PHASE_TIMES['DEAP_4class_Tier2'] = time.time() - t0
    print(f'\n-- TIER 2 SUMMARY -- [{time.time()-t0:.0f}s]')
    for m in MODELS:
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

    tier3_per_subject = {m: [] for m in MODELS}

    # Checkpointed per (subject, model): the N-1 pooled training set here is
    # as large as Tier0's, so the same per-model granularity applies.
    for test_sub in sorted(deap_data.keys()):
        test_mask = all_subs_arr == int(test_sub)
        train_mask = ~test_mask
        X_tr_n, X_te_n = normalize_features(
            all_X_deap[train_mask], all_X_deap[test_mask], DEAP_N_CH, n_feats)
        Y_tr, Y_te = all_Y_deap[train_mask], all_Y_deap[test_mask]
        # Validation holds out whole training subjects, so early stopping is
        # judged on the same cross-subject transfer this tier measures.
        G_tr = all_subs_arr[train_mask]

        sub_out = {}
        for model_name, _ in MODEL_FACTORIES:
            def _run(model_name=model_name, X_tr_n=X_tr_n, Y_tr=Y_tr,
                     X_te_n=X_te_n, Y_te=Y_te, G_tr=G_tr):
                r = evaluate_one_model(model_name, X_tr_n, Y_tr, X_te_n, Y_te,
                                       DEAP_N_CH, n_feats, hparams, SEED,
                                       groups_tr=G_tr)
                return {'f1': r['f1'], 'epochs_ran': r['epochs_ran']}

            sub_out[model_name] = unit(
                'deap_t3', f'deap4_t3_s{test_sub}_{model_name}', _run,
                'DEAP 4-class Tier3 LOSO')

        for m in MODELS:
            tier3_per_subject[m].append(sub_out[m]['f1'])
        log(f'  s{test_sub}: '
            + ', '.join(f'{m}={sub_out[m]["f1"]:.4f}' for m in MODELS))

    all_results['DEAP']['Tier3'] = {
        m: {'per_subject': tier3_per_subject[m],
            'f1_mean': float(np.mean(tier3_per_subject[m])),
            'f1_std': float(np.std(tier3_per_subject[m])),
            'ci_95': list(bootstrap_ci(tier3_per_subject[m]))}
        for m in MODELS
    }
    PHASE_TIMES['DEAP_4class_Tier3'] = time.time() - t0
    print(f'\n-- TIER 3 SUMMARY (LOSO) -- [{time.time()-t0:.0f}s]')
    for m in MODELS:
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
    ob_tier0 = {m: [] for m in MODELS}
    for split_i in range(5):
        def _run(split_i=split_i):
            tr_idx, te_idx = train_test_split(
                np.arange(len(openbci_X)), test_size=0.2,
                stratify=openbci_Y, random_state=SEED + split_i)
            res = evaluate_fold(openbci_X[tr_idx], openbci_Y[tr_idx],
                               openbci_X[te_idx], openbci_Y[te_idx],
                               OPENBCI_N_CH, ob_n_feats, hparams, SEED + split_i)
            return {m: float(res[m]['f1']) for m in MODELS}

        out = unit('openbci', f'ob_t0_split{split_i}', _run, 'OpenBCI Tier0')
        for m in MODELS:
            ob_tier0[m].append(out[m])
        log('  Split %d: ' % (split_i + 1)
            + ', '.join(f'{m}={out[m]:.4f}' for m in MODELS))

    # Tier 1: StratifiedKFold (leaky)
    print('\n-- OpenBCI Tier 1: StratifiedKFold --')
    skf_ob = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    ob_tier1 = {m: [] for m in MODELS}
    for fold, (tr_idx, te_idx) in enumerate(skf_ob.split(openbci_X, openbci_Y)):
        def _run(tr_idx=tr_idx, te_idx=te_idx, fold=fold):
            res = evaluate_fold(openbci_X[tr_idx], openbci_Y[tr_idx],
                               openbci_X[te_idx], openbci_Y[te_idx],
                               OPENBCI_N_CH, ob_n_feats, hparams, SEED + fold)
            return {m: float(res[m]['f1']) for m in MODELS}

        out = unit('openbci', f'ob_t1_fold{fold}', _run, 'OpenBCI Tier1')
        for m in MODELS:
            ob_tier1[m].append(out[m])
        log('  Fold %2d: ' % (fold + 1)
            + ', '.join(f'{m}={out[m]:.4f}' for m in MODELS))

    # Tier 2: Trial-aware (StratifiedGroupKFold)
    print('\n-- OpenBCI Tier 2: Trial-Aware --')
    sgkf_ob = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    ob_tier2 = {m: [] for m in MODELS}
    for fold, (tr_idx, te_idx) in enumerate(
        sgkf_ob.split(openbci_X, openbci_Y, groups=openbci_groups)):
        def _run(tr_idx=tr_idx, te_idx=te_idx, fold=fold):
            res = evaluate_fold(openbci_X[tr_idx], openbci_Y[tr_idx],
                               openbci_X[te_idx], openbci_Y[te_idx],
                               OPENBCI_N_CH, ob_n_feats, hparams, SEED + fold,
                               groups_tr=openbci_groups[tr_idx])
            return {m: float(res[m]['f1']) for m in MODELS}

        out = unit('openbci', f'ob_t2_fold{fold}', _run, 'OpenBCI Tier2')
        for m in MODELS:
            ob_tier2[m].append(out[m])
        log('  Fold %2d: ' % (fold + 1)
            + ', '.join(f'{m}={out[m]:.4f}' for m in MODELS))

    all_results['OpenBCI'] = {
        'Tier0': {m: {'f1_scores': ob_tier0[m],
                      'f1_mean': float(np.mean(ob_tier0[m])),
                      'f1_std': float(np.std(ob_tier0[m]))}
                  for m in MODELS},
        'Tier1': {m: {'f1_scores': ob_tier1[m],
                      'f1_mean': float(np.mean(ob_tier1[m])),
                      'f1_std': float(np.std(ob_tier1[m]))}
                  for m in MODELS},
        'Tier2': {m: {'f1_scores': ob_tier2[m],
                      'f1_mean': float(np.mean(ob_tier2[m])),
                      'f1_std': float(np.std(ob_tier2[m]))}
                  for m in MODELS},
    }

    PHASE_TIMES['OpenBCI_all_tiers'] = time.time() - t0
    print(f'\n-- OpenBCI SUMMARY -- [{time.time()-t0:.0f}s]')
    for tier_name, tier_data in [('Tier0', ob_tier0), ('Tier1', ob_tier1), ('Tier2', ob_tier2)]:
        for m in MODELS:
            print(f'  {tier_name} {m}: F1 = {np.mean(tier_data[m]):.4f} +/- {np.std(tier_data[m]):.4f}')
    print()

    # ═══════════════════════════════════════════════════════════════════════════
    # STATISTICAL COMPARISONS (Bonferroni-corrected)
    # ═══════════════════════════════════════════════════════════════════════════

    print('=' * 70)
    print('STATISTICAL COMPARISONS (paired t-tests, Bonferroni-corrected)')
    print('=' * 70)

    def _sig(stat):
        p = stat['p_corrected']
        return '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))

    print(f'  Sample unit for every test below: SUBJECT '
          f'(n={len(tier1_per_subject["GAT"])} paired observations).')
    print('  Each subject contributes one macro-F1 per tier, already averaged over that '
          'subject\'s folds.\n')

    # Tier-to-tier degradation, reported for BOTH architectures
    tier_pairs = [('Tier1', 'Tier2'), ('Tier1', 'Tier3'), ('Tier2', 'Tier3')]
    n_comp = len(tier_pairs)
    stats_results = {}

    for model_name in MODELS:
        per_tier = {
            'Tier1': tier1_per_subject[model_name],
            'Tier2': tier2_per_subject[model_name],
            'Tier3': tier3_per_subject[model_name],
        }
        stats_results[model_name] = []
        print(f'  Tier degradation [{model_name}]:')
        for ta, tb in tier_pairs:
            stat = tier_comparison_stats(per_tier[ta], per_tier[tb], ta, tb, n_comp)
            stats_results[model_name].append(stat)
            print(f'    {ta} vs {tb}: dF1={stat["mean_diff"]:+.4f} '
                  f'95% CI [{stat["diff_ci_95"][0]:+.4f}, {stat["diff_ci_95"][1]:+.4f}], '
                  f't={stat["t_stat"]:.3f}, p_corr={stat["p_corrected"]:.2e}, '
                  f'd_z={stat["cohens_d"]:.2f} [{_sig(stat)}] (n={stat["n_pairs"]})')

    # Architecture comparison within each tier (the claim reviewers questioned)
    print('\n  GAT vs MLP (paired by subject, within tier):')
    model_comparisons = []
    for tier_name, mlp_scores, gat_scores in [
        ('Tier1', tier1_per_subject['MLP'], tier1_per_subject['GAT']),
        ('Tier2', tier2_per_subject['MLP'], tier2_per_subject['GAT']),
        ('Tier3', tier3_per_subject['MLP'], tier3_per_subject['GAT']),
    ]:
        stat = tier_comparison_stats(gat_scores, mlp_scores,
                                     f'GAT_{tier_name}', f'MLP_{tier_name}', 3)
        model_comparisons.append(stat)
        print(f'    {tier_name}: GAT-MLP = {stat["mean_diff"]:+.4f} '
              f'95% CI [{stat["diff_ci_95"][0]:+.4f}, {stat["diff_ci_95"][1]:+.4f}], '
              f't={stat["t_stat"]:.3f}, p_corr={stat["p_corrected"]:.4f}, '
              f'd_z={stat["cohens_d"]:.2f} [{_sig(stat)}] (n={stat["n_pairs"]})')

    all_results['statistics'] = {
        'sample_unit': 'subject (one macro-F1 per subject, averaged over that subject\'s folds)',
        'n_subjects': len(tier1_per_subject['GAT']),
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

    # Train a GAT on all DEAP data (80/20 split) for attention extraction.
    # Use a dedicated RNG: the global NumPy stream has been consumed an
    # unpredictable number of times by bootstrap_ci above, so np.random here
    # would make this split depend on how many tiers/models ran before it.
    n = len(all_X_deap)
    idx = np.random.RandomState(SEED).permutation(n)
    split = int(0.8 * n)
    X_tr_xai, X_te_xai = all_X_deap[idx[:split]], all_X_deap[idx[split:]]
    Y_tr_xai, Y_te_xai = all_Y_deap[idx[:split]], all_Y_deap[idx[split:]]

    X_tr_xai_n, X_te_xai_n = normalize_features(X_tr_xai, X_te_xai, DEAP_N_CH, n_feats)

    def _run_xai():
        log('Training GAT for attention extraction...')
        xai_model = create_gat(DEAP_N_CH, n_feats, hparams)
        _, _ = train_model(xai_model, X_tr_xai_n, Y_tr_xai, X_te_xai_n, Y_te_xai,
                           hparams, SEED)
        n_xai_samples = min(1000, len(X_te_xai_n))
        x_input = torch.from_numpy(X_te_xai_n[:n_xai_samples]).float().to(device)
        xai_model.eval()
        with torch.no_grad():
            _, attns = xai_model(x_input, return_attn=True)
        final = [aw.mean(dim=(0, 1)).cpu().numpy() for aw in attns][-1]
        return {'attention_matrix': final.tolist(),
                'channel_importance': final.sum(axis=0).tolist()}

    xai_out = unit('xai', 'deap4_xai_attention', _run_xai, 'XAI attention')
    final_attn = np.array(xai_out['attention_matrix'])
    ch_importance = np.array(xai_out['channel_importance'])

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
        bin_t0 = {m: [] for m in MODELS}
        for split_i in range(5):
            tr_idx, te_idx = train_test_split(
                np.arange(len(all_X_bin)), test_size=0.2,
                stratify=all_Y_bin, random_state=SEED + split_i)
            X_tr_n, X_te_n = normalize_features(
                all_X_bin[tr_idx], all_X_bin[te_idx], DEAP_N_CH, n_feats)
            Y_tr, Y_te = all_Y_bin[tr_idx], all_Y_bin[te_idx]

            split_out = {}
            for model_name, _ in MODEL_FACTORIES:
                def _run(model_name=model_name, split_i=split_i,
                         X_tr_n=X_tr_n, Y_tr=Y_tr, X_te_n=X_te_n, Y_te=Y_te):
                    r = evaluate_one_model(model_name, X_tr_n, Y_tr, X_te_n, Y_te,
                                           DEAP_N_CH, n_feats, hparams,
                                           SEED + split_i, n_classes=2)
                    return {'f1': r['f1'], 'epochs_ran': r['epochs_ran']}

                split_out[model_name] = unit(
                    'bin_t0', f'bin_{dim_name}_t0_split{split_i}_{model_name}',
                    _run, f'Binary {dim_name} Tier0')

            for m in MODELS:
                bin_t0[m].append(split_out[m]['f1'])
        for m in MODELS:
            print(f'    {m}: {np.mean(bin_t0[m]):.4f} +/- {np.std(bin_t0[m]):.4f}')
        dim_results['Tier0'] = {
            m: {'f1_scores': bin_t0[m], 'f1_mean': float(np.mean(bin_t0[m])),
                'f1_std': float(np.std(bin_t0[m]))}
            for m in MODELS
        }

        # Binary Tier 1: Within-subject StratifiedKFold
        print(f'  Tier 1 (within-subject, leaky):')
        bin_t1 = {m: [] for m in MODELS}
        for sub_id, sub_data in deap_binary.items():
            X, Y = sub_data['X'], sub_data[y_key]
            if len(np.unique(Y)) < 2:
                continue

            def _run(X=X, Y=Y):
                skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
                sub_scores = {m: [] for m in MODELS}
                for fold, (tr_idx, te_idx) in enumerate(skf.split(X, Y)):
                    res = evaluate_fold(X[tr_idx], Y[tr_idx], X[te_idx], Y[te_idx],
                                       DEAP_N_CH, n_feats, hparams, SEED + fold,
                                       n_classes=2)
                    for m in MODELS:
                        sub_scores[m].append(res[m]['f1'])
                return {m: float(np.mean(sub_scores[m])) for m in MODELS}

            out = unit('bin_t1', f'bin_{dim_name}_t1_s{sub_id}', _run,
                       f'Binary {dim_name} Tier1')
            for m in MODELS:
                bin_t1[m].append(out[m])
        for m in MODELS:
            print(f'    {m}: {np.mean(bin_t1[m]):.4f} +/- {np.std(bin_t1[m]):.4f}')
        dim_results['Tier1'] = {
            m: {'per_subject': bin_t1[m], 'f1_mean': float(np.mean(bin_t1[m])),
                'f1_std': float(np.std(bin_t1[m])), 'ci_95': list(bootstrap_ci(bin_t1[m]))}
            for m in MODELS
        }

        # Binary Tier 2: Trial-aware within-subject
        print(f'  Tier 2 (trial-aware):')
        bin_t2 = {m: [] for m in MODELS}
        for sub_id, sub_data in deap_binary.items():
            X, Y, trials = sub_data['X'], sub_data[y_key], sub_data['trials']
            n_unique = len(np.unique(trials))
            actual_folds = min(N_FOLDS, n_unique)
            if len(np.unique(Y)) < 2 or actual_folds < 2:
                continue

            def _run(X=X, Y=Y, trials=trials, actual_folds=actual_folds):
                sgkf = StratifiedGroupKFold(n_splits=actual_folds, shuffle=True,
                                            random_state=SEED)
                sub_scores = {m: [] for m in MODELS}
                for fold, (tr_idx, te_idx) in enumerate(sgkf.split(X, Y, groups=trials)):
                    res = evaluate_fold(X[tr_idx], Y[tr_idx], X[te_idx], Y[te_idx],
                                       DEAP_N_CH, n_feats, hparams, SEED + fold,
                                       n_classes=2, groups_tr=trials[tr_idx])
                    for m in MODELS:
                        sub_scores[m].append(res[m]['f1'])
                return {m: float(np.mean(sub_scores[m])) for m in MODELS}

            out = unit('bin_t2', f'bin_{dim_name}_t2_s{sub_id}', _run,
                       f'Binary {dim_name} Tier2')
            for m in MODELS:
                bin_t2[m].append(out[m])
        for m in MODELS:
            print(f'    {m}: {np.mean(bin_t2[m]):.4f} +/- {np.std(bin_t2[m]):.4f}')
        dim_results['Tier2'] = {
            m: {'per_subject': bin_t2[m], 'f1_mean': float(np.mean(bin_t2[m])),
                'f1_std': float(np.std(bin_t2[m])), 'ci_95': list(bootstrap_ci(bin_t2[m]))}
            for m in MODELS
        }

        # Binary Tier 3 (LOSO) is deliberately not run. It was 128 full-corpus
        # trainings -- roughly half the total runtime of the entire sweep -- for
        # the least load-bearing result in the paper: cross-subject
        # generalization is already demonstrated by the 4-class Tier 3, and the
        # binary analysis exists to give a trial-aware number comparable with
        # published DEAP benchmarks, which is Tier 2. Binary results are
        # therefore reported for Tiers 0-2 only. Set RUN_BINARY_T3 = True to
        # restore it.
        if RUN_BINARY_T3:
            print(f'  Tier 3 (LOSO):')
            all_subs_bin = np.concatenate([
                np.full(len(d['X']), int(sid)) for sid, d in deap_binary.items()
            ])
            bin_t3 = {m: [] for m in MODELS}
            for test_sub in sorted(deap_binary.keys()):
                test_mask = all_subs_bin == int(test_sub)
                train_mask = ~test_mask
                X_tr_n, X_te_n = normalize_features(
                    all_X_bin[train_mask], all_X_bin[test_mask], DEAP_N_CH, n_feats)
                Y_tr, Y_te = all_Y_bin[train_mask], all_Y_bin[test_mask]

                sub_out = {}
                for model_name, _ in MODEL_FACTORIES:
                    def _run(model_name=model_name, X_tr_n=X_tr_n, Y_tr=Y_tr,
                             X_te_n=X_te_n, Y_te=Y_te):
                        r = evaluate_one_model(model_name, X_tr_n, Y_tr, X_te_n, Y_te,
                                               DEAP_N_CH, n_feats, hparams, SEED,
                                               n_classes=2)
                        return {'f1': r['f1'], 'epochs_ran': r['epochs_ran']}

                    sub_out[model_name] = unit(
                        'bin_t3', f'bin_{dim_name}_t3_s{test_sub}_{model_name}',
                        _run, f'Binary {dim_name} Tier3 LOSO')

                for m in MODELS:
                    bin_t3[m].append(sub_out[m]['f1'])
            for m in MODELS:
                print(f'    {m}: {np.mean(bin_t3[m]):.4f} +/- {np.std(bin_t3[m]):.4f}')
            dim_results['Tier3'] = {
                m: {'per_subject': bin_t3[m], 'f1_mean': float(np.mean(bin_t3[m])),
                    'f1_std': float(np.std(bin_t3[m])),
                    'ci_95': list(bootstrap_ci(bin_t3[m]))}
                for m in MODELS
            }
        else:
            print('  Tier 3 (LOSO): skipped (RUN_BINARY_T3=False)')

        all_results['DEAP_binary'][dim_name] = dim_results

    PHASE_TIMES['DEAP_binary_all_tiers'] = time.time() - t0
    print(f'\n-- BINARY SUMMARY -- [{time.time()-t0:.0f}s]')
    for dim in ('arousal', 'valence'):
        print(f'  {dim.upper()}:')
        for tier in ('Tier0', 'Tier1', 'Tier2', 'Tier3'):
            td = all_results['DEAP_binary'][dim].get(tier)
            if td is None:
                continue
            print(f'    {tier}: MLP={td["MLP"]["f1_mean"]:.4f}, GAT={td["GAT"]["f1_mean"]:.4f}')
    print()

    # ═══════════════════════════════════════════════════════════════════════════
    # BINARY: ARCHITECTURE COMPARISON (paired by subject, per tier)
    # ═══════════════════════════════════════════════════════════════════════════

    print('=' * 70)
    print('BINARY STATISTICS (GAT vs MLP, paired by subject)')
    print('=' * 70)
    binary_model_comparisons = {}
    for dim in ('arousal', 'valence'):
        binary_model_comparisons[dim] = []
        print(f'  {dim.upper()}:')
        for tier_name in ('Tier1', 'Tier2', 'Tier3'):
            td = all_results['DEAP_binary'][dim].get(tier_name)
            if td is None or 'per_subject' not in td['MLP']:
                continue
            stat = tier_comparison_stats(td['GAT']['per_subject'], td['MLP']['per_subject'],
                                         f'GAT_{tier_name}', f'MLP_{tier_name}', 6)
            binary_model_comparisons[dim].append(stat)
            print(f'    {tier_name}: GAT-MLP = {stat["mean_diff"]:+.4f} '
                  f'95% CI [{stat["diff_ci_95"][0]:+.4f}, {stat["diff_ci_95"][1]:+.4f}], '
                  f'p_corr={stat["p_corrected"]:.4f}, d_z={stat["cohens_d"]:.2f} '
                  f'[{_sig(stat)}] (n={stat["n_pairs"]})')
    all_results['statistics']['binary_model_comparisons'] = binary_model_comparisons
    print()

    # ═══════════════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════════════════

    PHASE_TIMES['TOTAL'] = time.time() - SCRIPT_START
    all_results['compute_cost'] = {
        'device': str(device),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'phase_seconds': {k: round(v, 1) for k, v in PHASE_TIMES.items()},
        'total_hours': round(PHASE_TIMES['TOTAL'] / 3600.0, 2),
        'note': ('Wall-clock for one full four-tier sweep over 3 architectures, '
                 '4-class + binary arousal + binary valence, on the full DEAP corpus. '
                 'Feature extraction is cached and excluded.'),
    }
    all_results['_meta'] = {
        'source': 'run_evaluation_unified.py (deterministic, seed=42)',
        'metric': 'Macro F1',
        'deap_subjects': len(deap_data),
        'deap_windows': int(total_windows),
        'openbci_windows': int(len(openbci_X)),
        'openbci_trials': int(len(np.unique(openbci_groups))),
        'folds': N_FOLDS,
        'hparams': hparams,
    }

    out_path = os.path.join(OUT_DIR, 'all_results.json')

    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=_np_safe)
    print(f'Results saved to: {out_path}')

    # ═══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════

    print(f'\n{"="*70}')
    print('FINAL RESULTS')
    print('=' * 70)
    print(f'\n  DEAP ({len(deap_data)} subjects, 4-class quadrant, Macro F1)')
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

    print(f'\n  DEAP Binary ({len(deap_data)} subjects, per-subject median, Macro F1)')
    print(f'  {"─"*55}')
    for dim in ('arousal', 'valence'):
        print(f'  {dim.upper()}:')
        print(f'  {"Tier":<12} {"MLP":<20} {"GAT":<20}')
        for tier_name in ('Tier0', 'Tier1', 'Tier2', 'Tier3'):
            td = all_results['DEAP_binary'][dim].get(tier_name)
            if td is None:
                continue
            mlp_str = f'{td["MLP"]["f1_mean"]:.4f} +/- {td["MLP"].get("f1_std", 0):.4f}'
            gat_str = f'{td["GAT"]["f1_mean"]:.4f} +/- {td["GAT"].get("f1_std", 0):.4f}'
            print(f'  {tier_name:<12} {mlp_str:<20} {gat_str:<20}')

    print(f'\n{"="*70}')
    print(f'DONE — all results in: {OUT_DIR}')
    print(f'{"="*70}')
