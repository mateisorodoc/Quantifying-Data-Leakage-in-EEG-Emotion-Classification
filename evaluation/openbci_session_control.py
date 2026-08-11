"""
═══════════════════════════════════════════════════════════════════════════════
OpenBCI SESSION-CONFOUND CONTROL EXPERIMENTS
═══════════════════════════════════════════════════════════════════════════════
The custom OpenBCI corpus was recorded one emotion class per session, each
session on a different calendar day, with the 25 tracks of that class played
back-to-back in fixed order. Emotion label is therefore perfectly collinear
with recording session. These probes quantify how much *non-emotional*
session structure a classifier can exploit under exactly the same trial-aware
protocol used for the headline result.

PROBE A  Recording-order decoding WITHIN one session.
         Early trials (1-12) vs late trials (14-25) of the SAME session --
         same emotion label, same stimulus category, same day. Any accuracy
         above chance is non-affective drift (impedance, electrode settling,
         posture, fatigue), and is a LOWER BOUND on the nuisance signal
         available to the 4-class model, which additionally gets to exploit
         BETWEEN-session differences.

PROBE B  Sham-label negative control.
         Same protocol, trial labels randomly permuted within a session.
         Must sit at chance -- confirms the trial-aware splitter itself is
         not leaky and that Probe A is not a protocol artifact.

PROBE C  Pairwise class discriminability vs. days between sessions.
         Six one-vs-one problems. If affect drives the result, difficulty
         should track distance in the arousal-valence plane. If session
         identity drives it, difficulty should track the calendar gap.
         The design dissociates the two: e.g. happy/calm differ in arousal
         but were recorded 2 days apart, while calm/sad differ only in
         valence but were recorded 23 days apart.

Usage:  python openbci_session_control.py
Output: outputs/openbci_session_control.json
═══════════════════════════════════════════════════════════════════════════════
"""

import os, sys, glob, json, time, datetime
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
from scipy import stats as scipy_stats
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score

from run_evaluation_unified import (
    OPENBCI_N_CH, OUT_DIR, SEED, DEFAULT_HPARAMS, OPENBCI_FEAT, OPENBCI_BASE,
    normalize_features, train_model, create_mlp, create_gat, device,
)

EMOTIONS = ['calm', 'happy', 'sad', 'stressed']       # label_map order in feature_extraction.py
EMO_LABEL = {e: i for i, e in enumerate(EMOTIONS)}    # matches Y in openbci_all.npz
RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                       'data', 'recordings_raw')

# Position in Russell's circumplex: (arousal, valence), each in {0, 1}
AV_COORDS = {'happy': (1, 1), 'stressed': (1, 0), 'calm': (0, 1), 'sad': (0, 0)}

N_PROBE_FOLDS = 5
N_SHAM_REPEATS = 5


# ─────────────────────────────────────────────────────────────────────────────
# Metadata reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_trial_metadata():
    """Rebuild trial_id -> (emotion, recording_index) exactly as the feature
    cache was built: emotions in label_map order, sorted(glob) within each.
    """
    meta, trial_id = {}, 0
    for emo in EMOTIONS:
        csv_dir = os.path.join(OPENBCI_BASE, f'recordings_{emo}_cleaned')
        for fp in sorted(glob.glob(os.path.join(csv_dir, 'clean_trial_*.csv'))):
            base = os.path.basename(fp)
            rec_idx = int(base.split('_')[2])          # clean_trial_<N>_<title>.csv
            meta[trial_id] = {'emotion': emo, 'rec_index': rec_idx, 'file': base}
            trial_id += 1
    return meta


def session_dates():
    """Median modification time of each class's raw recordings = session date."""
    dates = {}
    for emo in EMOTIONS:
        files = glob.glob(os.path.join(RAW_DIR, f'recordings_{emo}', '*.csv'))
        mtimes = sorted(os.path.getmtime(f) for f in files)
        dates[emo] = datetime.datetime.fromtimestamp(mtimes[len(mtimes) // 2])
    return dates


# ─────────────────────────────────────────────────────────────────────────────
# Shared trial-aware evaluation
# ─────────────────────────────────────────────────────────────────────────────

def trial_aware_binary(X, y, groups, hparams, models=('MLP', 'GAT'),
                       n_folds=N_PROBE_FOLDS, seed=SEED):
    """StratifiedGroupKFold by trial; returns {model: {'f1': [...], 'acc': [...]}}."""
    n_groups = len(np.unique(groups))
    folds = min(n_folds, n_groups, np.bincount(y).min())
    if folds < 2:
        return None
    sgkf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    out = {m: {'f1': [], 'acc': []} for m in models}
    factory = {'MLP': create_mlp, 'GAT': create_gat}

    for fold, (tr, te) in enumerate(sgkf.split(X, y, groups=groups)):
        X_tr, X_te = normalize_features(X[tr], X[te], OPENBCI_N_CH, X.shape[2])
        for m in models:
            model = factory[m](OPENBCI_N_CH, X.shape[2], hparams, n_classes=2)
            # groups_tr keeps the validation split (for early stopping) trial-aware,
            # matching the main sweep's fix -- early stopping never sees the test fold.
            preds, _ = train_model(model, X_tr, y[tr], X_te, y[te], hparams, seed + fold,
                                   groups_tr=groups[tr])
            out[m]['f1'].append(f1_score(y[te], preds, average='macro', zero_division=0))
            out[m]['acc'].append(accuracy_score(y[te], preds))
    return out


def summarize(scores):
    a = np.asarray(scores, float)
    return {'mean': float(a.mean()), 'std': float(a.std(ddof=1) if len(a) > 1 else 0.0),
            'folds': [float(v) for v in a]}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    t_start = time.time()

    hp_file = os.path.join(OUT_DIR, 'best_hparams.json')
    hparams = json.load(open(hp_file)) if os.path.exists(hp_file) else DEFAULT_HPARAMS

    d = np.load(OPENBCI_FEAT)
    X_all, Y_raw, G_all = d['X'], d['Y'], d['groups']
    meta = reconstruct_trial_metadata()
    dates = session_dates()

    assert len(meta) == len(np.unique(G_all)), 'trial metadata does not match feature cache'
    # sanity: reconstructed emotion per trial must match the cached labels
    for t, info in meta.items():
        cached = np.unique(Y_raw[G_all == t])
        assert cached.tolist() == [EMO_LABEL[info['emotion']]], f'trial {t} label mismatch'

    print('=' * 74)
    print('OpenBCI SESSION-CONFOUND CONTROL')
    print('=' * 74)
    print(f'Device: {device}   Windows: {len(X_all):,}   Trials: {len(meta)}')
    print('\nRecording sessions (median file timestamp of the raw recordings):')
    for emo in EMOTIONS:
        print(f'  {emo:<9} {dates[emo]:%Y-%m-%d %H:%M}  (25 tracks, fixed order)')
    print(flush=True)

    results = {
        'session_dates': {e: dates[e].isoformat() for e in EMOTIONS},
        'protocol_note': ('One emotion class per recording session, one session per '
                          'calendar day, 25 tracks played back-to-back in fixed order. '
                          'Class label is perfectly collinear with session identity.'),
    }

    rec_index = np.array([meta[int(t)]['rec_index'] for t in G_all])
    emo_of_win = np.array([meta[int(t)]['emotion'] for t in G_all])

    # ── PROBE A: within-session recording-order decoding ─────────────────────
    print('=' * 74)
    print('PROBE A: early (tracks 1-12) vs late (tracks 14-25) WITHIN one session')
    print('  Same emotion, same stimulus class, same day -> chance = 0.50')
    print('=' * 74, flush=True)

    probe_a = {}
    for emo in EMOTIONS:
        sel = (emo_of_win == emo) & (rec_index != 13)
        Xs, gs, ri = X_all[sel], G_all[sel], rec_index[sel]
        ys = (ri > 13).astype(np.int64)          # 0 = early half, 1 = late half
        res = trial_aware_binary(Xs, ys, gs, hparams)
        probe_a[emo] = {m: summarize(res[m]['f1']) for m in res}
        probe_a[emo]['n_trials'] = int(len(np.unique(gs)))
        probe_a[emo]['n_windows'] = int(sel.sum())
        print(f'  {emo:<9} MLP F1={probe_a[emo]["MLP"]["mean"]:.4f} '
              f'+/- {probe_a[emo]["MLP"]["std"]:.4f}   '
              f'GAT F1={probe_a[emo]["GAT"]["mean"]:.4f} '
              f'+/- {probe_a[emo]["GAT"]["std"]:.4f}', flush=True)

    for m in ('MLP', 'GAT'):
        vals = [probe_a[e][m]['mean'] for e in EMOTIONS]
        probe_a[f'{m}_across_sessions'] = {'mean': float(np.mean(vals)),
                                           'std': float(np.std(vals, ddof=1)),
                                           'per_session': vals}
        print(f'  -> {m} mean across the 4 sessions: {np.mean(vals):.4f}')
    results['probe_A_recording_order'] = probe_a
    print(flush=True)

    # ── PROBE B: sham-label negative control ─────────────────────────────────
    print('=' * 74)
    print('PROBE B: sham labels (trial labels permuted within session) -> must be ~0.50')
    print('=' * 74, flush=True)

    sham_scores = []
    rng = np.random.default_rng(SEED)
    for emo in EMOTIONS:
        sel = (emo_of_win == emo) & (rec_index != 13)
        Xs, gs = X_all[sel], G_all[sel]
        trials = np.unique(gs)
        for rep in range(N_SHAM_REPEATS):
            fake = rng.permutation(np.array([0] * (len(trials) // 2) +
                                            [1] * (len(trials) - len(trials) // 2)))
            lut = dict(zip(trials, fake))
            ys = np.array([lut[int(t)] for t in gs], dtype=np.int64)
            res = trial_aware_binary(Xs, ys, gs, hparams, models=('MLP',))
            sham_scores.append(float(np.mean(res['MLP']['f1'])))
        print(f'  {emo:<9} sham F1 over {N_SHAM_REPEATS} permutations: '
              f'{np.mean(sham_scores[-N_SHAM_REPEATS:]):.4f}', flush=True)

    results['probe_B_sham_labels'] = {
        'model': 'MLP', 'n_runs': len(sham_scores),
        'mean': float(np.mean(sham_scores)), 'std': float(np.std(sham_scores, ddof=1)),
        'scores': sham_scores,
    }
    print(f'  -> sham mean across all {len(sham_scores)} runs: {np.mean(sham_scores):.4f} '
          f'+/- {np.std(sham_scores, ddof=1):.4f}\n', flush=True)

    # ── PROBE C: pairwise discriminability vs. calendar gap ──────────────────
    print('=' * 74)
    print('PROBE C: pairwise class discriminability vs. days between sessions')
    print('=' * 74, flush=True)

    pairs = [(a, b) for i, a in enumerate(EMOTIONS) for b in EMOTIONS[i + 1:]]
    probe_c = []
    for a, b in pairs:
        sel = (emo_of_win == a) | (emo_of_win == b)
        Xs, gs = X_all[sel], G_all[sel]
        ys = (emo_of_win[sel] == b).astype(np.int64)
        res = trial_aware_binary(Xs, ys, gs, hparams)
        gap_days = abs((dates[a] - dates[b]).total_seconds()) / 86400.0
        av_dist = (abs(AV_COORDS[a][0] - AV_COORDS[b][0]) +
                   abs(AV_COORDS[a][1] - AV_COORDS[b][1]))
        entry = {
            'pair': f'{a} vs {b}',
            'gap_days': round(gap_days, 2),
            'av_distance': int(av_dist),
            'MLP': summarize(res['MLP']['f1']),
            'GAT': summarize(res['GAT']['f1']),
        }
        probe_c.append(entry)
        print(f'  {a:<9} vs {b:<9} gap={gap_days:5.1f}d  AVdist={av_dist}  '
              f'MLP F1={entry["MLP"]["mean"]:.4f}  GAT F1={entry["GAT"]["mean"]:.4f}',
              flush=True)

    gaps = [e['gap_days'] for e in probe_c]
    avds = [e['av_distance'] for e in probe_c]
    corr = {}
    for m in ('MLP', 'GAT'):
        f1s = [e[m]['mean'] for e in probe_c]
        r_gap, p_gap = scipy_stats.spearmanr(gaps, f1s)
        r_av, p_av = scipy_stats.spearmanr(avds, f1s)
        corr[m] = {'spearman_vs_gap_days': [float(r_gap), float(p_gap)],
                   'spearman_vs_av_distance': [float(r_av), float(p_av)]}
        print(f'  -> {m}: Spearman F1 vs calendar gap  rho={r_gap:+.3f} (p={p_gap:.3f})')
        print(f'     {m}: Spearman F1 vs A-V distance  rho={r_av:+.3f} (p={p_av:.3f})')
    print('  (n=6 pairs -- descriptive, not inferential)')

    results['probe_C_pairwise'] = {'pairs': probe_c, 'correlations': corr,
                                   'note': 'n=6 pairs; correlations are descriptive only'}

    results['runtime_seconds'] = round(time.time() - t_start, 1)
    out_fp = os.path.join(OUT_DIR, 'openbci_session_control.json')
    with open(out_fp, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {out_fp}   [{results["runtime_seconds"]:.0f}s]')
