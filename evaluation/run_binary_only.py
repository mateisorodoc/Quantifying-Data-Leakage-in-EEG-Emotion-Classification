"""
Run ONLY the binary (arousal + valence) classification section.
Loads existing 4-class results from all_results.json and appends binary results.
Run this after run_evaluation_unified.py has completed the 4-class evaluation.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Import everything from the unified script
from run_evaluation_unified import (
    load_deap_binary, evaluate_fold, bootstrap_ci, normalize_features,
    DEAP_N_CH, N_FOLDS, SEED, OUT_DIR, SUBJECTS,
    StratifiedKFold, StratifiedGroupKFold, train_test_split,
    np, json, time
)

import json as _json

def main():
    # Load hparams
    hparams_file = os.path.join(OUT_DIR, 'best_hparams.json')
    with open(hparams_file) as f:
        hparams = _json.load(f)

    # Load existing 4-class results
    out_path = os.path.join(OUT_DIR, 'all_results.json')
    if not os.path.exists(out_path):
        raise FileNotFoundError(f'Run run_evaluation_unified.py first to generate {out_path}')
    with open(out_path) as f:
        all_results = _json.load(f)
    print(f'Loaded existing results from {out_path}')

    n_feats = all_results['config']['n_feats']

    print('=' * 70)
    print('DEAP BINARY CLASSIFICATION (per-subject median, chance = 50%)')
    print('=' * 70)
    t0 = time.time()

    deap_binary = load_deap_binary()
    all_results['DEAP_binary'] = {}

    for dim_name, y_key in [('arousal', 'Y_aro'), ('valence', 'Y_val')]:
        print(f'\n  --- {dim_name.upper()} ---')

        all_X_bin = np.concatenate([d['X'] for d in deap_binary.values()])
        all_Y_bin = np.concatenate([d[y_key] for d in deap_binary.values()])

        dim_results = {}

        # Tier 0
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

        # Tier 1
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
                bin_t1[m].append(float(np.mean(sub_scores[m])))
        for m in ('MLP', 'GAT'):
            print(f'    {m}: {np.mean(bin_t1[m]):.4f} +/- {np.std(bin_t1[m]):.4f}')
        dim_results['Tier1'] = {
            m: {'per_subject': bin_t1[m], 'f1_mean': float(np.mean(bin_t1[m])),
                'f1_std': float(np.std(bin_t1[m])), 'ci_95': list(bootstrap_ci(bin_t1[m]))}
            for m in ('MLP', 'GAT')
        }

        # Tier 2
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
                bin_t2[m].append(float(np.mean(sub_scores[m])))
        for m in ('MLP', 'GAT'):
            print(f'    {m}: {np.mean(bin_t2[m]):.4f} +/- {np.std(bin_t2[m]):.4f}')
        dim_results['Tier2'] = {
            m: {'per_subject': bin_t2[m], 'f1_mean': float(np.mean(bin_t2[m])),
                'f1_std': float(np.std(bin_t2[m])), 'ci_95': list(bootstrap_ci(bin_t2[m]))}
            for m in ('MLP', 'GAT')
        }

        # Tier 3 (LOSO)
        print(f'  Tier 3 (LOSO):')
        all_subs_bin = np.concatenate([
            np.full(len(d['X']), int(sid)) for sid, d in deap_binary.items()
        ])
        bin_t3 = {'MLP': [], 'GAT': []}
        for test_sub in sorted(deap_binary.keys()):
            test_mask = all_subs_bin == int(test_sub)
            train_mask = ~test_mask
            res = evaluate_fold(all_X_bin[train_mask], all_Y_bin[train_mask],
                               all_X_bin[test_mask], all_Y_bin[test_mask],
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

    # Save merged results
    with open(out_path, 'w') as f:
        _json.dump(all_results, f, indent=2)
    print(f'\nResults saved to: {out_path}')
    print('DONE')


if __name__ == '__main__':
    main()
