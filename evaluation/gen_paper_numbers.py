"""
Emit every number the conference paper cites, straight from all_results.json.

Exists because the submitted version of main.tex drifted out of sync with the
results file (its tables came from an earlier run than its statistics), so each
figure quoted in the text is regenerated here from a single source of truth.

Usage:  python gen_paper_numbers.py [--json outputs/all_results.json]
"""

import os, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# Models and tiers are read from the results file rather than hardcoded, so this
# stays correct as the sweep's scope changes (LightGAT dropped, binary LOSO
# skipped) without silently reporting stale structure.
TIERS = ('Tier0', 'Tier1', 'Tier2', 'Tier3')
TIER_LABEL = {
    'Tier0': '0 (Max leakage)',
    'Tier1': '1 (Within-subj., leaky)',
    'Tier2': '2 (Trial-aware)',
    'Tier3': '3 (LOSO)',
}


def g(node, *path, default=None):
    for p in path:
        if node is None:
            return default
        node = node.get(p) if isinstance(node, dict) else None
    return default if node is None else node


def fmt_pm(entry, digits=3):
    return f"{entry['f1_mean']:.{digits}f} $\\pm$ {entry['f1_std']:.{digits}f}"


def fmt_p(p):
    if p is None:
        return 'n/a'
    if p < 1e-4:
        return '$p<10^{-4}$'
    if p < 0.001:
        return '$p<0.001$'
    return f'$p={p:.3f}$'


def stat_line(s):
    return (f"d={s['mean_diff']:+.4f} 95% CI [{s['diff_ci_95'][0]:+.4f}, "
            f"{s['diff_ci_95'][1]:+.4f}], t={s['t_stat']:.3f}, "
            f"p_corr={s['p_corrected']:.2e}, d_z={s['cohens_d']:.2f}, "
            f"n={s['n_pairs']} {s['unit']}s")


def main(path):
    R = json.load(open(path))
    meta = R.get('_meta', {})
    n_sub = meta.get('deap_subjects', '?')

    # Derive the actual scope from the results file
    MODELS = tuple(R['DEAP']['Tier0'].keys())
    PARAMS = {'MLP': '$\\sim$116K', 'GAT': '$\\sim$126K', 'LightGAT': '$\\sim$16K'}
    bin_tiers = [t for t in TIERS if t in R['DEAP_binary']['valence']]

    print('=' * 78)
    print(f'PAPER NUMBERS  <-  {os.path.relpath(path, HERE)}')
    print(f'DEAP subjects: {n_sub} | windows: {meta.get("deap_windows","?"):,} | '
          f'OpenBCI windows: {meta.get("openbci_windows","?")} '
          f'({meta.get("openbci_trials","?")} trials)')
    print(f'Architectures: {", ".join(MODELS)} | binary tiers: {", ".join(bin_tiers)}')
    if len(bin_tiers) < 4:
        print('NOTE: binary LOSO (Tier3) not present -- Table II must not show T3 rows.')
    print('=' * 78)

    # ── Table: DEAP 4-class ──────────────────────────────────────────────────
    print(f'\n--- TABLE: 4-Class Results (DEAP, {n_sub} subjects, chance = 0.25) ---\n')
    print('Tier & ' + ' & '.join(f'{m} ({PARAMS.get(m, "")})' for m in MODELS) + ' \\\\')
    print('\\midrule')
    for t in TIERS:
        row = ' & '.join(fmt_pm(R['DEAP'][t][m]) for m in MODELS)
        print(f'{TIER_LABEL[t]} & {row} \\\\')

    # ── Table: DEAP binary ───────────────────────────────────────────────────
    print(f'\n--- TABLE: Binary Classification (DEAP, macro F1, chance = 0.50) ---\n')
    for dim in ('arousal', 'valence'):
        print(f'  [{dim}]')
        for t in bin_tiers:
            e = R['DEAP_binary'][dim][t]
            row = ' & '.join(f"{e[m]['f1_mean']:.3f}{{\\scriptsize$\\pm${e[m]['f1_std']:.3f}}}"
                             for m in MODELS)
            print(f'   & {TIER_LABEL[t]} & {row} \\\\')

    # ── Table: OpenBCI ───────────────────────────────────────────────────────
    print(f'\n--- TABLE: OpenBCI within-subject (4-class, chance = 0.25) ---\n')
    for t in ('Tier0', 'Tier1', 'Tier2'):
        row = ' & '.join(fmt_pm(R['OpenBCI'][t][m]) for m in MODELS)
        print(f'{TIER_LABEL[t]} & {row} \\\\')

    # ── Inline quantities: tier deltas ───────────────────────────────────────
    print('\n' + '=' * 78)
    print('INLINE NUMBERS CITED IN THE TEXT')
    print('=' * 78)

    print('\n[tier deltas, 4-class]')
    for m in MODELS:
        d = R['DEAP']
        t0, t1 = d['Tier0'][m]['f1_mean'], d['Tier1'][m]['f1_mean']
        t2, t3 = d['Tier2'][m]['f1_mean'], d['Tier3'][m]['f1_mean']
        print(f'  {m:<9} T0={t0:.4f} T1={t1:.4f} T2={t2:.4f} T3={t3:.4f} | '
              f'T0->T1 {t1-t0:+.4f} | T1->T2 {t2-t1:+.4f} | T2->T3 {t3-t2:+.4f} | '
              f'T0->T3 {t3-t0:+.4f} ({abs(t3-t0)*100:.1f} pts)')

    print('\n[binary tier deltas]')
    for dim in ('arousal', 'valence'):
        for m in MODELS:
            e = R['DEAP_binary'][dim]
            t1, t2 = e['Tier1'][m]['f1_mean'], e['Tier2'][m]['f1_mean']
            print(f'  {dim:<8} {m:<4} T1->T2 = {t2-t1:+.4f} ({abs(t2-t1)*100:.1f} pts)')

    print('\n[tier degradation tests -- paired by subject]')
    tc = g(R, 'statistics', 'tier_comparisons', default={})
    if isinstance(tc, dict):
        for m in MODELS:
            for s in tc.get(m, []):
                print(f'  {m:<9} {s["comparison"]:<16} {stat_line(s)}')
    else:  # older flat format
        for s in tc:
            print(f'  {s["comparison"]:<16} {stat_line(s)}')

    print('\n[architecture tests: GAT vs MLP, 4-class]')
    for s in g(R, 'statistics', 'model_comparisons', default=[]):
        verdict = 'SIGNIFICANT' if s.get('significant_bonferroni') else 'NOT SIGNIFICANT'
        print(f'  {s["comparison"]:<26} {stat_line(s)}   -> {verdict}')

    print('\n[architecture tests: GAT vs MLP, binary]')
    bmc = g(R, 'statistics', 'binary_model_comparisons', default={})
    for dim, lst in bmc.items():
        for s in lst:
            verdict = 'SIG' if s.get('significant_bonferroni') else 'ns'
            print(f'  {dim:<8} {s["comparison"]:<22} {stat_line(s)}   -> {verdict}')

    # ── LightGAT capacity control (only if it was evaluated) ─────────────────
    if 'LightGAT' in MODELS:
        print('\n[capacity control: LightGAT vs GAT, 4-class]')
        for t in TIERS:
            lg, gt = R['DEAP'][t]['LightGAT']['f1_mean'], R['DEAP'][t]['GAT']['f1_mean']
            print(f'  {t}: LightGAT={lg:.4f}  GAT={gt:.4f}  diff={lg-gt:+.4f}')
    else:
        print('\n[capacity control] LightGAT not evaluated in this run -- the paper '
              'must not cite it.')

    # ── Attention ────────────────────────────────────────────────────────────
    att = R.get('attention')
    if att:
        imp = np.asarray(att['channel_importance'])
        names = att['channel_names']
        order = np.argsort(imp)[::-1][:5]
        print('\n[attention top-5 channels, final GAT layer]')
        print('  ' + ', '.join(f'{names[i]} ({imp[i]:.4f})' for i in order))
    elif R.get('attention_top5_layer1'):
        print('\n[attention top-5 (legacy field)]')
        print('  ' + ', '.join(f"{e['channel']} ({e['score']})"
                               for e in R['attention_top5_layer1']))

    # ── Compute cost ─────────────────────────────────────────────────────────
    cc = R.get('compute_cost')
    if cc:
        print('\n[computational cost]')
        print(f'  device: {cc.get("gpu")}')
        for k, v in cc.get('phase_seconds', {}).items():
            print(f'    {k:<26} {v/60:8.1f} min')
        print(f'  TOTAL: {cc.get("total_hours")} h')

    # ── Consistency guard ────────────────────────────────────────────────────
    print('\n' + '=' * 78)
    print('CONSISTENCY CHECKS')
    print('=' * 78)
    ok = True
    for t in TIERS:
        for m in MODELS:
            e = R['DEAP'][t][m]
            scores = e.get('per_subject', e.get('f1_scores'))
            if scores is None:
                continue
            if not np.isclose(np.mean(scores), e['f1_mean'], atol=5e-4):
                print(f'  MISMATCH {t}/{m}: mean(scores)={np.mean(scores):.4f} '
                      f'vs stored {e["f1_mean"]:.4f}')
                ok = False
            if t != 'Tier0' and len(scores) != n_sub:
                print(f'  WARNING {t}/{m}: {len(scores)} per-subject scores '
                      f'but _meta says {n_sub} subjects')
                ok = False
    print('  all stored means match their score vectors' if ok else '  ^^ resolve before citing')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default=os.path.join(HERE, 'outputs', 'all_results.json'))
    main(ap.parse_args().json)
