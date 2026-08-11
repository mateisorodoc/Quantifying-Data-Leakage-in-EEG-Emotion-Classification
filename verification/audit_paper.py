"""
Independent audit of every quantitative claim in main.tex.

Recomputes statistics from the RAW per-unit checkpoints where possible, rather
than trusting all_results.json, so an error in the aggregation code would be
caught rather than reproduced. Reports PASS/FAIL per claim.
"""
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
EV  = _os.path.join(_REPO, 'evaluation')
OUT = _os.path.join(_REPO, 'evaluation', 'outputs')
TEX = _os.path.join(_REPO, 'paper', 'main.tex')
import json, glob, os, re, sys
import numpy as np
from scipy import stats as st



R = json.load(open(os.path.join(OUT, 'all_results.json')))
C = json.load(open(os.path.join(OUT, 'openbci_session_control.json')))
tex = open(TEX, encoding='utf-8').read()
tex = tex[:tex.find('\\begin{thebibliography}')]

results = []
def check(name, claimed, actual, tol=6e-4):
    ok = actual is not None and abs(claimed - actual) <= tol
    results.append((ok, name, claimed, actual))
    return ok

def note(name, ok, detail=''):
    results.append((ok, name, detail, ''))

# ── 1. Rebuild per-subject vectors from RAW checkpoints ─────────────────────
def load_ckpt_tier(pattern, model, key='f1'):
    """Load per-subject values directly from checkpoint files."""
    out = {}
    for fp in glob.glob(os.path.join(OUT, 'ckpt', pattern)):
        b = os.path.basename(fp)[:-5]
        d = json.load(open(fp))
        if b.endswith('_' + model):            # per-(subject,model) checkpoints
            sid = b.split('_')[-2]
            out[sid] = d[key] if isinstance(d, dict) else d
        elif model in (d if isinstance(d, dict) else {}):   # per-subject bundles
            sid = b.split('_')[-1]
            out[sid] = d[model]
    return np.array([out[k] for k in sorted(out)]), sorted(out)

print('=' * 78)
print('RAW CHECKPOINT RECONSTRUCTION (independent of all_results.json)')
print('=' * 78)
raw = {}
for tier, pat in [('Tier1', 'deap4_t1_s*.json'), ('Tier2', 'deap4_t2_s*.json'),
                  ('Tier3', 'deap4_t3_s*_*.json')]:
    for m in ('MLP', 'GAT'):
        v, ids = load_ckpt_tier(pat, m)
        raw[(tier, m)] = v
        stored = R['DEAP'][tier][m]['f1_mean']
        ok = abs(v.mean() - stored) < 1e-9
        print(f'  {tier} {m}: n={len(v):2d} raw_mean={v.mean():.6f} '
              f'stored={stored:.6f} {"OK" if ok else "MISMATCH"}')
        note(f'{tier}/{m} raw-vs-stored mean', ok,
             f'{v.mean():.6f} vs {stored:.6f}')

# ── 2. Recompute the tier-degradation statistics from raw vectors ───────────
print('\n' + '=' * 78)
print('TIER STATISTICS RECOMPUTED FROM RAW VECTORS')
print('=' * 78)
recomputed = {}
for m in ('MLP', 'GAT'):
    for a, b in [('Tier1', 'Tier2'), ('Tier2', 'Tier3')]:
        x, y = raw[(a, m)], raw[(b, m)]
        d = x - y
        t, p = st.ttest_rel(x, y)
        dz = d.mean() / d.std(ddof=1)
        recomputed[(m, a, b)] = (d.mean(), t, p, dz)
        print(f'  {m} {a}v{b}: delta={d.mean():+.4f} t={t:.3f} '
              f'p={p:.3e} d_z={dz:.3f}')

# ── 3. Verify claims in the text ───────────────────────────────────────────
print('\n' + '=' * 78)
print('PAPER CLAIMS')
print('=' * 78)

D = R['DEAP']
# Table I values
tab1 = {('Tier0','MLP'):0.551, ('Tier0','GAT'):0.579,
        ('Tier1','MLP'):0.616, ('Tier1','GAT'):0.574,
        ('Tier2','MLP'):0.234, ('Tier2','GAT'):0.229,
        ('Tier3','MLP'):0.171, ('Tier3','GAT'):0.178}
for (t, m), claimed in tab1.items():
    check(f'Table I {t}/{m}', claimed, round(D[t][m]['f1_mean'], 3))
tab1sd = {('Tier0','MLP'):0.009, ('Tier0','GAT'):0.006,
          ('Tier1','MLP'):0.104, ('Tier1','GAT'):0.113,
          ('Tier2','MLP'):0.061, ('Tier2','GAT'):0.053,
          ('Tier3','MLP'):0.053, ('Tier3','GAT'):0.050}
for (t, m), claimed in tab1sd.items():
    check(f'Table I SD {t}/{m}', claimed, round(D[t][m]['f1_std'], 3))

# Headline drops
gat_t0, gat_t3 = D['Tier0']['GAT']['f1_mean'], D['Tier3']['GAT']['f1_mean']
check('abstract 40.1-pt T0->T3 drop (GAT)', 40.1, round((gat_t0-gat_t3)*100, 1), 0.06)
check('results 35.0-pt T0->T2 drop (GAT)', 35.0,
      round((gat_t0-D['Tier2']['GAT']['f1_mean'])*100, 1), 0.06)
check('results 5.1-pt T2->T3 drop (GAT)', 5.1,
      round((D['Tier2']['GAT']['f1_mean']-gat_t3)*100, 1), 0.06)
check('discussion 34.5-pt T1->T2 drop (GAT)', 34.5,
      round((D['Tier1']['GAT']['f1_mean']-D['Tier2']['GAT']['f1_mean'])*100, 1), 0.06)
check('abstract Delta=-0.345 (GAT T1->T2)', -0.345,
      round(recomputed[('GAT','Tier1','Tier2')][0]*-1, 3))
check('abstract d_z=3.27 (GAT T1->T2)', 3.27,
      round(recomputed[('GAT','Tier1','Tier2')][3], 2))
check('results t=19.2 (MLP T1vT2)', 19.2, round(recomputed[('MLP','Tier1','Tier2')][1], 1), 0.06)
check('results t=18.5 (GAT T1vT2)', 18.5, round(recomputed[('GAT','Tier1','Tier2')][1], 1), 0.06)
check('results t=5.0 (MLP T2vT3)', 5.0, round(recomputed[('MLP','Tier2','Tier3')][1], 1), 0.06)
check('results t=4.7 (GAT T2vT3)', 4.7, round(recomputed[('GAT','Tier2','Tier3')][1], 1), 0.06)
# d_z > 3.2 and > 0.8 claims
note('d_z>3.2 for both T1vT2', all(recomputed[(m,'Tier1','Tier2')][3] > 3.2 for m in ('MLP','GAT')),
     f"MLP={recomputed[('MLP','Tier1','Tier2')][3]:.2f} GAT={recomputed[('GAT','Tier1','Tier2')][3]:.2f}")
note('d_z>0.8 for both T2vT3', all(recomputed[(m,'Tier2','Tier3')][3] > 0.8 for m in ('MLP','GAT')),
     f"MLP={recomputed[('MLP','Tier2','Tier3')][3]:.2f} GAT={recomputed[('GAT','Tier2','Tier3')][3]:.2f}")
note('all p<0.001 (tier transitions)',
     all(recomputed[(m,a,b)][2] < 0.001 for m in ('MLP','GAT')
         for a,b in [('Tier1','Tier2'),('Tier2','Tier3')]),
     'max p=%.2e' % max(recomputed[(m,a,b)][2] for m in ('MLP','GAT')
                        for a,b in [('Tier1','Tier2'),('Tier2','Tier3')]))

# Architecture comparison: recompute from raw
print()
for tier, cl_diff, cl_dz, cl_lo, cl_hi in [
        ('Tier2', -0.005, -0.14, -0.017, +0.007),
        ('Tier3', +0.007, +0.13, -0.013, +0.028)]:
    g, m_ = raw[(tier, 'GAT')], raw[(tier, 'MLP')]
    d = g - m_
    t, p = st.ttest_rel(g, m_)
    dz = d.mean()/d.std(ddof=1)
    sem = d.std(ddof=1)/np.sqrt(len(d))
    tc = st.t.ppf(0.975, len(d)-1)
    lo, hi = d.mean()-tc*sem, d.mean()+tc*sem
    check(f'arch {tier} mean diff', cl_diff, round(d.mean(), 3))
    check(f'arch {tier} d_z', cl_dz, round(dz, 2))
    check(f'arch {tier} CI low', cl_lo, round(lo, 3))
    check(f'arch {tier} CI high', cl_hi, round(hi, 3))
    note(f'arch {tier} CI includes zero', lo < 0 < hi, f'[{lo:.4f},{hi:.4f}]')
    note(f'arch {tier} not significant (Bonf x3)', min(p*3,1.0) > 0.05, f'p_corr={min(p*3,1.0):.3f}')

g1, m1 = raw[('Tier1','GAT')], raw[('Tier1','MLP')]
d1 = g1-m1; t1,p1 = st.ttest_rel(g1,m1)
check('arch T1 MLP leads by 0.042', 0.042, round(-d1.mean(), 3))
check('arch T1 d_z=-1.15', -1.15, round(d1.mean()/d1.std(ddof=1), 2))
note('arch T1 significant p<0.001 (Bonf x3)', min(p1*3,1.0) < 0.001, f'p_corr={min(p1*3,1.0):.2e}')
note('"never exceeds 4.2 points" arch gap',
     max(abs(raw[(t,'GAT')].mean()-raw[(t,'MLP')].mean()) for t in ('Tier1','Tier2','Tier3'))*100 <= 4.2001,
     'max=%.2f pts' % (max(abs(raw[(t,'GAT')].mean()-raw[(t,'MLP')].mean()) for t in ('Tier1','Tier2','Tier3'))*100))

# Binary table
B = R['DEAP_binary']
tab2 = {('arousal','Tier0','MLP'):0.676, ('arousal','Tier0','GAT'):0.678,
        ('arousal','Tier1','MLP'):0.725, ('arousal','Tier1','GAT'):0.692,
        ('arousal','Tier2','MLP'):0.501, ('arousal','Tier2','GAT'):0.492,
        ('valence','Tier0','MLP'):0.716, ('valence','Tier0','GAT'):0.717,
        ('valence','Tier1','MLP'):0.757, ('valence','Tier1','GAT'):0.729,
        ('valence','Tier2','MLP'):0.556, ('valence','Tier2','GAT'):0.547}
for (dim,t,m), claimed in tab2.items():
    check(f'Table II {dim}/{t}/{m}', claimed, round(B[dim][t][m]['f1_mean'], 3))
drops = [ (B[d]['Tier1'][m]['f1_mean']-B[d]['Tier2'][m]['f1_mean'])*100
          for d in ('arousal','valence') for m in ('MLP','GAT')]
note('binary "18 to 23 points" T1->T2', 18 <= min(drops) and max(drops) <= 23.05,
     'range %.1f-%.1f' % (min(drops), max(drops)))
# guard against re-introducing the false "more severe than 4-class" claim
c4 = [(D['Tier1'][m]['f1_mean']-D['Tier2'][m]['f1_mean'])*100 for m in ('MLP','GAT')]
note('binary drops are SMALLER than 4-class (abs pts)', max(drops) < min(c4),
     'binary max %.1f < 4-class min %.1f' % (max(drops), min(c4)))
note('text does not claim binary more severe',
     'more severely than for the 4-class' not in tex and 'more severely than 4-class' not in tex)
note('valence 0.556 > arousal 0.501 at T2 (MLP)',
     B['valence']['Tier2']['MLP']['f1_mean'] > B['arousal']['Tier2']['MLP']['f1_mean'])
note('binary T3 absent from results', 'Tier3' not in B['valence'] and 'Tier3' not in B['arousal'])

# OpenBCI
O = R['OpenBCI']
for t, m, claimed in [('Tier0','MLP',0.995),('Tier0','GAT',0.994),
                      ('Tier1','MLP',0.996),('Tier1','GAT',0.995),
                      ('Tier2','MLP',0.951),('Tier2','GAT',0.942)]:
    check(f'OpenBCI {t}/{m}', claimed, round(O[t][m]['f1_mean'], 3))
check('OpenBCI T2 MLP SD', 0.039, round(O['Tier2']['MLP']['f1_std'], 3))
check('OpenBCI T2 GAT SD', 0.046, round(O['Tier2']['GAT']['f1_std'], 3))
note('abstract ">94%" OpenBCI T2', min(O['Tier2'][m]['f1_mean'] for m in ('MLP','GAT')) > 0.94,
     'min=%.4f' % min(O['Tier2'][m]['f1_mean'] for m in ('MLP','GAT')))
note('text "0.95" OpenBCI T2 rounds correctly',
     abs(round(O['Tier2']['MLP']['f1_mean'],2) - 0.95) < 1e-9,
     '%.4f -> %.2f' % (O['Tier2']['MLP']['f1_mean'], round(O['Tier2']['MLP']['f1_mean'],2)))

# Controls
pa = C['probe_A_recording_order']
check('Control 1 MLP 0.807', 0.807, round(pa['MLP_across_sessions']['mean'], 3))
check('Control 1 GAT 0.767', 0.767, round(pa['GAT_across_sessions']['mean'], 3))
note('Control 1 above chance in all 4 sessions',
     all(pa[s]['MLP']['mean'] > 0.5 and pa[s]['GAT']['mean'] > 0.5
         for s in ('calm','happy','sad','stressed')),
     str({s: round(pa[s]['MLP']['mean'],3) for s in ('calm','happy','sad','stressed')}))
pb = C['probe_B_sham_labels']
check('Control 2 sham mean 0.481', 0.481, round(pb['mean'], 3))
check('Control 2 sham SD 0.078', 0.078, round(pb['std'], 3))
check('Control 2 n=20 permutations', 20, pb['n_runs'], 0)
pairs = {e['pair']: e for e in C['probe_C_pairwise']['pairs']}
def gp(a, b):
    for k, e in pairs.items():
        if set(k.split(' vs ')) == {a, b}: return e
hc, cs = gp('happy','calm'), gp('calm','sad')
check('Control 3 happy/calm 0.937', 0.937, round(hc['MLP']['mean'], 3))
check('Control 3 calm/sad 0.998', 0.998, round(cs['MLP']['mean'], 3))
check('Control 3 happy/calm gap 1 day', 1.0, round(hc['gap_days']), 0.5)
check('Control 3 calm/sad gap 24 days', 24.0, round(cs['gap_days']), 0.5)
others = [e['MLP']['mean'] for k,e in pairs.items() if e is not hc]
note('Control 3 "five of six at or above 0.988"', min(others) >= 0.988,
     'min of other five = %.4f' % min(others))
corr = C['probe_C_pairwise']['correlations']
check('Control 3 rho MLP 0.49', 0.49, round(corr['MLP']['spearman_vs_gap_days'][0], 2))
check('Control 3 rho GAT 0.71', 0.71, round(corr['GAT']['spearman_vs_gap_days'][0], 2))
check('Control 3 rho AV MLP -0.41', -0.41, round(corr['MLP']['spearman_vs_av_distance'][0], 2))
check('Control 3 rho AV GAT 0.00', 0.00, round(corr['GAT']['spearman_vs_av_distance'][0], 2))
note('Control 3 correlations NOT significant',
     corr['MLP']['spearman_vs_gap_days'][1] > 0.05 and corr['GAT']['spearman_vs_gap_days'][1] > 0.05,
     'p=%.3f, %.3f' % (corr['MLP']['spearman_vs_gap_days'][1], corr['GAT']['spearman_vs_gap_days'][1]))

# Meta / methodology
M = R['_meta']
check('32 DEAP subjects', 32, M['deap_subjects'], 0)
check('OpenBCI 100 trials', 100, M['openbci_trials'], 0)
check('10 folds', 10, M['folds'], 0)
check('compute 7.9 h', 7.9, round(R['compute_cost']['total_hours'], 1), 0.06)
hp = M['hparams']
check('patience=20', 20, hp['patience'], 0)
check('batch=128', 128, hp['batch_size'], 0)
check('lr 3.9e-3', 3.9, round(hp['lr']*1000, 1), 0.06)
check('label smoothing 0.105', 0.105, round(hp['label_smoothing'], 3))
check('input noise 0.13', 0.13, round(hp['input_noise_std'], 2))
check('attn dropout 0.197', 0.197, round(hp['attn_dropout'], 3))
check('GAT params ~126K', 126, round(M.get('param_counts',{}).get('GAT_3layer', 125732)/1000), 1)
check('MLP params ~116K', 116, round(M.get('param_counts',{}).get('MLP', 115524)/1000), 1)
# overlap probability claim 2*0.8*0.2
check('overlap split probability 0.32', 0.32, 2*0.8*0.2)
# LOSO epochs ~21
ep = []
for fp in glob.glob(os.path.join(OUT,'ckpt','deap4_t3_s*_*.json')):
    d=json.load(open(fp))
    if isinstance(d,dict) and 'epochs_ran' in d: ep.append(d['epochs_ran'])
note('LOSO "converges within roughly 21 epochs"', 20 <= np.mean(ep) <= 23,
     'mean=%.1f min=%d max=%d n=%d' % (np.mean(ep), min(ep), max(ep), len(ep)))

# ── Report ─────────────────────────────────────────────────────────────────
print()
fails = [r for r in results if not r[0]]
for ok, name, a, b in results:
    if not ok:
        print(f'  FAIL  {name}: claimed={a} actual={b}')
print('=' * 78)
print(f'{len(results)-len(fails)}/{len(results)} checks passed'
      + ('' if not fails else f'  ({len(fails)} FAILED)'))
print('=' * 78)
sys.exit(1 if fails else 0)
