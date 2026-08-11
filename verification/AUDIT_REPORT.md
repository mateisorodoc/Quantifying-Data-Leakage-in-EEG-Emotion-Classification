# Audit report

Independent verification of the paper against the code and the raw results.
All checks were re-run from inside this folder after the final edits.

| Audit | Scope | Result |
|---|---|---|
| `audit_paper.py` | 108 numeric claims, recomputed from the 312 raw checkpoints | **108/108 pass** |
| `audit_methods.py` | 45 methodological claims, checked against `code/` | **45/45 pass** |
| `audit_runtime.py` | Leak-freedom, instrumented on real DEAP data | **all pass** (needs data + GPU) |
| `check_refs.py` | 30 references cited, present, first-citation order | **0 problems** |

## What the numeric audit does

It does not read the reported means and compare them to the paper. It reloads each of
the 312 per-unit checkpoint files, rebuilds the per-subject score vectors, and
recomputes every statistic with scipy from scratch: paired t-tests, Cohen's *d_z*,
confidence intervals and the tier deltas. Those recomputed values then have to match
both `all_results.json` and the text of the paper. Reconstructed means matched the
stored aggregates to six decimal places across all six tier/model combinations.

## What the runtime audit proves

It monkey-patches the dataset class inside a live `train_model` call on real DEAP data
and hashes every row that reaches each loader, then asserts:

- the validation set shares **zero rows** with the test fold;
- sub-train and validation are disjoint and together reconstruct the training set exactly;
- the returned predictions are for the test fold, and there is one per test row;
- no trial straddles sub-train and validation at Tier 2, and no subject does at Tier 3;
- the feature scaler reproduces a train-only fit, so no test statistics leak into it.

## Errors this audit found and corrected

1. **False numeric claim.** The binary T1→T2 drop was stated as "20 to 23 points"; the
   true range is 18.2 to 22.5. Corrected, with a permanent guard in the audit.
2. **False comparative claim.** The text said binary degradation was more severe than
   4-class. It is the opposite: binary drops 18–23 points, 4-class drops 34.5–38.2.
   Removed, with a guard.
3. **Overstatement.** Architecture differences were called "an order of magnitude"
   smaller than tier effects; true at T2 (74×) but not T3 (7.0×). Replaced with the
   measured factors. The separate order-of-magnitude claim in the Discussion was
   verified at 9.6× and left standing.
4. **Unexplained below-chance results.** T2 and T3 fall at or below the chance line and
   the figure plots them there. The chance baseline is now derived empirically and the
   below-chance T3 result is tested and explained rather than left for a reviewer to
   discover.

## Known limits of this audit

- Third-party numbers taken from cited papers (for example Koelstra's F1 = 0.563) were
  checked against the source PDFs but not independently re-derived.
- `audit_runtime.py` requires the DEAP features and a CUDA device, so it cannot be run
  from this folder alone.
- The audit verifies internal consistency and leak-freedom. It cannot validate the
  upstream feature extraction against an external reference implementation.
