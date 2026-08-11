# Quantifying Data Leakage in EEG Emotion Classification: A Four-Tier Evaluation Framework

Code, results and verification for the paper submitted to ICSTCC 2026.
Matei-George Sorodoc, Roxana Rusu Both, Technical University of Cluj-Napoca.

Every number in the paper can be re-derived from this repository without a GPU and
without the datasets, using the scripts in `verification/`.

## Key findings

DEAP, all 32 subjects, 4-class quadrant, macro F1 (nominal chance = 0.25):

| Tier | Protocol | MLP | GAT |
|---|---|---|---|
| T0 | random split, subjects pooled | 0.551 | 0.579 |
| T1 | within-subject, trial boundaries ignored | 0.616 | 0.574 |
| T2 | trial-aware | 0.234 | 0.229 |
| T3 | leave-one-subject-out | 0.171 | 0.178 |

- Evaluation protocol accounts for a **40.1-point macro F1 swing**; the largest
  architecture difference at any tier is 4.2 points.
- **Trial leakage is the dominant component**: Δ = −0.345, Cohen's *d_z* = 3.27,
  *p* < 0.001, paired across n = 32 subjects.
- **MLP and GAT are statistically indistinguishable** at T2 and T3 (confidence
  intervals span zero). The only significant architecture effect is at T1 and favours
  the MLP, which is consistent with a flat high-capacity model exploiting leakage
  rather than decoding emotion.
- T2 sits at the empirical chance baseline (0.243) and **T3 falls significantly below
  it** (*p* < 0.001), indicating cross-subject transfer that actively misleads.
- The single-subject OpenBCI recording reaches 0.95 under trial-aware evaluation, but
  control experiments show this reflects a **session confound**, not emotion decoding:
  recording order *within* a single session, holding emotion constant, is itself
  decodable at 0.807, while sham labels sit at chance (0.481).

The headline conclusion is that **trial-aware evaluation is necessary but not
sufficient**: it constrains the window-to-trial relationship and leaves untouched any
nuisance variable that is constant within a trial and aligned with the label.

## Repository structure

```
paper/
├── main.tex                        Paper source
└── fig_tier_comparison.png         Figure 1
paper.pdf                           Compiled paper
evaluation/
├── feature_extraction.py           26 features/channel (BP, DE, PLV, coherence, PAC, temporal)
├── run_evaluation_unified.py       Four-tier sweep, checkpointed and resumable
├── openbci_session_control.py      The three session-confound control experiments
├── optuna_tuning.py                Hyperparameter search
├── gen_paper_numbers.py            Regenerates every number cited in the paper
├── gen_fig_revised.py              Regenerates Figure 1
└── outputs/
    ├── all_results.json            Master results: every tier, model and task
    ├── openbci_session_control.json  Control-experiment results
    ├── best_hparams.json           Hyperparameters used for all reported runs
    ├── optuna_summary.json         Search summary
    └── ckpt/                       312 per-unit checkpoints (raw, unaggregated evidence)
verification/
├── audit_paper.py                  108 numeric claims vs. the raw checkpoints
├── audit_methods.py                45 methodological claims vs. the source
├── audit_runtime.py                Runtime proof that evaluation is leak-free
├── check_refs.py                   Reference integrity and ordering
└── AUDIT_REPORT.md                 Audit results, including errors found and corrected
```

## Verifying the results

```bash
pip install -r requirements.txt
cd verification
python audit_paper.py     # 108/108 numeric claims
python audit_methods.py   # 45/45 methodological claims
python check_refs.py      # references cited, present, correctly ordered
```

`audit_paper.py` does not compare the paper against the summary file. It rebuilds the
per-subject score vectors from the 312 individual checkpoints in
`evaluation/outputs/ckpt/` and recomputes every t-test, Cohen's *d_z* and confidence
interval with scipy, so an error in the aggregation code would surface rather than be
reproduced. `audit_runtime.py` additionally proves leak-freedom by instrumenting a live
training call, but needs the DEAP features and a CUDA device.

## Reproducing from scratch

Requires the DEAP dataset and the custom OpenBCI recording, neither of which is
redistributed here.

```bash
python evaluation/feature_extraction.py --dataset deap
python evaluation/run_evaluation_unified.py      # ~8 h on an RTX 3080
python evaluation/openbci_session_control.py
python evaluation/gen_paper_numbers.py
```

The sweep checkpoints every work unit, so an interrupted run resumes where it stopped;
resumed results are bit-identical to an uninterrupted run. A startup guard refuses to
launch a second concurrent instance, since two processes sharing a checkpoint directory
can silently overwrite each other's results. Seed is fixed at 42 with
`cudnn.deterministic = True`.

## Methodological notes

**Model selection is kept off the test fold.** Early stopping uses a 15% validation
split carved from the training data, grouped by trial at Tier 2 and by subject at
Tier 3. The test fold is used exactly once, for the reported predictions.

**Chance is established empirically, not assumed.** The quadrant classes are unbalanced
after the per-subject median split, so uniform-random prediction against the observed
label distributions gives macro F1 = 0.243 and majority-class prediction gives 0.121.

**The class boundary is subject-relative.** The arousal median ranges from 1.97 to 7.02
across the 32 subjects, so the same raw rating denotes high arousal for one participant
and low arousal for another. Tiers 0 to 2 are unaffected; the paper bounds what Tier 3
can be said to measure because of it.

**Hyperparameters** were selected with Optuna on 5 of the 32 subjects (3 for the LOSO
term) with coarser folds than the reported runs. One configuration is shared across
both architectures and all four tiers, so no tier is tuned preferentially.

## Note on earlier versions

An earlier revision of this repository reported results from a pipeline that selected
the early-stopping epoch on the test fold, and that evaluated 20 of the 32 DEAP
subjects. Those numbers are superseded by everything published here. Two claims in
particular did not survive correction and should not be cited: a statistically
significant GAT advantage over MLP under trial-aware evaluation, and the reading of the
single-subject OpenBCI result as evidence for personalized BCI viability.

## Data availability

The DEAP dataset is available from its maintainers under their own licence. The custom
OpenBCI recording is not redistributed; its acquisition protocol, including the
class-per-session structure that confounds it, is described in full in the paper.
