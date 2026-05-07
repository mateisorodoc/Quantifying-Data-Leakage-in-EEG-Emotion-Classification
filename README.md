# Quantifying Data Leakage in EEG Emotion Classification: A Four-Tier Evaluation Framework

Code and results for the paper submitted to ICSTCC 2026.

## Key Findings

- Evaluation protocol selection accounts for a **39.7-point macro F1 variation** (T0: 0.607 → T3: 0.210), dominating any architectural difference.
- **Trial leakage** (Δ = −0.337, Cohen's d = 3.21) is the primary inflation mechanism.
- Under trial-aware evaluation, GAT provides a modest advantage over MLP (d = 0.74).
- Personalized within-subject models achieve **0.981 F1** (trial-aware), demonstrating BCI viability when inter-subject variability is eliminated.

## Repository Structure

```
evaluation/
├── feature_extraction.py       # 26-feature extraction (BP, DE, PLV, Coherence, PAC, Temporal)
├── run_evaluation_unified.py   # Four-tier evaluation (MLP + GAT, 4-class)
├── run_binary_only.py          # Binary arousal/valence evaluation
├── optuna_tuning.py            # Bayesian hyperparameter optimization (150 trials)
├── gen_figures.py              # Generate paper figures from results
└── outputs/
    ├── all_results.json        # Full results (all tiers, both datasets)
    ├── best_hparams.json       # Optimized hyperparameters
    └── optuna_summary.json     # Optuna search summary
paper.pdf
```

## Setup

### Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

```bash
pip install -r requirements.txt
```

### DEAP Dataset

1. Request access at [DEAP dataset](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/).
2. Download `data_preprocessed_python/` and place it at:
   ```
   data/DEAP/data_preprocessed_python/s01.dat ... s20.dat
   ```

## Reproducing Results

### 1. Feature Extraction

Extracts 26 features per channel (32 channels for DEAP, 16 for OpenBCI) with 2s windows and 50% overlap:

```bash
cd evaluation
python feature_extraction.py
```

Output: `data/DEAP/output/features_v6/` (one `.npz` per subject).

### 2. Hyperparameter Optimization (optional)

Uses Optuna (TPE sampler, 150 trials) to optimize GAT hyperparameters across Tiers 1-3:

```bash
python optuna_tuning.py
```

Pre-optimized hyperparameters are provided in `evaluation/outputs/best_hparams.json`.

### 3. Four-Tier Evaluation

Runs both MLP and GAT across all four tiers (T0-T3) for 4-class quadrant classification:

```bash
python run_evaluation_unified.py
```

Then run binary arousal/valence classification:

```bash
python run_binary_only.py
```

Results are saved to `evaluation/outputs/all_results.json`.

### 4. Generate Figures

```bash
python gen_figures.py
```

## Four-Tier Evaluation Protocol

| Tier | Protocol | Description |
|------|----------|-------------|
| T0 | Maximum Leakage | All subjects pooled, random 80/20 split |
| T1 | Within-Subject, Leaky | Per-subject StratifiedKFold, trial boundaries not enforced |
| T2 | Trial-Aware | Per-subject StratifiedGroupKFold (groups = trials) |
| T3 | Cross-Subject LOSO | Leave-One-Subject-Out |

## Citation

```bibtex
@inproceedings{sorodoc2026quantifying,
  title={Quantifying Data Leakage in EEG Emotion Classification: A Four-Tier Evaluation Framework},
  author={Sorodoc, Matei-George and Both, Roxana Rusu},
  booktitle={Proceedings of the International Conference on System Theory, Control and Computing (ICSTCC)},
  year={2026}
}
```

## License

MIT
