"""
═══════════════════════════════════════════════════════════════════════════════
DeepGAT: Enhanced Feature Extraction for EEG Emotion Classification
═══════════════════════════════════════════════════════════════════════════════
Extracts enhanced features from raw DEAP/OpenBCI data:
  - Band Power (BP): 5 bands × 16 channels
  - Differential Entropy (DE): 5 bands × 16 channels
  - Phase-Locking Value (PLV): mean connectivity per channel per band
  - Coherence: mean Welch coherence per channel per band
  - Phase-Amplitude Coupling (PAC): theta-gamma MI per channel
  - Temporal Variability: sub-window BP variance per channel per band

Output: (N_windows, 32, N_FEATS) arrays cached as features_v6/ [DEAP]
        (N_windows, 16, N_FEATS) arrays cached as features_v5/ [OpenBCI]
═══════════════════════════════════════════════════════════════════════════════
"""

import os, pickle, sys, glob
import numpy as np
from scipy import signal
from scipy.signal import hilbert
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ─── Configuration ────────────────────────────────────────────────────────────
# DEAP: BioSemi ActiveTwo @ 128 Hz
DEAP_FS = 128
DEAP_WINDOW = 256   # 2 seconds @ 128 Hz
DEAP_STEP = 128     # 1 second step (50% overlap)
PRE_TRIAL = 3 * DEAP_FS  # 3s baseline to skip in DEAP

# OpenBCI: Cyton+Daisy @ 125 Hz (confirmed in filter_data.ipynb: TARGET_SFREQ=125.0)
OPENBCI_FS = 125
OPENBCI_WINDOW = 250  # 2 seconds @ 125 Hz
OPENBCI_STEP = 125    # 1 second step (50% overlap)

# Legacy alias (used by feature functions as default)
FS = 128

BANDS = [(4, 8), (8, 12), (12, 16), (16, 25), (25, 45)]
BAND_NAMES = ['theta', 'alpha', 'beta_l', 'beta_h', 'gamma']
N_BANDS = len(BANDS)

# DEAP 32 EEG channels (full set)
# Order: Fp1,AF3,F3,F7,FC5,FC1,C3,T7,CP5,CP1,P3,P7,PO3,O1,Oz,Pz,
#        Fp2,AF4,F4,F8,FC6,FC2,C4,T8,CP6,CP2,P4,P8,PO4,O2,O9,O10
DEAP_32_CHANNELS = list(range(32))  # Use all 32 EEG channels
DEAP_CHANNEL_NAMES = [
    'Fp1','AF3','F3','F7','FC5','FC1','C3','T7',
    'CP5','CP1','P3','P7','PO3','O1','Oz','Pz',
    'Fp2','AF4','F4','F8','FC6','FC2','C4','T8',
    'CP6','CP2','P4','P8','PO4','O2','O9','O10'
]
DEAP_N_CH = 32

# OpenBCI uses 16 channels (unchanged)
OPENBCI_N_CH = 16
OPENBCI_CHANNEL_NAMES = ['Fp1','F3','F7','C3','T7','P3','P7','O1',
                         'Fp2','F4','F8','C4','T8','P4','P8','O2']
N_CH = 16  # Default for OpenBCI extraction functions

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEAP_RAW = os.path.join(PROJECT_ROOT, '..', 'data', 'DEAP', 'data_preprocessed_python')
DEAP_OUT = os.path.join(PROJECT_ROOT, '..', 'data', 'DEAP', 'output', 'features_v6')
OPENBCI_RAW = os.path.join(PROJECT_ROOT, '..', 'data', 'recordings_clean')
OPENBCI_OUT = os.path.join(OPENBCI_RAW, 'features_v5')

os.makedirs(DEAP_OUT, exist_ok=True)
os.makedirs(OPENBCI_OUT, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def bandpass_filter(data, lo, hi, fs=FS, order=4):
    """Apply Butterworth bandpass filter."""
    nyq = fs / 2
    lo_n, hi_n = lo / nyq, min(hi / nyq, 0.99)
    b, a = signal.butter(order, [lo_n, hi_n], btype='band')
    return signal.filtfilt(b, a, data, axis=-1)


def compute_band_power(window, fs=FS):
    """Compute band power for all channels using Welch's method.
    
    Args:
        window: (N_CH, WINDOW) array
    Returns:
        bp: (N_CH, N_BANDS) array
    """
    freqs, psd = signal.welch(window, fs=fs, nperseg=min(128, window.shape[-1]),
                              noverlap=64, axis=-1)
    bp = np.zeros((window.shape[0], N_BANDS))
    for b, (lo, hi) in enumerate(BANDS):
        mask = (freqs >= lo) & (freqs < hi)
        bp[:, b] = np.log1p(psd[:, mask].mean(axis=-1))
    return bp


def compute_differential_entropy(window, fs=FS):
    """Compute DE for all channels per band.
    
    Args:
        window: (N_CH, WINDOW) array
    Returns:
        de: (N_CH, N_BANDS) array
    """
    de = np.zeros((window.shape[0], N_BANDS))
    for b, (lo, hi) in enumerate(BANDS):
        filtered = bandpass_filter(window, lo, hi, fs)
        var = np.var(filtered, axis=-1) + 1e-10
        de[:, b] = 0.5 * np.log(2 * np.pi * np.e * var)
    return de


def compute_plv_summary(window, fs=FS):
    """Compute mean Phase-Locking Value for each channel per band.
    
    PLV between channels i and j in band b:
      PLV_ij = |mean(exp(j*(phase_i - phase_j)))|
    
    Per-channel summary: mean PLV to all other channels.
    
    Args:
        window: (N_CH, WINDOW) array
    Returns:
        plv_summary: (N_CH, N_BANDS) array
    """
    n_ch = window.shape[0]
    plv_summary = np.zeros((n_ch, N_BANDS))
    
    for b, (lo, hi) in enumerate(BANDS):
        filtered = bandpass_filter(window, lo, hi, fs)
        analytic = hilbert(filtered, axis=-1)
        phases = np.angle(analytic)  # (N_CH, WINDOW)
        
        # Compute pairwise PLV efficiently
        # Phase differences: (N_CH, N_CH, WINDOW)
        phase_diff = phases[:, np.newaxis, :] - phases[np.newaxis, :, :]
        plv_matrix = np.abs(np.mean(np.exp(1j * phase_diff), axis=-1))  # (N_CH, N_CH)
        
        # Per-channel mean (exclude self-connection)
        np.fill_diagonal(plv_matrix, 0)
        plv_summary[:, b] = plv_matrix.sum(axis=1) / (n_ch - 1)
    
    return plv_summary


def compute_coherence_summary(window, fs=FS):
    """Compute mean magnitude-squared coherence per channel per band.
    
    Uses Welch cross-spectral density.
    
    Args:
        window: (N_CH, WINDOW) array
    Returns:
        coh_summary: (N_CH, N_BANDS) array
    """
    n_ch = window.shape[0]
    nperseg = min(64, window.shape[-1] // 2)
    coh_summary = np.zeros((n_ch, N_BANDS))
    
    # Compute PSD for each channel
    freqs, psd_all = signal.welch(window, fs=fs, nperseg=nperseg, axis=-1)
    
    # For each channel pair, compute coherence
    # This is O(N_CH^2) but N_CH=16 so it's fine
    coh_matrix = np.zeros((n_ch, n_ch, N_BANDS))
    
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            freqs_c, cxy = signal.coherence(window[i], window[j], fs=fs, nperseg=nperseg)
            for b, (lo, hi) in enumerate(BANDS):
                mask = (freqs_c >= lo) & (freqs_c < hi)
                if mask.any():
                    coh_val = cxy[mask].mean()
                    coh_matrix[i, j, b] = coh_val
                    coh_matrix[j, i, b] = coh_val
    
    # Per-channel mean coherence
    for b in range(N_BANDS):
        mat = coh_matrix[:, :, b]
        coh_summary[:, b] = mat.sum(axis=1) / (n_ch - 1)
    
    return coh_summary


def compute_pac(window, fs=FS):
    """Compute Phase-Amplitude Coupling (theta phase → gamma amplitude).
    
    Modulation Index (MI) using mean vector length method.
    
    Args:
        window: (N_CH, WINDOW) array
    Returns:
        pac: (N_CH, 1) array
    """
    n_ch = window.shape[0]
    
    # Theta phase (4-8 Hz)
    theta_filtered = bandpass_filter(window, 4, 8, fs)
    theta_phase = np.angle(hilbert(theta_filtered, axis=-1))
    
    # Gamma amplitude (25-45 Hz)
    gamma_filtered = bandpass_filter(window, 25, 45, fs)
    gamma_amp = np.abs(hilbert(gamma_filtered, axis=-1))
    
    # Modulation Index: |mean(amp * exp(j*phase))| / mean(amp)
    mi = np.abs(np.mean(gamma_amp * np.exp(1j * theta_phase), axis=-1)) / (np.mean(gamma_amp, axis=-1) + 1e-10)
    
    return mi.reshape(n_ch, 1)


def compute_temporal_variability(window, fs=FS, n_subwindows=4):
    """Compute temporal variability of band power within the window.
    
    Split window into n_subwindows chunks, compute BP for each,
    then take the variance across chunks.
    
    Args:
        window: (N_CH, WINDOW) array
    Returns:
        tv: (N_CH, N_BANDS) array
    """
    n_ch = window.shape[0]
    chunk_size = window.shape[-1] // n_subwindows
    
    # BP for each sub-window
    sub_bps = []
    for i in range(n_subwindows):
        chunk = window[:, i*chunk_size:(i+1)*chunk_size]
        freqs, psd = signal.welch(chunk, fs=fs, nperseg=min(32, chunk_size),
                                  noverlap=16, axis=-1)
        bp = np.zeros((n_ch, N_BANDS))
        for b, (lo, hi) in enumerate(BANDS):
            mask = (freqs >= lo) & (freqs < hi)
            if mask.any():
                bp[:, b] = psd[:, mask].mean(axis=-1)
        sub_bps.append(bp)
    
    sub_bps = np.stack(sub_bps, axis=0)  # (n_subwindows, N_CH, N_BANDS)
    tv = np.var(sub_bps, axis=0)  # (N_CH, N_BANDS)
    tv = np.log1p(tv)  # Log-scale to prevent explosion
    
    return tv


def extract_features_window(window, fs=FS, feature_groups=None):
    """Extract all features for a single window.
    
    Args:
        window: (N_CH, WINDOW_SAMPLES) array
        fs: sampling frequency
        feature_groups: list of groups to compute, or None for all
    Returns:
        features: (N_CH, N_FEATS) array where N_FEATS depends on groups selected
    """
    if feature_groups is None:
        feature_groups = ['bp', 'de', 'plv', 'coherence', 'pac', 'temporal']
    
    parts = []
    
    if 'bp' in feature_groups:
        parts.append(compute_band_power(window, fs))  # (16, 5)
    
    if 'de' in feature_groups:
        parts.append(compute_differential_entropy(window, fs))  # (16, 5)
    
    if 'plv' in feature_groups:
        parts.append(compute_plv_summary(window, fs))  # (16, 5)
    
    if 'coherence' in feature_groups:
        parts.append(compute_coherence_summary(window, fs))  # (16, 5)
    
    if 'pac' in feature_groups:
        parts.append(compute_pac(window, fs))  # (16, 1)
    
    if 'temporal' in feature_groups:
        parts.append(compute_temporal_variability(window, fs))  # (16, 5)
    
    return np.concatenate(parts, axis=1)  # (16, total_feats)


# ═══════════════════════════════════════════════════════════════════════════════
# DEAP EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_deap_subject(sub_id, feature_groups=None):
    """Extract enhanced features for one DEAP subject.
    
    Args:
        sub_id: subject string like '01'
    Returns:
        dict with 'features' (N, 32, F), 'labels' (N, 2), 'trials' (N,)
    """
    fp = os.path.join(DEAP_RAW, f's{sub_id}.dat')
    with open(fp, 'rb') as f:
        d = pickle.load(f, encoding='latin1')
    
    data = d['data']  # (40, 40, 8064)
    labels = d['labels']  # (40, 4): valence, arousal, dominance, liking
    
    all_features = []
    all_labels = []
    all_trials = []
    
    for trial_idx in range(40):
        # Extract all 32 EEG channels
        eeg = data[trial_idx, :DEAP_N_CH, :]  # (32, 8064)
        
        # Skip pre-trial baseline (first 3 seconds)
        eeg = eeg[:, PRE_TRIAL:]  # (32, 7680)
        
        # Sliding window
        n_samples = eeg.shape[1]
        n_windows = (n_samples - DEAP_WINDOW) // DEAP_STEP + 1
        
        for w_idx in range(n_windows):
            start = w_idx * DEAP_STEP
            window = eeg[:, start:start + DEAP_WINDOW]
            
            feats = extract_features_window(window, DEAP_FS, feature_groups)
            all_features.append(feats)
            all_labels.append([labels[trial_idx, 1], labels[trial_idx, 0]])  # arousal, valence
            all_trials.append(trial_idx)
    
    return {
        'features': np.array(all_features, dtype=np.float32),
        'labels': np.array(all_labels, dtype=np.float32),
        'trials': np.array(all_trials, dtype=np.int32)
    }


def extract_all_deap(feature_groups=None, subjects=None):
    """Extract features for all DEAP subjects and cache."""
    if subjects is None:
        subjects = [f'{i:02d}' for i in range(1, 21)]
    
    print(f'Extracting DEAP features (v6) for {len(subjects)} subjects...')
    print(f'  Window: {DEAP_WINDOW} samples ({DEAP_WINDOW/DEAP_FS:.1f}s), Step: {DEAP_STEP} ({DEAP_STEP/DEAP_FS:.2f}s)')
    print(f'  Sampling rate: {DEAP_FS} Hz')
    print(f'  Feature groups: {feature_groups or "all"}')
    
    for sub_id in tqdm(subjects, desc='DEAP subjects'):
        out_fp = os.path.join(DEAP_OUT, f's{sub_id}.npz')
        if os.path.exists(out_fp):
            # Verify shape
            d = np.load(out_fp)
            if d['features'].ndim == 3 and d['features'].shape[1] == DEAP_N_CH:
                continue
        
        result = extract_deap_subject(sub_id, feature_groups)
        np.savez_compressed(out_fp,
                           features=result['features'],
                           labels=result['labels'],
                           trials=result['trials'])
        print(f'  s{sub_id}: {result["features"].shape[0]} windows, '
              f'{result["features"].shape[2]} features/channel')
    
    print('DEAP extraction complete.')


# ═══════════════════════════════════════════════════════════════════════════════
# OpenBCI EXTRACTION (from cleaned CSV files)
# ═══════════════════════════════════════════════════════════════════════════════

# CSV column order: relative_time,Fp1,Fp2,C3,C4,P7,P8,O1,O2,F7,F8,F3,F4,T7,T8,P3,P4,video
# We reorder to standard 10-20 layout for consistent spatial interpretation
CSV_EEG_COLUMNS = ['Fp1','Fp2','C3','C4','P7','P8','O1','O2',
                   'F7','F8','F3','F4','T7','T8','P3','P4']
# Target order: left hemisphere anterior→posterior, then right hemisphere
OPENBCI_TARGET_ORDER = ['Fp1','F3','F7','C3','T7','P3','P7','O1',
                        'Fp2','F4','F8','C4','T8','P4','P8','O2']
# Column reorder indices (CSV order → target order)
_REORDER_IDX = [CSV_EEG_COLUMNS.index(ch) for ch in OPENBCI_TARGET_ORDER]


def extract_openbci_from_csv(feature_groups=None):
    """Extract 26 features from cleaned OpenBCI CSV recordings.
    
    Reads CSV files from recordings_{class}_cleaned/clean_trial_*.csv
    Windows at 2s (250 samples) with 1s step (125 samples) at 125 Hz.
    
    Returns:
        X: (N_windows, 16, 26) feature array
        Y: (N_windows,) class labels [0=calm, 1=happy, 2=sad, 3=stressed]
        groups: (N_windows,) trial identifiers for GroupKFold
    """
    label_map = {'calm': 0, 'happy': 1, 'sad': 2, 'stressed': 3}
    all_X, all_Y, all_groups = [], [], []
    trial_id = 0
    
    for emotion, label in label_map.items():
        csv_dir = os.path.join(OPENBCI_RAW, f'recordings_{emotion}_cleaned')
        if not os.path.isdir(csv_dir):
            print(f'  ⚠ Directory not found: {csv_dir}')
            continue
        
        csv_files = sorted(glob.glob(os.path.join(csv_dir, 'clean_trial_*.csv')))
        print(f'  {emotion}: {len(csv_files)} trials')
        
        for fp in csv_files:
            try:
                # Read CSV, skip header
                import pandas as pd
                df = pd.read_csv(fp)
                
                # Extract EEG columns and reorder
                eeg = df[CSV_EEG_COLUMNS].values  # (n_samples, 16)
                eeg = eeg[:, _REORDER_IDX]  # Reorder to standard layout
                eeg = eeg.T  # (16, n_samples)
                
                n_samples = eeg.shape[1]
                n_windows = (n_samples - OPENBCI_WINDOW) // OPENBCI_STEP + 1
                
                if n_windows < 1:
                    print(f'    ⚠ Trial too short ({n_samples} samples): {os.path.basename(fp)}')
                    continue
                
                for w_idx in range(n_windows):
                    start = w_idx * OPENBCI_STEP
                    window = eeg[:, start:start + OPENBCI_WINDOW]
                    
                    feats = extract_features_window(window, OPENBCI_FS, feature_groups)
                    all_X.append(feats)
                    all_Y.append(label)
                    all_groups.append(trial_id)
                
                trial_id += 1
                
            except Exception as e:
                print(f'    ⚠ Error processing {os.path.basename(fp)}: {e}')
                continue
    
    X = np.array(all_X, dtype=np.float32)
    Y = np.array(all_Y, dtype=np.int64)
    groups = np.array(all_groups, dtype=np.int32)
    
    print(f'  Total: {X.shape[0]} windows from {trial_id} trials')
    print(f'  Class distribution: {dict(zip(*np.unique(Y, return_counts=True)))}')
    
    return X, Y, groups


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['deap', 'openbci', 'all'], default='all')
    parser.add_argument('--subjects', nargs='+', default=None,
                        help='DEAP subject IDs (e.g., 01 02 03)')
    parser.add_argument('--force', action='store_true', help='Re-extract even if cache exists')
    args = parser.parse_args()
    
    feature_groups = ['bp', 'de', 'plv', 'coherence', 'pac', 'temporal']
    
    if args.dataset in ('deap', 'all'):
        if args.force:
            for f in glob.glob(os.path.join(DEAP_OUT, '*.npz')):
                os.remove(f)
        extract_all_deap(feature_groups, args.subjects)
    
    if args.dataset in ('openbci', 'all'):
        out_fp = os.path.join(OPENBCI_OUT, 'openbci_all.npz')
        if args.force and os.path.exists(out_fp):
            os.remove(out_fp)
        
        if os.path.exists(out_fp) and not args.force:
            d = np.load(out_fp)
            print(f'\nOpenBCI features already cached: {d["X"].shape}')
        else:
            print(f'\nExtracting OpenBCI features from CSV (fs={OPENBCI_FS}Hz)...')
            print(f'  Window: {OPENBCI_WINDOW} samples ({OPENBCI_WINDOW/OPENBCI_FS:.1f}s)')
            print(f'  Step: {OPENBCI_STEP} samples ({OPENBCI_STEP/OPENBCI_FS:.1f}s)')
            X, Y, groups = extract_openbci_from_csv(feature_groups)
            np.savez_compressed(out_fp, X=X, Y=Y, groups=groups)
            print(f'  Saved: {out_fp}')
            print(f'  Shape: X={X.shape}, Y={Y.shape}')
