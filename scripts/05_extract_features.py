"""
Step 5: Master Feature Extractor (86 Dimensions)

Features extracted:
  - MFCC Mean & Std (26 dims)
  - MFCC Delta Mean & Std (26 dims)
  - MFCC Delta-Delta (Acceleration) Mean & Std (26 dims) -- NEW
  - Zero-Crossing Rate Mean & Std (2 dims)
  - Spectral Centroid Mean & Std (2 dims)
  - Spectral Flatness Mean & Std (2 dims)
  - Spectral Bandwidth Mean & Std (2 dims)
"""
from pathlib import Path
import numpy as np
import soundfile as sf
import pandas as pd
from scipy.fftpack import dct

ROOT = Path(__file__).parent.parent
CLIPS_DIR = ROOT / "processed" / "clips"
SR = 22050
N_MFCC = 13
TEST_MATCHES = ["match9", "match7", "match13"]

def normalize(y):
    max_abs = np.max(np.abs(y))
    if max_abs < 1e-4:
        return y
    return y / (max_abs + 1e-9)

def time_shift(y, sr, shift_sec=0.05):
    return np.roll(y, int(shift_sec * sr))

def add_noise(y, noise_factor=0.001):
    return y + noise_factor * np.random.normal(0, 1, len(y))

def compute_ultimate_features(y, sr=SR, n_mfcc=N_MFCC, n_fft=512, hop_length=256, n_mels=40):
    window = np.hanning(n_fft)
    frames = [y[i:i+n_fft] * window for i in range(0, len(y) - n_fft + 1, hop_length)]
    
    if not frames:
        return np.zeros(86)
        
    frames = np.array(frames)
    stft_matrix = np.fft.rfft(frames, n=n_fft, axis=-1)
    magnitude_spectrum = np.abs(stft_matrix)
    power_spectrum = magnitude_spectrum ** 2

    # Mel Filterbank Configuration
    low_mel = 0
    high_mel = 2595 * np.log10(1 + (sr / 2) / 700)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = 700 * (10**(mel_points / 2595) - 1)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    
    fbank = np.zeros((n_mels, int(n_fft / 2 + 1)))
    for m in range(1, n_mels + 1):
        f_m_minus, f_m, f_m_plus = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        for k in range(f_m_minus, f_m):
            if (f_m - f_m_minus) > 0: fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if (f_m_plus - f_m) > 0: fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    filter_banks = np.dot(power_spectrum, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)

    # 1. Timbral Properties (MFCCs)
    mfcc = dct(filter_banks, type=2, axis=-1, norm='ortho')[:, :n_mfcc]
    mfcc_mean, mfcc_std = np.mean(mfcc, axis=0), np.std(mfcc, axis=0)

    # 2. Temporal Velocity (Deltas)
    if len(mfcc) > 1:
        deltas = np.diff(mfcc, axis=0)
        delta_mean, delta_std = np.mean(deltas, axis=0), np.std(deltas, axis=0)
    else:
        deltas = np.zeros((1, n_mfcc))
        delta_mean, delta_std = np.zeros(n_mfcc), np.zeros(n_mfcc)

    # 3. NEW: Temporal Acceleration (Delta-Deltas)
    if len(deltas) > 1:
        delta_deltas = np.diff(deltas, axis=0)
        dd_mean, dd_std = np.mean(delta_deltas, axis=0), np.std(delta_deltas, axis=0)
    else:
        dd_mean, dd_std = np.zeros(n_mfcc), np.zeros(n_mfcc)

    # 4. Structural Periodicity (ZCR)
    zcr_frames = [np.sum(np.abs(np.diff(np.sign(f)))) / 2 / len(f) for f in frames]
    zcr_mean, zcr_std = np.mean(zcr_frames), np.std(zcr_frames)

    # 5. Center of Mass (Spectral Centroid)
    freqs = np.fft.rfftfreq(n_fft, d=1/sr)
    centroid_frames = []
    for mag in magnitude_spectrum:
        mag_sum = np.sum(mag)
        centroid_frames.append(np.sum(freqs * mag) / mag_sum if mag_sum > 0 else 0.0)
    centroid_mean, centroid_std = np.mean(centroid_frames), np.std(centroid_frames)

    # 6. Tonal Clarity (Spectral Flatness)
    flatness_frames = []
    for power_frame in power_spectrum:
        power_eps = power_frame + 1e-9
        log_mean = np.mean(np.log(power_eps))
        arith_mean = np.mean(power_eps)
        flatness_frames.append(np.exp(log_mean) / arith_mean if arith_mean > 0 else 0.0)
    flatness_mean, flatness_std = np.mean(flatness_frames), np.std(flatness_frames)

    # 7. Frequency Range Concentration (Spectral Bandwidth)
    bandwidth_frames = []
    for idx, power_frame in enumerate(power_spectrum):
        p_sum = np.sum(power_frame)
        if p_sum > 0:
            centroid = centroid_frames[idx]
            variance = np.sum(power_frame * (freqs - centroid)**2) / p_sum
            bandwidth_frames.append(np.sqrt(variance))
        else:
            bandwidth_frames.append(0.0)
    bandwidth_mean, bandwidth_std = np.mean(bandwidth_frames), np.std(bandwidth_frames)

    return np.concatenate([
        mfcc_mean, mfcc_std, 
        delta_mean, delta_std,
        dd_mean, dd_std,
        [zcr_mean, zcr_std], [centroid_mean, centroid_std],
        [flatness_mean, flatness_std], [bandwidth_mean, bandwidth_std]
    ])

def main():
    index_files = [
        ROOT / "processed" / "whistle_index.csv",
        ROOT / "processed" / "match_negative_index.csv",
        ROOT / "processed" / "iphone_negative_index.csv",
        ROOT / "processed" / "iphone_whistle_index.csv",
    ]
    dfs = [pd.read_csv(f) for f in index_files if f.exists()]
    if not dfs: return
    
    full_index = pd.concat(dfs, ignore_index=True)
    feature_rows = []

    for _, row in full_index.iterrows():
        subfolder = "whistle" if row["label"] == 1 else "non_whistle"
        clip_path = CLIPS_DIR / subfolder / row["filename"]
        if not clip_path.exists(): continue

        y, sr = sf.read(clip_path)
        if len(y.shape) > 1: y = np.mean(y, axis=1)
        y_norm = normalize(y)
        
        feat_base = compute_ultimate_features(y_norm, sr)
        feature_rows.append({
            "match_id": row["match_id"], "filename": row["filename"],
            "label": row["label"], "source": row["source"],
            **{f"mfcc_{i}": v for i, v in enumerate(feat_base)},
        })

        if row["match_id"] not in TEST_MATCHES:
            feat_shifted = compute_ultimate_features(time_shift(y_norm, sr), sr)
            feature_rows.append({
                "match_id": row["match_id"], "filename": f"shift_{row['filename']}",
                "label": row["label"], "source": f"{row['source']}_shifted",
                **{f"mfcc_{i}": v for i, v in enumerate(feat_shifted)},
            })
            feat_noisy = compute_ultimate_features(add_noise(y_norm), sr)
            feature_rows.append({
                "match_id": row["match_id"], "filename": f"noise_{row['filename']}",
                "label": row["label"], "source": f"{row['source']}_noisy",
                **{f"mfcc_{i}": v for i, v in enumerate(feat_noisy)},
            })

    feat_df = pd.DataFrame(feature_rows)
    out_csv = ROOT / "processed" / "features.csv"
    feat_df.to_csv(out_csv, index=False)
    print(f"Matrix saved -> {out_csv} ({len(feat_df)} rows, 86 dimensions)")

if __name__ == "__main__":
    main()