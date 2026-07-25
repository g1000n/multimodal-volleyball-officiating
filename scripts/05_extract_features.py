"""
Step 5: Master Feature Extractor (92 Dimensions)

CONSOLIDATED: merges the physics-based feature set (bandpass filter, HNR,
whistle-band energy ratio, peak concentration, pitch stability -- designed to
distinguish whistle tonal bursts from ball hits/squeaks/court noise) into the
working index-CSV pipeline (reads processed/clips/ via whistle_index.csv etc,
same as before). Do not run any other 05_*.py variant that scans data/raw/ --
that pipeline never reads your actual extracted clips and is not connected to
06_train_model.py's expected column format.

Features (92 dims):
  - MFCC Mean & Std (26)
  - MFCC Delta Mean & Std (26)
  - MFCC Delta-Delta Mean & Std (26)
  - Zero-Crossing Rate Mean & Std (2)
  - Spectral Centroid Mean & Std (2)
  - Spectral Flatness Mean & Std (2)
  - Spectral Bandwidth Mean & Std (2)
  - Whistle-band (2-4.5kHz) Energy Ratio Mean & Std (2)
  - Peak Energy Concentration Mean & Std (2)
  - Dominant Pitch Stability (1)
  - Harmonic-to-Noise Ratio (1)

All audio is bandpass filtered (1800-4800 Hz) before analysis -- strips out
low-frequency ball-hit/footstep rumble and very high hiss before any feature
is computed, not just relying on the classifier to learn it.
"""
from pathlib import Path
import numpy as np
import soundfile as sf
import pandas as pd
from scipy.fftpack import dct
from scipy.signal import butter, sosfilt

ROOT = Path(__file__).parent.parent
CLIPS_DIR = ROOT / "processed" / "clips"
SR = 22050
N_MFCC = 13
TEST_MATCHES = ["match9", "match7", "match13", "iphone_positive_audio"]

SOS_BANDPASS = butter(4, [1800, 4800], btype='bandpass', fs=SR, output='sos')


def apply_bandpass(y):
    return sosfilt(SOS_BANDPASS, y)


def normalize(y):
    max_abs = np.max(np.abs(y))
    if max_abs < 1e-4:
        return y
    return y / (max_abs + 1e-9)


def time_shift(y, sr, shift_sec=0.05):
    return np.roll(y, int(shift_sec * sr))


def add_noise(y, noise_factor=0.001):
    return y + noise_factor * np.random.normal(0, 1, len(y))


def scale_volume(y, factor):
    return y * factor


def compute_hnr(signal, sr=SR):
    """FIX: was using np.correlate(mode='full'), an O(n^2) full autocorrelation
    -- extremely slow (a full billion+ operations per clip). We only ever need
    a handful of lags (1000-5000 Hz pitch range), so compute just those directly."""
    if np.max(np.abs(signal)) < 1e-5:
        return 0.0
    min_lag, max_lag = int(sr / 5000), int(sr / 1000)
    if max_lag <= min_lag or max_lag >= len(signal):
        return 0.0
    r0 = float(np.dot(signal, signal))
    if r0 <= 0:
        return 0.0
    r_max = 0.0
    for lag in range(min_lag, max_lag):
        r = float(np.dot(signal[:-lag], signal[lag:]))
        if r > r_max:
            r_max = r
    if r_max <= 0 or r_max >= r0:
        return 0.0
    hnr = 10 * np.log10(r_max / (r0 - r_max + 1e-9))
    return float(np.clip(hnr, -20.0, 40.0))


def compute_advanced_features(raw_y, sr=SR, n_mfcc=N_MFCC, n_fft=512, hop_length=256, n_mels=40):
    """Extracts exactly 92 features. This exact function is duplicated in
    08_realtime_ui.py -- if you change math here, copy the change there too,
    or training and live inference will silently disagree."""
    y = apply_bandpass(raw_y)
    window = np.hanning(n_fft)
    frames = [y[i:i + n_fft] * window for i in range(0, len(y) - n_fft + 1, hop_length)]

    if not frames:
        return np.zeros(92)

    frames = np.array(frames)
    stft_matrix = np.fft.rfft(frames, n=n_fft, axis=-1)
    magnitude_spectrum = np.abs(stft_matrix)
    power_spectrum = magnitude_spectrum ** 2

    low_mel, high_mel = 0, 2595 * np.log10(1 + (sr / 2) / 700)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((n_mels, int(n_fft / 2 + 1)))
    for m in range(1, n_mels + 1):
        f_m_minus, f_m, f_m_plus = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        for k in range(f_m_minus, f_m):
            if (f_m - f_m_minus) > 0:
                fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if (f_m_plus - f_m) > 0:
                fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    filter_banks = np.dot(power_spectrum, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)

    mfcc = dct(filter_banks, type=2, axis=-1, norm='ortho')[:, :n_mfcc]
    mfcc_mean, mfcc_std = np.mean(mfcc, axis=0), np.std(mfcc, axis=0)

    if len(mfcc) > 1:
        deltas = np.diff(mfcc, axis=0)
        delta_mean, delta_std = np.mean(deltas, axis=0), np.std(deltas, axis=0)
    else:
        deltas = np.zeros((1, n_mfcc))
        delta_mean, delta_std = np.zeros(n_mfcc), np.zeros(n_mfcc)

    if len(deltas) > 1:
        delta_deltas = np.diff(deltas, axis=0)
        dd_mean, dd_std = np.mean(delta_deltas, axis=0), np.std(delta_deltas, axis=0)
    else:
        dd_mean, dd_std = np.zeros(n_mfcc), np.zeros(n_mfcc)

    zcr_frames = [np.sum(np.abs(np.diff(np.sign(f)))) / 2 / len(f) for f in frames]
    zcr_mean, zcr_std = np.mean(zcr_frames), np.std(zcr_frames)

    freqs = np.fft.rfftfreq(n_fft, d=1 / sr)
    centroid_frames = []
    for mag in magnitude_spectrum:
        mag_sum = np.sum(mag)
        centroid_frames.append(np.sum(freqs * mag) / mag_sum if mag_sum > 0 else 0.0)
    centroid_mean, centroid_std = np.mean(centroid_frames), np.std(centroid_frames)

    flatness_frames = []
    for power_frame in power_spectrum:
        power_eps = power_frame + 1e-9
        log_mean = np.mean(np.log(power_eps))
        arith_mean = np.mean(power_eps)
        flatness_frames.append(np.exp(log_mean) / arith_mean if arith_mean > 0 else 0.0)
    flatness_mean, flatness_std = np.mean(flatness_frames), np.std(flatness_frames)

    bandwidth_frames = []
    for idx, power_frame in enumerate(power_spectrum):
        p_sum = np.sum(power_frame)
        if p_sum > 0:
            centroid = centroid_frames[idx]
            variance = np.sum(power_frame * (freqs - centroid) ** 2) / p_sum
            bandwidth_frames.append(np.sqrt(variance))
        else:
            bandwidth_frames.append(0.0)
    bandwidth_mean, bandwidth_std = np.mean(bandwidth_frames), np.std(bandwidth_frames)

    # Physical whistle discriminator features -- these are the ones that
    # directly target "does this sound like a narrow tonal whistle blast,
    # or a broadband thud/squeak" rather than relying only on MFCCs.
    whistle_mask = (freqs >= 2000) & (freqs <= 4500)
    whistle_ratio_frames, peak_concentration_frames, dom_pitch_bins = [], [], []

    for mag in magnitude_spectrum:
        total_e = np.sum(mag) + 1e-9
        band_mags = mag[whistle_mask]
        band_e = np.sum(band_mags)
        whistle_ratio_frames.append(band_e / total_e)

        if band_e > 0:
            peak_concentration_frames.append(np.max(band_mags) / band_e)
        else:
            peak_concentration_frames.append(0.0)

        dom_pitch_bins.append(np.argmax(mag))

    whistle_ratio_mean, whistle_ratio_std = np.mean(whistle_ratio_frames), np.std(whistle_ratio_frames)
    peak_conc_mean, peak_conc_std = np.mean(peak_concentration_frames), np.std(peak_concentration_frames)
    pitch_stability = np.std(dom_pitch_bins)
    hnr_value = compute_hnr(y, sr=sr)

    return np.concatenate([
        mfcc_mean, mfcc_std,
        delta_mean, delta_std,
        dd_mean, dd_std,
        [zcr_mean, zcr_std], [centroid_mean, centroid_std],
        [flatness_mean, flatness_std], [bandwidth_mean, bandwidth_std],
        [whistle_ratio_mean, whistle_ratio_std],
        [peak_conc_mean, peak_conc_std],
        [pitch_stability, hnr_value]
    ])


def main():
    index_files = [
        ROOT / "processed" / "whistle_index.csv",
        ROOT / "processed" / "match_negative_index.csv",
        ROOT / "processed" / "iphone_negative_index.csv",
        ROOT / "processed" / "iphone_whistle_index.csv",
        ROOT / "processed" / "synthetic_negative_index.csv",
    ]
    dfs = [pd.read_csv(f) for f in index_files if f.exists()]
    if not dfs:
        print("No index files found. Run steps 2-4 first.")
        return

    full_index = pd.concat(dfs, ignore_index=True)
    feature_rows = []

    for _, row in full_index.iterrows():
        subfolder = "whistle" if row["label"] == 1 else "non_whistle"
        clip_path = CLIPS_DIR / subfolder / row["filename"]
        if not clip_path.exists():
            continue

        y, sr = sf.read(clip_path)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
        y_norm = normalize(y)

        feat_base = compute_advanced_features(y_norm, sr)
        feature_rows.append({
            "match_id": row["match_id"], "filename": row["filename"],
            "label": row["label"], "source": row["source"],
            **{f"mfcc_{i}": v for i, v in enumerate(feat_base)},
        })

        if row["match_id"] not in TEST_MATCHES:
            # Expanded augmentation: more variety per raw clip, since we can't
            # record more raw negative sessions right now. Each variant nudges
            # the same real sound slightly differently so the model sees more
            # acoustic diversity from the same limited source material.
            variants = [
                ("shift_pos", time_shift(y_norm, sr, shift_sec=0.05)),
                ("shift_neg", time_shift(y_norm, sr, shift_sec=-0.05)),
                ("noise_light", add_noise(y_norm, noise_factor=0.001)),
                ("noise_heavy", add_noise(y_norm, noise_factor=0.004)),
                ("vol_quiet", scale_volume(y_norm, 0.6)),
                ("vol_loud", scale_volume(y_norm, 1.4)),
            ]
            for tag, variant_audio in variants:
                feat_variant = compute_advanced_features(variant_audio, sr)
                feature_rows.append({
                    "match_id": row["match_id"], "filename": f"{tag}_{row['filename']}",
                    "label": row["label"], "source": f"{row['source']}_{tag}",
                    **{f"mfcc_{i}": v for i, v in enumerate(feat_variant)},
                })

    feat_df = pd.DataFrame(feature_rows)
    out_csv = ROOT / "processed" / "features.csv"
    feat_df.to_csv(out_csv, index=False)
    print(f"Matrix saved -> {out_csv} ({len(feat_df)} rows, 92 dimensions)")


if __name__ == "__main__":
    main()