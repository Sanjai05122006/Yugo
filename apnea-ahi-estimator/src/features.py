import numpy as np
from scipy.signal import welch

def extract_features(segment, fs):
    feats = {}

    # Time-domain
    feats["mean"] = np.mean(segment)
    feats["std"] = np.std(segment)
    feats["var"] = np.var(segment)
    feats["min"] = np.min(segment)
    feats["max"] = np.max(segment)
    feats["ptp"] = np.ptp(segment)
    feats["rms"] = np.sqrt(np.mean(segment**2))
    feats["zcr"] = np.mean(np.diff(np.sign(segment)) != 0)

    # Frequency-domain (PSD)
    f, pxx = welch(segment, fs=fs, nperseg=1024)

    def bandpower(fmin, fmax):
        mask = (f >= fmin) & (f < fmax)
        return np.trapz(pxx[mask], f[mask])

    feats["bp_0_0p5"] = bandpower(0, 0.5)
    feats["bp_0p5_3"] = bandpower(0.5, 3)
    feats["bp_3_10"] = bandpower(3, 10)
    feats["bp_10_40"] = bandpower(10, 40)

    feats["spec_centroid"] = np.sum(f * pxx) / (np.sum(pxx) + 1e-8)

    return feats
