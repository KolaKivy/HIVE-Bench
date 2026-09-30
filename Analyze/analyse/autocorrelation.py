import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_autocorrelation(pooled_features, max_lag=None):
    S = np.asarray(pooled_features, dtype=np.float32)
    if len(S) < 2:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

    centered = S - S.mean(axis=0, keepdims=True)
    denom = np.square(centered).sum(axis=0)
    valid = denom > 1e-12
    if not np.any(valid):
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

    centered = centered[:, valid]
    denom = denom[valid]
    max_possible_lag = len(S) - 1
    if max_lag is None:
        max_lag = max_possible_lag
    max_lag = min(int(max_lag), max_possible_lag)

    lags = np.arange(1, max_lag + 1, dtype=np.int32)
    autocorr = []
    for lag in lags:
        numerator = (centered[:-lag] * centered[lag:]).sum(axis=0)
        rho = numerator / denom
        autocorr.append(float(np.mean(rho)))

    return lags, np.asarray(autocorr, dtype=np.float32)


def summarize_autocorrelation(lags, autocorr):
    if len(autocorr) == 0:
        return {
            "lag1_autocorrelation": 0.0,
            "mean_autocorrelation": 0.0,
            "min_autocorrelation": 0.0,
            "half_life_lag": 0,
        }

    below_half = np.where(autocorr <= 0.5)[0]
    half_life_lag = int(lags[below_half[0]]) if len(below_half) else int(lags[-1])
    return {
        "lag1_autocorrelation": float(autocorr[0]),
        "mean_autocorrelation": float(autocorr.mean()),
        "min_autocorrelation": float(autocorr.min()),
        "half_life_lag": half_life_lag,
    }


def make_autocorrelation_figure(lags, autocorr, save_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lags, autocorr, marker="o", markersize=2, linewidth=1.2, color="steelblue")
    ax.axhline(0.5, color="darkorange", linestyle="--", linewidth=1, label="0.5")
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Mean channel autocorrelation")
    ax.set_title("Temporal Autocorrelation")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_autocorrelation_summary(lags, autocorr, num_frames, save_path):
    summary = summarize_autocorrelation(lags, autocorr)
    summary.update({
        "num_frames": int(num_frames),
        "num_lags": int(len(lags)),
        "lags": lags.tolist(),
        "autocorrelation": autocorr.tolist(),
    })
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)
