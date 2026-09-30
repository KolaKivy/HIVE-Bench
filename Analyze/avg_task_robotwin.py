import os
import json
import numpy as np
from collections import OrderedDict

BASE_DIR = "Analyze/outputs/DepthVLM-4B_robotwin_new"

TASK_DIRS = [
    "adjust_bottle",
    "click_alarmclock",
    "lift_pot",
    "open_laptop",
    "place_can_basket",
    "stamp_seal",
    "beat_block_hammer",
    "handover_block",
    "move_playingcard_away",
    "place_burger_fries",
    "rotate_qrcode",
    "turn_switch",
]



CORE_METRICS = OrderedDict([
    ("avg_token_cos", ["mean_avg_cos_values"]),
    ("dist_sim_decay", []),  
    ("mean_token_norm", ["mean_mean_norms"]),
    ("neighbor_sim", ["mean_neighbor_sim_values"]),
    ("token_cov_rank", ["mean_eff_ranks"]),
    ("token_norm_entropy", ["mean_entropies"]),
    ("token_norm_var", ["mean_var_norms"]),
    ("token_to_global", ["mean_mean_sims"]),
    ("frequency_metrics", [
        "mean_low_ratio",
        "mean_mid_ratio",
        "mean_high_ratio",
        "mean_spectral_entropy",
        "mean_spectral_centroid",
        "mean_spectral_bandwidth",
    ]),
])

SKIP_ANALYSES = {
    "dist_sim_decay",  
}


FREQ_DERIVED_METRICS = OrderedDict([
    ("high_low_ratio", "High-frequency energy divided by low-frequency energy; larger values emphasize detail."),
    ("high_mid_ratio", "High-frequency energy divided by mid-frequency energy."),
    ("mid_low_ratio", "Mid-frequency energy divided by low-frequency energy."),
    ("high_freq_emphasis", "High-frequency share of non-low-frequency energy."),
    ("band_balance", "Low/high-frequency balance; 1 means equal energy."),
    ("centroid_norm", "Normalized spectral centroid from low (0) to high (1) frequency."),
    ("bandwidth_centroid_ratio", "Spectral bandwidth divided by centroid; larger values indicate wider spread."),
    ("entropy_norm", "Normalized spectral entropy; larger values indicate flatter energy distribution."),
])

EPS = 1e-12


def _get_float(d, key):
    v = d.get(key, None)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return None


def compute_frequency_derived(freq_data):
    'Compute frequency derived function.'
    derived = {}

    low = _get_float(freq_data, "mean_low_ratio")
    mid = _get_float(freq_data, "mean_mid_ratio")
    high = _get_float(freq_data, "mean_high_ratio")
    entropy = _get_float(freq_data, "mean_spectral_entropy")
    centroid = _get_float(freq_data, "mean_spectral_centroid")
    bandwidth = _get_float(freq_data, "mean_spectral_bandwidth")

    if low is not None and high is not None:
        derived["high_low_ratio"] = high / (low + EPS)
        derived["band_balance"] = 1.0 - abs(low - high)

    if high is not None and mid is not None:
        derived["high_mid_ratio"] = high / (mid + EPS)

    if mid is not None and low is not None:
        derived["mid_low_ratio"] = mid / (low + EPS)

    if high is not None and low is not None and mid is not None:
        derived["high_freq_emphasis"] = high / (low + mid + high + EPS)

    if centroid is not None:
        derived["centroid_norm"] = centroid

    if bandwidth is not None and centroid is not None:
        derived["bandwidth_centroid_ratio"] = bandwidth / (centroid + EPS)

    if entropy is not None:
        derived["entropy_norm"] = entropy

    return derived


def extract_core_metrics(analyses):
    'Extract core metrics function.'
    result = {}

    for analysis_name, metrics in CORE_METRICS.items():
        if analysis_name in SKIP_ANALYSES:
            continue
        if analysis_name not in analyses:
            continue

        analysis_data = analyses[analysis_name]
        if not isinstance(analysis_data, dict):
            continue

        for metric in metrics:
            if metric in analysis_data:
                value = analysis_data[metric]
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    key = f"{analysis_name}.{metric}"
                    result[key] = float(value)

        
        if analysis_name == "frequency_metrics":
            derived = compute_frequency_derived(analysis_data)
            for dk, dv in derived.items():
                result[f"frequency_metrics.{dk}"] = float(dv)

    return result


def main():
    all_metrics = []

    for task in TASK_DIRS:
        json_path = os.path.join(BASE_DIR, task, "all_videos_summary.json")
        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found, skip.")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "analyses" not in data:
            print(f"Warning: {json_path} has no 'analyses' field, skip.")
            continue

        metrics = extract_core_metrics(data["analyses"])
        if metrics:
            all_metrics.append(metrics)
        else:
            print(f"Warning: {json_path} has no valid core metrics.")

    if len(all_metrics) == 0:
        raise ValueError("No valid JSON files found.")

    
    ordered_keys = []
    for analysis_name, metrics in CORE_METRICS.items():
        if analysis_name in SKIP_ANALYSES:
            continue
        for metric in metrics:
            ordered_keys.append(f"{analysis_name}.{metric}")
    for dk in FREQ_DERIVED_METRICS:
        ordered_keys.append(f"frequency_metrics.{dk}")

    avg_metrics = OrderedDict()
    sum_metrics = OrderedDict()
    count_metrics = OrderedDict()

    for key in ordered_keys:
        values = [m[key] for m in all_metrics if key in m]
        if values:
            avg_metrics[key] = float(np.mean(values))
            sum_metrics[key] = float(np.sum(values))
            count_metrics[key] = len(values)

    output = OrderedDict([
        ("task_count", len(all_metrics)),
        ("tasks", TASK_DIRS),
        ("averaged_core_metrics", avg_metrics),
        ("summed_core_metrics", sum_metrics),   
        ("metric_counts", count_metrics),
        ("by_analysis", OrderedDict()),
        ("frequency_derived_explanation", FREQ_DERIVED_METRICS),
    ])

    for analysis_name, metrics in CORE_METRICS.items():
        if analysis_name in SKIP_ANALYSES:
            continue
        output["by_analysis"][analysis_name] = OrderedDict()

        for metric in metrics:
            key = f"{analysis_name}.{metric}"
            if key in avg_metrics:
                output["by_analysis"][analysis_name][metric] = avg_metrics[key]

        if analysis_name == "frequency_metrics":
            for dk in FREQ_DERIVED_METRICS:
                key = f"frequency_metrics.{dk}"
                if key in avg_metrics:
                    output["by_analysis"][analysis_name][dk] = avg_metrics[key]

    output_path = os.path.join(BASE_DIR, "avg_core_metrics.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved averaged core metrics to: {output_path}")
    print("\nAveraged core metrics (in order):")
    for key, value in avg_metrics.items():
        print(f"  {key}: {value:.6f}")


if __name__ == "__main__":
    main()