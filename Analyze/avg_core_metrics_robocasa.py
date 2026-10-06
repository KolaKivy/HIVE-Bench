import json
import os
import argparse
from collections import OrderedDict


TASK_DIRS = [
    "PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000",
    "PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_1000",
    "PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000",
    "PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_1000",
    "PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_1000",
    "PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_1000",
    "PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_1000",
    "PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000",
    "PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_1000",
    "PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_1000",
    "PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_1000",
    "PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000",
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
    ("within_between_var", ["variance_ratio"]),
    ("frequency_metrics", [
        "mean_low_ratio",
        "mean_mid_ratio",
        "mean_high_ratio",
        "mean_spectral_entropy",
        "mean_spectral_centroid",
        "mean_spectral_bandwidth",
    ]),
    ("temporal_smoothness", ["temporal_smoothness"]),
    ("temporal_cosine_shift", ["temporal_cosine_shift"]),
    ("lag_distance_curve", []),
    ("temporal_variance", ["mean_temporal_variance"]),
    ("temporal_effective_rank", ["temporal_effective_rank"]),
    ("temporal_spectral_entropy", ["temporal_spectral_entropy"]),
    ("autocorrelation", ["lag1_autocorrelation"]),
    ("total_trajectory_variation", ["normalized_total_trajectory_variation"]),
    ("patch_temporal_smoothness", ["mean_patch_temporal_smoothness"]),
    ("temporal_token_norm_entropy", ["mean_normalized_temporal_token_norm_entropy"]),
    ("trajectory_var_ratio", ["trajectory_var_ratio"]),
])

SKIP_ANALYSES = {
    "dist_sim_decay",
    "lag_distance_curve",
}

FREQ_DERIVED_METRICS = OrderedDict([
    ("high_low_ratio", "high frequency energy / low frequency energy"),
    ("high_mid_ratio", "high frequency energy / mid frequency energy"),
    ("mid_low_ratio", "mid frequency energy / low frequency energy"),
    ("high_freq_emphasis", "high frequency energy / total frequency energy"),
    ("band_balance", "low/high frequency balance, 1 means equal"),
    ("centroid_norm", "normalized spectral centroid"),
    ("bandwidth_centroid_ratio", "spectral bandwidth / spectral centroid"),
    ("entropy_norm", "normalized spectral entropy"),
])

EPS = 1e-12


def _get_float(data, key):
    value = data.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def compute_frequency_derived(freq_data):
    derived = OrderedDict()

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
    result = OrderedDict()
    missing = []

    for analysis_name, metric_names in CORE_METRICS.items():
        if analysis_name in SKIP_ANALYSES or not metric_names:
            continue

        analysis_data = analyses.get(analysis_name)
        if not isinstance(analysis_data, dict):
            missing.extend(f"{analysis_name}.{metric}" for metric in metric_names)
            if analysis_name == "frequency_metrics":
                missing.extend(f"frequency_metrics.{key}" for key in FREQ_DERIVED_METRICS)
            continue

        for metric in metric_names:
            value = analysis_data.get(metric)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                result[f"{analysis_name}.{metric}"] = float(value)
            else:
                missing.append(f"{analysis_name}.{metric}")

        if analysis_name == "frequency_metrics":
            derived = compute_frequency_derived(analysis_data)
            for metric in FREQ_DERIVED_METRICS:
                if metric in derived:
                    result[f"frequency_metrics.{metric}"] = float(derived[metric])
                else:
                    missing.append(f"frequency_metrics.{metric}")

    return result, missing


def average_metrics(all_metrics):
    ordered_keys = []
    for analysis_name, metric_names in CORE_METRICS.items():
        if analysis_name in SKIP_ANALYSES:
            continue

        for metric in metric_names:
            key = f"{analysis_name}.{metric}"
            if key not in ordered_keys:
                ordered_keys.append(key)

        if analysis_name == "frequency_metrics":
            for metric in FREQ_DERIVED_METRICS:
                key = f"frequency_metrics.{metric}"
                if key not in ordered_keys:
                    ordered_keys.append(key)

    averaged = OrderedDict()
    for key in ordered_keys:
        values = [metrics[key] for metrics in all_metrics if key in metrics]
        if values:
            averaged[key] = sum(values) / len(values)
    return averaged


def main(base_dir):
    all_metrics = []
    successful_tasks = []

    for task in TASK_DIRS:
        json_path = os.path.join(base_dir, task, "all_videos_summary.json")
        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found, skip.")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        analyses = data.get("analyses")
        if not isinstance(analyses, dict):
            print(f"Warning: {json_path} has no valid 'analyses' field, skip.")
            continue

        metrics, missing = extract_core_metrics(analyses)
        if missing:
            print(f"Missing metrics in {json_path}:")
            for metric in missing:
                print(f"  - {metric}")

        if metrics:
            all_metrics.append(metrics)
            successful_tasks.append(task)
        else:
            print(f"Warning: {json_path} has no successfully computed core metrics.")

    if not all_metrics:
        raise ValueError("No valid JSON files found.")

    averaged_metrics = average_metrics(all_metrics)
    if not averaged_metrics:
        raise ValueError("No core metrics were successfully computed.")

    by_analysis = OrderedDict()
    for key, value in averaged_metrics.items():
        analysis_name, metric_name = key.split(".", 1)
        by_analysis.setdefault(analysis_name, OrderedDict())[metric_name] = value

    output = OrderedDict([
        ("task_count", len(successful_tasks)),
        ("tasks", successful_tasks),
        ("averaged_core_metrics", averaged_metrics),
        ("by_analysis", by_analysis),
    ])

    output_path = os.path.join(base_dir, "avg_core_metrics.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved successfully computed averaged core metrics to: {output_path}")
    print("\nAveraged core metrics:")
    for key, value in averaged_metrics.items():
        print(f"  {key}: {value:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", help="Directory containing the per-task analysis outputs.")
    args = parser.parse_args()
    main(args.base_dir)
