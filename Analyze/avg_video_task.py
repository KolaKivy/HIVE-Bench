import os
import json
import numpy as np
from collections import OrderedDict

BASE_DIR = "Analyze/outputs/DepthVLM-4B_video_level_new"

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
    ("within_between_var", ["variance_ratio"]),
    ("trajectory_var_ratio", ["trajectory_var_ratio"]),
])


SKIP_ANALYSES = {
    "lag_distance_curve",  
}


def extract_core_metrics(analyses):
    'Extract core metrics function.'
    result = {}
    for analysis_name, metrics in CORE_METRICS.items():
        if analysis_name in SKIP_ANALYSES:
            continue
        if analysis_name not in analyses:
            continue
        
        analysis_data = analyses[analysis_name]
        for metric in metrics:
            if metric in analysis_data:
                key = f"{analysis_name}.{metric}"
                value = analysis_data[metric]
                
                if isinstance(value, (int, float)):
                    result[key] = value
    return result


def main():
    all_metrics = []  

    
    for task in TASK_DIRS:
        json_path = os.path.join(BASE_DIR, task, "all_videos_summary.json")
        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found, skip.")
            continue

        with open(json_path, "r") as f:
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

    
    avg_metrics = OrderedDict()
    
    
    for analysis_name, metrics in CORE_METRICS.items():
        if analysis_name in SKIP_ANALYSES:
            continue
        for metric in metrics:
            key = f"{analysis_name}.{metric}"
            values = [m[key] for m in all_metrics if key in m]
            if values:
                avg_metrics[key] = float(np.mean(values))

    
    output = OrderedDict([
        ("task_count", len(all_metrics)),
        ("tasks", TASK_DIRS),
        ("averaged_core_metrics", avg_metrics),
        ("by_analysis", OrderedDict())
    ])

    
    for analysis_name, metrics in CORE_METRICS.items():
        if analysis_name in SKIP_ANALYSES:
            continue
        output["by_analysis"][analysis_name] = OrderedDict()
        for metric in metrics:
            key = f"{analysis_name}.{metric}"
            if key in avg_metrics:
                output["by_analysis"][analysis_name][metric] = avg_metrics[key]

    
    output_path = os.path.join(BASE_DIR, "avg_core_metrics.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved averaged core metrics to: {output_path}")
    print("\nAveraged core metrics (in order):")
    for key, value in avg_metrics.items():
        print(f"  {key}: {value:.6f}")


if __name__ == "__main__":
    main()