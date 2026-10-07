# 🔬 HIVE-Bench Analysis

This guide covers the complete analysis suite: frozen-encoder probes, representation metrics, visualization, robustness tests, execution, and output formats. Benchmark training and simulator evaluation are documented in the [RoboTwin](../Bench/Robotwin/README.md) and [RoboCasa](../Bench/Robocasa_tabletop/README.md) guides.

## Representation probes

HIVE-Bench provides frozen-encoder representation probes through [train_idm_fdm.py](analyse/train_idm_fdm.py). This section documents supported benchmark and model combinations. Metric definitions and methodology are covered in the HIVE-Bench paper.

### Supported representation sources

The probe launcher supports the visual encoders constructed by `hivebench.model.framework.VisionGR00T._build_vision_encoder` and three visual-language models (VLMs).

| Representation source | Launcher identifiers |
|---|---|
| VisionGR00T visual encoders | Any supported VisionGR00T factory identifier, such as `dinov2_base`, `dinov3_base`, `clip`, `siglip`, `radio`, and `lingbot_large` |
| Qwen3-VL | `qwen3`, `qwen3_layer16` |
| DepthVLM | `depthvlm`, `depthvlm_layer16` |
| Xiaomi Robotics-1 | `xiaomi`, `xiaomi_layer16` |

For each VLM, the plain identifier uses the default visual representation and the `_layer16` identifier selects `vlm_layer_idx=16`. The trainable-probe launcher accepts all six VLM identifiers.

### Benchmark support

| Benchmark | Available probe modes |
|---|---|
| RoboCasa | `idm`, `fdm`, `state` |
| RoboTwin | `idm`, `fdm`, `state`, `object` |

`object` is specific to RoboTwin and is not available for RoboCasa.

### Run a probe

Run the default RoboCasa IDM probe:

```bash
bash Analyze/run_idm.sh
```

Set launcher variables inline or edit [run_idm.sh](run_idm.sh):

```bash
# RoboCasa: idm, fdm, or state
DATASET=robocasa MODE=state MODEL=dinov3_base bash Analyze/run_idm.sh

# RoboTwin: idm, fdm, state, or object
DATASET=robotwin MODE=object MODEL=qwen3_layer16 bash Analyze/run_idm.sh
```

`MODEL` chooses the frozen representation source, `DATASET` chooses the benchmark configuration, and `MODE` chooses a probe available for that benchmark. Set `EPOCHS`, `ENCODER_DIM`, `FINETUNED_ENCODER_CHECKPOINT`, data roots, and output locations in the launcher as needed. Shared probe hyperparameters, including batch size, are in [configs/idm_fdm_robocasa.yaml](configs/idm_fdm_robocasa.yaml) and [configs/idm_fdm_robotwin.yaml](configs/idm_fdm_robotwin.yaml).

### Outputs and reproducibility

Each run writes its checkpoint, `training_log.csv`, `results.json`, and `data_manifest.json` under `playground/Checkpoints/`. The manifest records the selected episodes and normalization statistics for that run.

## 📊 Video representation metrics

The video-metric workflow supports single-frame statistics, temporal statistics, and multi-video metrics. The updated `run.py` delegates checkpoint loading, video decoding, metric execution, and summary writing to the modules under [tools/](tools/).

### Run the 12-task batch

Run from the HIVE-Bench repository root:

```bash
bash Analyze/run_12task_robocasa.sh dinov3 robocasa_video_level "0 1"
bash Analyze/run_12task_robotwin.sh dinov3 robotwin_video_level "0 1"
```

| Argument | Meaning |
|---|---|
| `model_name` | Configuration name under `Analyze/configs/model/`, such as `dinov3`, `dinov2`, `clip`, `siglip`, `qwen3`, or `xiaomirobotics_layer16` |
| `addition_name` | Run label used in the output directory, such as `video_level` |
| `gpu_ids` | Quoted, space-separated CUDA IDs; one worker runs per listed GPU |

Tasks are assigned round-robin by worker index, so nonconsecutive GPU IDs are supported. The RoboCasa script reads `observation.images.ego_view` videos; the RoboTwin script reads `observation.images.cam_high` videos from the converted head-camera LeRobot dataset.

Edit `BASE_DIR` in each script for your dataset location and `TASK_DIRS` for the tasks to analyze. The defaults select 12 tasks, stride 5, and batch sizes of 64 (RoboCasa) or 32 (RoboTwin). Directory mode samples every fifth video in sorted order and processes its full duration at the selected frame stride; the launchers' `frame_start` and `frame_end` settings apply only to individual-file runs. Change `ANALYSIS` to choose the desired metrics.

### Metric coverage

| Group | Included analysis identifiers |
|---|---|
| Single-frame | `avg_token_cos`, `dist_sim_decay`, `mean_token_norm`, `neighbor_sim`, `token_cov_rank`, `token_norm_entropy`, `token_norm_var`, `token_to_global`, `frequency_metrics` |
| Multi-frame | `within_between_var` |
| Temporal | `temporal_smoothness`, `temporal_cosine_shift`, `lag_distance_curve`, `temporal_variance`, `temporal_effective_rank`, `temporal_spectral_entropy`, `autocorrelation`, `total_trajectory_variation`, `patch_temporal_smoothness`, `temporal_token_norm_entropy` |
| Multi-video | `trajectory_var_ratio` |

The default batch selects 21 analysis methods. Their JSON summaries contain multiple measurements, including frequency-band energy, entropy, centroid, and bandwidth; analysis-method counts and scalar-metric counts are therefore different.

Qwen3-VL, DepthVLM, and Xiaomi Robotics-1 now extract image-position features through the full VLM backbone rather than only the visual tower. Layer selection is configured with `layer_idx` in the model YAML; `qwen3_layer16` and `xiaomirobotics_layer16` have dedicated configurations. LingBot remains a visual encoder.

### Outputs and automatic task averaging

Each task writes its averaged video summary to:

```text
Analyze/outputs/<model>_<addition_name>/<data_name>/all_videos_summary.json
```

For RoboCasa, `data_name` omits the `gr1_unified.` prefix; for RoboTwin, it is the task name. Depending on the selected methods, the directory also contains PNG visualizations and additional JSON summaries. Console logs are under `Analyze/outputs/logs/`.

After the workers finish, each batch script automatically calls its corresponding `avg_core_metrics_*.py` helper and writes:

```text
Analyze/outputs/<model>_<addition_name>/avg_core_metrics.json
```

The final file records contributing tasks, averaged scalar metrics, and metrics grouped by analysis. Missing task files or metric fields are reported and skipped; a run with no valid metrics fails. Curve-only results from `dist_sim_decay` and `lag_distance_curve` remain in per-task outputs and are excluded from the scalar average. The helpers also compute eight derived frequency measurements, including high/low energy ratio and normalized entropy.

To rerun aggregation independently:

```bash
python Analyze/avg_core_metrics_robocasa.py Analyze/outputs/dinov3_robocasa_video_level
python Analyze/avg_core_metrics_robotwin.py Analyze/outputs/dinov3_robotwin_video_level
```

Use separate run labels when analyzing both datasets with the same model. If you change the task selection in a batch script, also update `TASK_DIRS` in its averaging helper.

### Manual single-task or single-video run

```bash
python Analyze/run.py \
  model=dinov3 \
  analysis=avg_token_cos \
  video_path=playground/videos/example.mp4 \
  addition_name=debug \
  data_name=example
```

`video_path` accepts a video file or a directory of videos. Optional controls include `frame_start`, `frame_end`, `stride`, `batch_size`, and `device=cpu`. Frame indices are zero-based; temporal analyses require at least two sampled frames.

## 🔮 Gaussian / sphere representation analysis

The Gaussian/sphere workflow is included in this release.

### Representation shape (Gaussian / sphere)

The implementation is in `analyse/representation_shape.py`. It uses the head-camera RoboTwin LeRobot dataset and the RoboCasa PhysicalAI dataset. Candidate videos come from up to 500 RoboTwin or 1,000 RoboCasa episodes per task, with a default frame stride of 10. The launcher caps sampled frames per task at 2,000 for RoboTwin and 500 for RoboCasa; set `POINTS_PER_TASK=0` to use all candidate frames, or a smaller positive value for a smoke test. Set `STRIDE` to change the sampling interval. Each view is pooled to 14x14 tokens and projected with a fixed JL matrix to 256D. Run one dataset at a time:

```bash
DATASET=robotwin bash Analyze/run_representation_shape.sh
DATASET=robocasa bash Analyze/run_representation_shape.sh
```
