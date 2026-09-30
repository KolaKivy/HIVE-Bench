# 🔬 HIVE-Bench Analysis

This guide covers the complete analysis suite: frozen-encoder probes, representation metrics, visualization, robustness tests, execution, and output formats. Benchmark training and simulator evaluation remain in the repository root README.

## Representation probes

HIVE-Bench provides frozen-encoder representation probes through [train_idm_fdm.py](analyse/train_idm_fdm.py). This section documents supported benchmark and model combinations. Metric definitions and methodology are covered in the HIVE-Bench paper.

### Supported representation sources

The probe launcher supports the visual encoders constructed by `hivebench.model.framework.DinoGR00T._build_vision_encoder` and three visual-language models (VLMs).

| Representation source | Launcher identifiers |
|---|---|
| DinoGR00T visual encoders | Any supported DinoGR00T factory identifier, such as `dinov2_base`, `dinov3_base`, `clip`, `siglip`, `radio`, and `lingbot_large` |
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

## ✨ Features

- **Multiple Analysis Methods**: Support for both single-frame and temporal analysis
- **Hydra Configuration**: Easy configuration management with Hydra
- **Video-based Testing**: All analyses work with video frames (no image directory support)
- **Flexible Frame Selection**: Control frame range and stride
- **Batch Processing**: Efficient batch processing to avoid memory issues

## 🛠️ Installation

```bash
pip install hydra-core opencv-python torch transformers numpy matplotlib scikit-learn pillow
```

## 🚀 Usage

### Basic Command Structure

```bash
python run.py model=<model_name> analysis=<analysis_method> video_path=<path_to_video> [options]
```

### Available Models

- `dinov3` - DINOv3 (facebook/dinov3-vits16-pretrain-lvd1689m)
- `clip` - CLIP (openai/clip-vit-large-patch14)
- `sam` - Segment Anything Model
- `dinov2` - DINOv2
- `siglip` - SigLIP
- `vit` - Vision Transformer
- And more (see `configs/model/`)

### Single-Frame Analysis Methods

These methods analyze each frame independently and output averaged metrics:

```bash
# PCA visualization (only saves first frame)
python run.py model=dinov3 analysis=pca_vis video_path=playground/videos/example.mp4

# Average pairwise token cosine similarity
python run.py model=clip analysis=avg_token_cos video_path=playground/videos/example.mp4

# Distance-similarity decay curve
python run.py model=dinov3 analysis=dist_sim_decay video_path=playground/videos/example.mp4

# Mean token norm
python run.py model=sam analysis=mean_token_norm video_path=playground/videos/example.mp4

# Neighbor similarity
python run.py model=dinov3 analysis=neighbor_sim video_path=playground/videos/example.mp4

# Token covariance rank (effective rank)
python run.py model=clip analysis=token_cov_rank video_path=playground/videos/example.mp4

# Token norm entropy
python run.py model=dinov3 analysis=token_norm_entropy video_path=playground/videos/example.mp4

# Token norm variance
python run.py model=dinov3 analysis=token_norm_var video_path=playground/videos/example.mp4

# Token-to-global similarity
python run.py model=siglip analysis=token_to_global video_path=playground/videos/example.mp4
```

### Temporal Analysis Methods

These methods require multiple frames and analyze temporal dynamics:

```bash
# Temporal smoothness
python run.py model=dinov3 analysis=temporal_smoothness video_path=playground/videos/example.mp4

# Temporal cosine shift
python run.py model=clip analysis=temporal_cosine_shift video_path=playground/videos/example.mp4

# Lag-distance curve
python run.py model=dinov3 analysis=lag_distance_curve video_path=playground/videos/example.mp4

# Temporal variance
python run.py model=dinov3 analysis=temporal_variance video_path=playground/videos/example.mp4

# Temporal effective rank
python run.py model=dinov3 analysis=temporal_effective_rank video_path=playground/videos/example.mp4

# Temporal spectral entropy
python run.py model=dinov3 analysis=temporal_spectral_entropy video_path=playground/videos/example.mp4

# Autocorrelation
python run.py model=dinov3 analysis=autocorrelation video_path=playground/videos/example.mp4

# Total trajectory variation
python run.py model=clip analysis=total_trajectory_variation video_path=playground/videos/example.mp4

# Patch temporal smoothness
python run.py model=dinov3 analysis=patch_temporal_smoothness video_path=playground/videos/example.mp4

# Temporal token norm entropy
python run.py model=dinov3 analysis=temporal_token_norm_entropy video_path=playground/videos/example.mp4 num_bins=20
```

### Advanced Options

```bash
# Specify frame range
python run.py model=dinov3 analysis=temporal_smoothness video_path=playground/videos/example.mp4 frame_start=0 frame_end=100

# Use stride to sample frames
python run.py model=dinov3 analysis=temporal_variance video_path=playground/videos/example.mp4 stride=5

# Custom batch size (to manage memory)
python run.py model=dinov3 analysis=avg_token_cos video_path=playground/videos/example.mp4 batch_size=4

# Custom output directory
python run.py model=dinov3 analysis=pca_vis video_path=playground/videos/example.mp4 output_dir=./my_results

# Run on CPU
python run.py model=dinov3 analysis=temporal_smoothness video_path=playground/videos/example.mp4 device=cpu
```

## 📂 Output Structure

Results are saved in `output_dir/{model_name}_{analysis_name}/`:

### Single-Frame Analysis
```
output/dinov3_avg_token_cos/
├── frame_0000/
│   ├── reference.png
│   ├── avg_token_cos.png
│   └── summary.json
├── frame_0001/
│   ├── reference.png
│   ├── avg_token_cos.png
│   └── summary.json
├── ...
└── averaged_summary.json  # Averaged metrics across all frames
```

### Temporal Analysis
```
output/dinov3_temporal_smoothness/
├── temporal_smoothness.png  # Visualization
└── summary.json             # Detailed metrics
```

### PCA Visualization (Special Case)
```
output/dinov3_pca_vis/
├── frame_0000/
│   ├── reference.png
│   └── pca_vis.png          # Only first frame has PCA visualization
├── frame_0001/              # Other frames exist but no PCA
├── ...
└── averaged_summary.json
```

## ⚙️ Configuration

Edit `configs/config.yaml` for default settings:

```yaml
device: cuda
batch_size: 8
output_dir: ./output
video_path: null
frame_start: null
frame_end: null
stride: 1
analysis: null
num_bins: 16  # For temporal_token_norm_entropy
```

Select different models in `configs/model/*.yaml`.

## 🧩 Adding New Analysis Methods

1. Create a new Python file in `analyse/` directory
2. Implement the analysis functions following existing patterns
3. Import the module in `run.py`
4. Add the method name to either `single_frame_methods` or `temporal_methods` list

## 🎯 Key Design Principles

1. **Video-only input**: All analyses use frames extracted from videos
2. **Single method execution**: Each run executes only one analysis method
3. **PCA restriction**: PCA visualization only saves the initial frame
4. **Averaged metrics**: Single-frame methods output averaged metrics across all frames
5. **Hydra-based configuration**: Clean and flexible configuration management

## 💡 Examples

### Example 1: Compare different encoders on temporal smoothness
```bash
python run.py model=dinov3 analysis=temporal_smoothness video_path=test.mp4
python run.py model=clip analysis=temporal_smoothness video_path=test.mp4
python run.py model=sam analysis=temporal_smoothness video_path=test.mp4
```

### Example 2: Analyze specific time window
```bash
python run.py model=dinov3 analysis=autocorrelation video_path=test.mp4 frame_start=50 frame_end=150
```

### Example 3: Reduce memory usage for large videos
```bash
python run.py model=dinov3 analysis=dist_sim_decay video_path=long_video.mp4 batch_size=4 stride=2
```

## 📝 Notes

- All frame indices are 0-based
- If `frame_start` and `frame_end` are not specified, all frames are used
- Temporal analyses require at least 2 frames
- Batch processing helps manage GPU memory for large videos
- Results include both visualizations (PNG) and numerical summaries (JSON)


## Analysis layout and released workflows

The released analysis entrypoints live in `Analyze/analyse/`. Use [run_idm.sh](run_idm.sh) for the representation probes, [action_robustness.py](analyse/action_robustness.py) for static-texture robustness, and [run_representation_shape.sh](run_representation_shape.sh) for Gaussian and sphere representation-shape analysis.

The Gaussian/sphere representation-shape workflow is included in this open-source release.

### Representation shape (Gaussian / sphere)

The implementation is in `analyse/representation_shape.py`. It uses the head-camera Robotwin LeRobot dataset and the RoboCasa PhysicalAI dataset, samples every 10 frames, and uses all episodes (500 Robotwin / 1000 RoboCasa per task) and samples every 10th frame; `POINTS_PER_TASK` can cap samples for smoke tests. Each view is pooled to 14x14 tokens and projected with a fixed JL matrix to 256D. Run one dataset at a time:

```bash
DATASET=robotwin bash Analyze/run_representation_shape.sh
DATASET=robocasa bash Analyze/run_representation_shape.sh
```
