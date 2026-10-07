# 🤖 RoboTwin 2.0

<p align="center">
  <strong>Complete HIVE-Bench workflow for RoboTwin data, training, and evaluation.</strong>
</p>

<p align="center">
  <a href="../../README.md">← HIVE-Bench</a> ·
  <a href="../../Analyze/README.md">🔬 Representation Analysis</a>
</p>

This guide is self-contained. Run every HIVE-Bench command from the repository root unless a step explicitly says otherwise.

The RoboTwin recipes use three RGB streams—head, left wrist, and right wrist—and a 14-D dual-arm joint action. RoboTwin itself remains an external repository with its own simulator environment.

## 🧭 Workflow

1. Download the HIVE-Bench RoboTwin demonstrations.
2. Convert `demo_clean` and/or `demo_randomized` to LeRobot format.
3. Select a policy framework and train it in the `hivebench` environment.
4. Install and verify the official RoboTwin environment.
5. Patch the external RoboTwin launcher so it forwards the checkpoint path.
6. Run the 12-task batch evaluator or the manual single-task workflow.

## 📦 1. Download the dataset

The RoboTwin demonstrations are published as the `RoboTwin_data/` folder in the [HIVE-Bench Hugging Face repository](https://huggingface.co/datasets/zhengtu666/HIVE-Bench-Data/tree/main). Download the folder directly with the Hugging Face CLI:

```bash
pip install -U huggingface_hub

hf download zhengtu666/HIVE-Bench-Data --repo-type dataset \
  --include "RoboTwin_data/**" --local-dir playground/Datasets
```

After the download, the task directories should be available directly under:

```text
playground/Datasets/RoboTwin_data/
├── adjust_bottle/
├── beat_block_hammer/
├── click_alarmclock/
└── ...
```

## 🔄 2. Convert to LeRobot format

Convert the randomized demonstrations:

```bash
python Policy/hivebench/dataloader/convert_robotwin_to_lerobot_headcam.py \
  --src_root playground/Datasets/RoboTwin_data \
  --dst_root playground/RoboTwin_LeRobot_HeadCam \
  --split Randomized
```

Convert the clean demonstrations when needed:

```bash
python Policy/hivebench/dataloader/convert_robotwin_to_lerobot_headcam.py \
  --src_root playground/Datasets/RoboTwin_data \
  --dst_root playground/RoboTwin_LeRobot_HeadCam \
  --split Clean
```

| Converter argument | Meaning |
| --- | --- |
| `--src_root` | Directory containing the raw RoboTwin task folders. |
| `--dst_root` | Parent directory for the generated LeRobot dataset. |
| `--split` | `Randomized` reads `demo_randomized`; `Clean` reads `demo_clean`. |
| `--tasks TASK ...` | Optional task subset. Omit it to convert every task found under `src_root`. |

Converted episodes contain Parquet trajectory data, MP4 observations, task metadata, and these camera mappings:

| Raw camera | HIVE-Bench feature |
| --- | --- |
| `cam_high` | `observation/head_camera/rgb` |
| `cam_left_wrist` | `observation/left_camera/rgb` |
| `cam_right_wrist` | `observation/right_camera/rgb` |

If you choose another output location, update `data_root_dir` in the selected YAML under `Bench/Robotwin/train_files/`.

## 🏋️ 3. Train a policy

All launchers use Accelerate with DeepSpeed ZeRO-2. Inspect the selected shell script before running it and make the GPU count consistent between `CUDA_VISIBLE_DEVICES` and `--num_processes`.

### Choose single-task or multi-task training

The RoboTwin YAML files use the `robotwin_all` dataset mixture defined in [`mixtures.py`](../../Policy/hivebench/dataloader/gr00t_lerobot/mixtures.py). Each active tuple selects one converted task and split:

```python
"robotwin_all": [
    ("Randomized/adjust_bottle", 1.0, "robotwin"),
]
```

- **Single-task training:** leave exactly one task tuple active.
- **Multi-task training:** leave all and only the tasks you want active; comment out the rest or remove them from the list.
- Use the `Randomized/` prefix for converted `demo_randomized` data and `Clean/` for converted `demo_clean` data. The selected directory must exist under the YAML's `data_root_dir`.

The same `robotwin_all` mixture is used by the three released RoboTwin training configurations. Changing it therefore changes the training task set for `VisionGR00T`, `VisionCLIPGR00T`, and `VLMVisionGR00T`. Use a new `run_id` for every task selection. Evaluation tasks are selected separately with `eval_12_tasks.sh --tasks` or the first argument to `eval.sh`.

The three training entrypoints and their matching YAML files are:

| Framework | Launcher (`train_files/`) | Configuration (`train_files/`) |
| --- | --- | --- |
| `VisionGR00T` | [run_vision_robotwin_train.sh](train_files/run_vision_robotwin_train.sh) | [vision_robotwin.yaml](train_files/vision_robotwin.yaml) |
| `VisionCLIPGR00T` | [run_vision_clip_robotwin.sh](train_files/run_vision_clip_robotwin.sh) | [vision_clip_robotwin.yaml](train_files/vision_clip_robotwin.yaml) |
| `VLMVisionGR00T` | [run_vlm_robotwin.sh](train_files/run_vlm_robotwin.sh) | [vlm_robotwin.yaml](train_files/vlm_robotwin.yaml) |

### Pure visual encoder: `VisionGR00T`

```bash
bash Bench/Robotwin/train_files/run_vision_robotwin_train.sh
```

Choose another visual encoder and a deterministic data subset:

```bash
VISION_MODEL=dinov3_base \
NUM_TRAJECTORIES=100 \
TRAJECTORY_SUBSET_SEED=42 \
bash Bench/Robotwin/train_files/run_vision_robotwin_train.sh
```

The first positional argument also overrides the trajectory count:

```bash
bash Bench/Robotwin/train_files/run_vision_robotwin_train.sh 100
```

### Visual encoder + CLIP text: `VisionCLIPGR00T`

```bash
bash Bench/Robotwin/train_files/run_vision_clip_robotwin.sh
```

```bash
VISION_MODEL=vjepa2.1_base \
NUM_TRAJECTORIES=100 \
TRAJECTORY_SUBSET_SEED=42 \
bash Bench/Robotwin/train_files/run_vision_clip_robotwin.sh
```

### VLM visual tokens + CLIP text: `VLMVisionGR00T`

```bash
BASE_VLM=playground/Pretrained_models/Qwen3-VL-4B-Instruct \
bash Bench/Robotwin/train_files/run_vlm_robotwin.sh
```

To train with VLM layer 16, add this line to the `accelerate launch` arguments in the shell launcher:

```bash
  --framework.vision_text_fusion.vlm_layer_idx 16 \
```

The released RoboTwin VLM launcher already includes this line. Change `16` to another layer index for a layer sweep; remove the line and set the YAML value to `-1` to use the final layer.

### Training controls

| Variable / setting | Used by | Meaning |
| --- | --- | --- |
| `VISION_MODEL` | VisionGR00T, VisionCLIPGR00T | Encoder shorthand or supported model ID. |
| `BASE_VLM` | VLMVisionGR00T | Local VLM directory or supported model ID. |
| `NUM_TRAJECTORIES` | Supported launchers | Maximum trajectories selected from each configured task. A positional count takes priority where supported. |
| `TRAJECTORY_SUBSET_SEED` | Supported launchers | Seed for deterministic trajectory selection. |
| `RUN_ROOT_DIR` | VLMVisionGR00T launcher | Checkpoint/output root. Other launchers currently define this in the shell script. |
| `data_root_dir` | Training YAML | Converted dataset root. Edit the selected YAML if your dataset is elsewhere. |
| `run_id` | Shell launcher or YAML | Experiment directory name. Set a unique value for each run. |
| `CUDA_VISIBLE_DEVICES` | Shell launcher | Training GPUs. Keep it consistent with Accelerate `--num_processes`. |

The complete encoder alias catalog is in the [root README](../../README.md#policy).

## 🧰 4. Install RoboTwin

First follow the [official RoboTwin installation guide](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) and verify the [official RoboTwin repository](https://github.com/RoboTwin-Platform/RoboTwin) in its own `robotwin` Conda environment.

Complete RoboTwin's official smoke test before connecting it to HIVE-Bench. HIVE-Bench does not replace the RoboTwin simulator, assets, renderer, or environment setup.

Clone RoboTwin wherever you keep third-party repositories:

```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin.git /path/to/RoboTwin
export ROBOTWIN_PATH=/path/to/RoboTwin
```

`ROBOTWIN_PATH` must be the repository root containing `script/eval_policy.py`. It is not a Conda environment, dataset, or checkpoint path.

### Prepare both runtime environments

Install HIVE-Bench in the policy environment:

```bash
conda activate hivebench
cd /path/to/HIVE-Bench
pip install -r requirements.txt
pip install -e .
```

Install the lightweight HIVE-Bench evaluation-client requirements in the already working RoboTwin environment:

```bash
conda activate robotwin
cd /path/to/HIVE-Bench
pip install -r Bench/Robotwin/eval_files/requirements.txt
```

Record an interpreter from each environment. The batch launcher starts both processes, so activating only one Conda environment is not enough:

```bash
# Run 'which python' inside each environment and place the results here.
export HIVEBENCH_PYTHON=/path/to/miniconda/envs/hivebench/bin/python
export ROBOTWIN_PYTHON=/path/to/miniconda/envs/robotwin/bin/python
export ROBOTWIN_PATH=/path/to/RoboTwin
```

| Variable | Purpose |
| --- | --- |
| `ROBOTWIN_PATH` | Root of the external RoboTwin checkout. |
| `HIVEBENCH_PYTHON` | Python executable that imports HIVE-Bench and loads the policy checkpoint. |
| `ROBOTWIN_PYTHON` | Python executable from the official RoboTwin environment. |

The launchers do not run `pip install` automatically.

## 🩹 5. Patch the external RoboTwin launcher

RoboTwin is maintained separately, so this patch is documented here instead of being vendored into HIVE-Bench. Apply it to `$ROBOTWIN_PATH/script/eval_policy.py` so the simulator accepts and forwards `--policy_ckpt_path`:

```diff
diff --git a/script/eval_policy.py b/script/eval_policy.py
index eded198..9fb36e3 100644
--- a/script/eval_policy.py
+++ b/script/eval_policy.py
@@ -69,6 +69,7 @@ def main(usr_args):
     # checkpoint_num = usr_args['checkpoint_num']
     policy_name = usr_args["policy_name"]
     instruction_type = usr_args["instruction_type"]
+    policy_ckpt_path = usr_args["policy_ckpt_path"]
     save_dir = None
     video_save_dir = None
     video_size = None
@@ -81,6 +82,7 @@ def main(usr_args):
     args['task_name'] = task_name
     args["task_config"] = task_config
     args["ckpt_setting"] = ckpt_setting
+    args["policy_ckpt_path"] = policy_ckpt_path

     embodiment_type = args.get("embodiment")
     embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
@@ -327,11 +329,13 @@ def eval_policy(task_name,
 def parse_args_and_config():
     parser = argparse.ArgumentParser()
     parser.add_argument("--config", type=str, required=True)
+    parser.add_argument("--policy_ckpt_path", type=str, required=True)
     parser.add_argument("--overrides", nargs=argparse.REMAINDER)
     args = parser.parse_args()

     with open(args.config, "r", encoding="utf-8") as f:
         config = yaml.safe_load(f)
+    config["policy_ckpt_path"] = args.policy_ckpt_path

     # Parse overrides
     def parse_override_pairs(pairs):
```

The HIVE-Bench launcher passes `--policy_ckpt_path` at runtime. Without this patch, RoboTwin cannot forward the checkpoint path into `model2robotwin_interface.py`. `eval.sh` checks the external file before launch and stops with a clear error when the patch is missing.

## 🎯 6. Recommended: 12-task batch evaluation

`eval_12_tasks.sh` starts policy servers, waits for readiness, schedules tasks across disjoint server/simulator GPU pairs, streams success-rate updates, restarts each server between tasks, writes summaries, and cleans up child processes on exit.

```bash
bash Bench/Robotwin/eval_files/eval_12_tasks.sh \
  playground/Checkpoints/<run>/pytorch_model.pt \
  --server-gpus 0,1 \
  --eval-gpus 2,3 \
  --base-port 5555
```

### Parameters

| Argument | Default | Meaning |
| --- | --- | --- |
| `<checkpoint>` | required | Checkpoint file, or a directory containing `pytorch_model.pt`. |
| `--server-gpus` | `2` | Comma-separated GPUs that host policy servers. |
| `--eval-gpus` | `5` | Comma-separated GPUs that run RoboTwin simulators. |
| `-n, --name` | `main_test_v1` | Value forwarded to RoboTwin as `ckpt_setting` for result naming. |
| `-p, --base-port` | `5555` | First port considered for policy workers. |
| `--server-timeout` | `600` | Seconds to wait for each policy server. |
| `--log-dir` | generated | Explicit output directory. |
| `--tasks` | built-in 12 | Comma-separated task subset. |

The GPU lists must be the same length and cannot overlap. Entries pair by position: `--server-gpus 0,1 --eval-gpus 2,3` creates `0 → 2` and `1 → 3`.

The output directory contains `all_runs.tsv`, `summary.tsv`, one simulator log per run, and one policy-server log per task/worker.

## 🧪 7. Manual single-task evaluation

Start the policy server in the HIVE-Bench environment:

```bash
conda activate hivebench

bash Bench/Robotwin/eval_files/run_policy_server.sh \
  playground/Checkpoints/<run>/pytorch_model.pt 0 5694
```

Positional arguments are:

1. checkpoint file or directory;
2. server GPU, default `ROBOTWIN_SERVER_GPU` or `5`;
3. server port, default `ROBOTWIN_SERVER_PORT` or `5694`.

The launcher does not force bfloat16. Set `ROBOTWIN_USE_BF16=1` only when you deliberately want bf16 serving and the selected model and GPU support it. `HIVEBENCH_OFFLINE=1` enables Hugging Face offline mode.

In a second terminal, start RoboTwin:

```bash
conda activate robotwin
export ROBOTWIN_PATH=/path/to/RoboTwin
export ROBOTWIN_PYTHON=python

bash Bench/Robotwin/eval_files/eval.sh \
  adjust_bottle demo_clean local_run 0 1 \
  playground/Checkpoints/<run>/pytorch_model.pt 5694
```

`eval.sh` positional arguments are:

1. task name, such as `adjust_bottle`;
2. task configuration: `demo_clean` or `demo_randomized`;
3. `ckpt_setting` label used by RoboTwin for result naming;
4. evaluation seed;
5. simulator GPU;
6. checkpoint used to recover normalization metadata;
7. optional policy-server port, default `5694`;
8. optional policy-server host, default `127.0.0.1`.

The checkpoint and port must match the policy server. Keep the server and simulator on different GPUs.

The client template is `Bench/Robotwin/eval_files/deploy_policy.yml`. It controls `host`, `port`, `unnorm_key`, `action_mode`, `normalization_mode`, `use_state`, and the optional `max_joint_delta` guard. `eval.sh` creates a temporary copy and replaces the host/port for each run; it does not modify the tracked template.

## 🧯 Troubleshooting

- **`ROBOTWIN_PATH` is rejected:** confirm `$ROBOTWIN_PATH/script/eval_policy.py` exists.
- **Missing `policy_ckpt_path`:** apply the patch above to the external RoboTwin checkout.
- **Server timeout:** inspect the generated server log, confirm the checkpoint loads with `HIVEBENCH_PYTHON`, and increase `--server-timeout` for large backbones.
- **Port already occupied:** choose another `--base-port`. The batch launcher refuses to connect to an unknown stale server.
- **CUDA out of memory:** reduce worker pairs or use a smaller encoder. You may opt into bf16 with `ROBOTWIN_USE_BF16=1` when the selected model and GPU support it.
- **Dataset not found:** update `data_root_dir` in the selected training YAML; do not add machine-specific absolute paths to shared configs.

## Training recipe configuration

The matching shell launcher and YAML in `Bench/Robotwin/train_files/` define each training recipe. Edit the paired YAML to change the model configuration, per-device batch size, training steps, save interval, optimizer settings, and other training hyperparameters. Use the shell launcher for run-specific values such as the selected model path, GPUs, dataset root, and run ID.
