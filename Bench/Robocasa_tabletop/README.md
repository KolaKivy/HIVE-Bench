# 🏠 RoboCasa GR1 Tabletop

<p align="center">
  <strong>Complete HIVE-Bench workflow for RoboCasa GR1 Tabletop data, training, and evaluation.</strong>
</p>

<p align="center">
  <a href="../../README.md">← HIVE-Bench</a> ·
  <a href="../../Analyze/README.md">🔬 Representation Analysis</a>
</p>

This guide is self-contained. Run HIVE-Bench commands from the repository root.

The provided recipes use the GR1 tabletop tasks with a 29-D action, 58-D state metadata, one ego-view image stream, and a 16-step training action chunk. Evaluation connects a HIVE-Bench policy server to RoboCasa through WebSocket, allowing the simulator and policy to use separate environments.

## 🧭 Workflow

1. Download the 24 selected GR1 tabletop datasets.
2. Select a policy framework and train it in the `hivebench` environment.
3. Install and verify the official RoboCasa GR1 tabletop simulator environment.
4. For manual evaluation, start one policy server and run one simulator task.
5. For the full benchmark, run the 12-task batch launcher; it manages all policy servers automatically.

## 📦 1. Download the dataset

The downloader selects the 24 GR1 tabletop folders used by HIVE-Bench from NVIDIA's `PhysicalAI-Robotics-GR00T-X-Embodiment-Sim` dataset:

```bash
python Bench/Robocasa_tabletop/download_gr00t_robocasa_data.py
```

The default output root is:

```text
playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/
```

The script downloads the 24 released GR1 task folders and resumes files already present in the local cache.

If you store the data elsewhere, change `data_root_dir` in the selected YAML or set `DATA_ROOT_DIR` for a launcher that supports it. Do not leave a machine-specific absolute path in a shared YAML or shell script.

## 🏋️ 2. Train a policy

All launchers use Accelerate. Before training, inspect the selected shell script and keep `CUDA_VISIBLE_DEVICES` consistent with `--num_processes`.

### Choose single-task or multi-task training

The provided RoboCasa recipes use the `fourier_gr1_unified_1` dataset mixture defined in [`mixtures.py`](../../Policy/hivebench/dataloader/gr00t_lerobot/mixtures.py). Each tuple in that list is one training task:

```python
"fourier_gr1_unified_1": [
    ("gr1_unified.<task-folder-name>", 1.0, "fourier_gr1_arms_waist"),
]
```

- **Single-task training:** keep only the tuple for the task you want to train.
- **Multi-task training:** keep all and only the task tuples you want in the mixture.

Do not change the sampling weight or robot type unless you are intentionally designing a new mixture. The dataset name must match a directory under the configured `data_root_dir`. All three RoboCasa launchers select this same mixture, so editing it changes the task set for `DinoGR00T`, `Dinov3CLIPGR00T`, and `QwenVisionGR00T`. Give each task selection a new `run_id` so checkpoints are not mixed between experiments.

### Pure visual encoder: `DinoGR00T`

```bash
bash Bench/Robocasa_tabletop/train_files/run_dino_robocasa.sh
```

Select a visual encoder and deterministic trajectory subset:

```bash
VISION_MODEL=vggt_omega \
DATA_ROOT_DIR=playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim \
NUM_TRAJECTORIES=100 \
TRAJECTORY_SUBSET_SEED=42 \
RUN_ROOT_DIR=playground/Checkpoints \
bash Bench/Robocasa_tabletop/train_files/run_dino_robocasa.sh
```

A positional trajectory count takes priority:

```bash
bash Bench/Robocasa_tabletop/train_files/run_dino_robocasa.sh 100
```

### Visual encoder + CLIP text: `Dinov3CLIPGR00T`

```bash
bash Bench/Robocasa_tabletop/train_files/run_dinov3_clip_robocasa.sh
```

```bash
VISION_MODEL=dinov3_base \
DATA_ROOT_DIR=playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim \
NUM_TRAJECTORIES=100 \
TRAJECTORY_SUBSET_SEED=42 \
bash Bench/Robocasa_tabletop/train_files/run_dinov3_clip_robocasa.sh
```

### VLM visual tokens + CLIP text: `QwenVisionGR00T`

```bash
BASE_VLM=playground/Pretrained_models/Xiaomi-Robotics \
DATA_ROOT_DIR=playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim \
RUN_ROOT_DIR=playground/Checkpoints \
bash Bench/Robocasa_tabletop/train_files/run_qwenvision_robocasa.sh
```

To train with layer 16 instead of the default final VLM layer, add this argument to the `accelerate launch` command in `run_qwenvision_robocasa.sh`:

```bash
  --framework.vision_text_fusion.vlm_layer_idx 16 \
```

Place it with the other `--framework.*` arguments. Omit the line, or use `-1`, to select the final layer.

### Training controls

| Variable / setting | Used by | Meaning |
| --- | --- | --- |
| `VISION_MODEL` | DinoGR00T, Dinov3CLIPGR00T | Encoder shorthand or supported model ID. |
| `BASE_VLM` | QwenVisionGR00T | Local VLM directory or supported model ID. |
| `DATA_ROOT_DIR` | All three launchers | Root containing the downloaded GR1 task folders. |
| `NUM_TRAJECTORIES` | DinoGR00T, Dinov3CLIPGR00T | Maximum trajectories per configured task. |
| `TRAJECTORY_SUBSET_SEED` | DinoGR00T, Dinov3CLIPGR00T | Seed for deterministic subset selection. |
| `RUN_ROOT_DIR` | All three launchers | Checkpoint and run-output root. |
| `data_mix` | Launcher/YAML | Dataset mixture; provided recipes use `fourier_gr1_unified_1`. |
| `run_id` | Shell launcher or YAML | Experiment directory name. Use a unique name for each run. |
| `CUDA_VISIBLE_DEVICES` | Shell launcher | Training GPUs. Keep it consistent with Accelerate `--num_processes`. |

The complete encoder and VLM catalog is in the [root README](../../README.md#visual-encoder-support).

## 🧰 3. Prepare the RoboCasa simulator

RoboCasa GR1 Tabletop is an external simulator dependency. Follow the installation instructions in the [official RoboCasa GR1 Tabletop repository](https://github.com/robocasa/robocasa-gr1-tabletop-tasks) and verify that its own example or smoke test runs before connecting HIVE-Bench.

After cloning `robocasa-gr1-tabletop-tasks`, open `robocasa/__init__.py` in that external checkout and allow the Robosuite version used by the released environment. Preserve the existing entries and add `"1.5.2"`:

```diff
 assert robosuite.__version__ in [
     "1.5.0",
     "1.5.1",
+    "1.5.2",
-], "robosuite version must be 1.5.{0,1}. Please install the correct version"
+], "robosuite version must be 1.5.{0,1,2}. Please install the correct version"
```

Make this change in the external RoboCasa checkout, not in HIVE-Bench. Then finish the official installation and confirm that RoboCasa imports successfully in its own environment.

Keep the environments separate:

- `hivebench` loads the trained policy and hosts the WebSocket server;
- your RoboCasa environment loads `robocasa`, `robosuite`, MuJoCo, the task assets, and the HIVE-Bench client bridge.

After the official RoboCasa environment imports and runs successfully, install the additional HIVE-Bench evaluation-client dependencies in that same environment:

```bash
conda activate <robocasa-env>
cd /path/to/HIVE-Bench
pip install -r Bench/Robocasa_tabletop/eval_files/requirements.txt
```

This lightweight requirements file includes `tyro`, `websockets`, `hydra-core`, `lightning`, video/serialization packages, and the Python libraries imported by the HIVE-Bench client bridge. It intentionally leaves RoboCasa, Robosuite, MuJoCo, Gymnasium, and NumPy under the official simulator environment's version control.

Run the simulator command from the HIVE-Bench repository root so the relative imports resolve:

```bash
export PYTHONPATH="$PWD/Policy:$PWD:$PYTHONPATH"
```

You normally do **not** need to export `MUJOCO_GL`. Let the working official RoboCasa environment choose its configured rendering backend. Set `MUJOCO_GL=egl` only if your own headless setup explicitly requires EGL.

## 🌐 4. Start the policy server for manual evaluation

This step is only for the manual single-task workflow. The 12-task batch launcher in Section 6 starts and stops its own policy servers.

In the HIVE-Bench environment:

```bash
conda activate hivebench
cd /path/to/HIVE-Bench

CUDA_VISIBLE_DEVICES=0 \
python Policy/deployment/model_server/server_policy.py \
  --ckpt_path playground/Checkpoints/<run>/pytorch_model.pt \
  --port 5681
```

| Server argument | Default | Meaning |
| --- | --- | --- |
| `--ckpt_path` | required | Trained checkpoint file or compatible checkpoint directory path. |
| `--port` | `10093` | WebSocket port. It must match the simulator's `--args.port` or batch `PORT`. |

The server listens on all interfaces. Use `127.0.0.1` from the simulator when both processes run on the same machine.

## 🎯 5. Manual single-task evaluation

In a second terminal, activate the verified RoboCasa environment and run:

```bash
conda activate <robocasa-env>
cd /path/to/HIVE-Bench

PYTHONPATH="$PWD/Policy:$PWD:$PYTHONPATH" \
python Bench/Robocasa_tabletop/eval_files/simulation_env.py \
  --args.host 127.0.0.1 \
  --args.port 5681 \
  --args.env_name gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env \
  --args.n_episodes 50 \
  --args.n_envs 1 \
  --args.max_episode_steps 720 \
  --args.n_action_steps 12 \
  --args.video_out_path results/robocasa_eval \
  --args.pretrained_path playground/Checkpoints/<run>/pytorch_model.pt
```

### Simulator parameters

| Argument | Meaning |
| --- | --- |
| `--args.host` | Policy-server host. Use `127.0.0.1` for the same machine. |
| `--args.port` | Policy-server port; it must match `server_policy.py --port`. |
| `--args.env_name` | Registered RoboCasa GR1 environment ID. |
| `--args.n_episodes` | Number of completed episodes to evaluate. |
| `--args.n_envs` | Number of vectorized simulator environments. Start with `1`. |
| `--args.max_episode_steps` | Maximum environment steps per episode. |
| `--args.n_action_steps` | Number of predicted actions executed before requesting another chunk. |
| `--args.video_out_path` | Directory for recorded evaluation videos. |
| `--args.pretrained_path` | The same trained checkpoint, read locally by the client to recover action normalization statistics. |
| `--args.seed` | Simulator/client seed. |

`pretrained_path` does not load a second policy in the simulator process. It supplies the normalization metadata required to decode the server's normalized action output.

The server and client must agree on checkpoint, port, action representation, and action chunk behavior.

## 📊 6. Recommended: 12-task batch evaluation

`run_encoder_eval.sh` evaluates **one checkpoint on all 12 RoboCasa tasks, three times per task**. It creates one worker per configured GPU, distributes tasks through a shared queue, starts a policy server for each task, runs the simulator three times, shuts that server down, and then takes the next task.

You do **not** need to start a policy server separately for this workflow.

### Configure the launcher

Open `Bench/Robocasa_tabletop/eval_files/run_encoder_eval.sh` and edit these three values near the top:

```bash
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${HIVE_BENCH_DIR}/playground/Checkpoints/your_run/pytorch_model.pt}"
GPUS_STR="${GPUS_STR:-0,1,2,3}"
PORT_BASE="${PORT_BASE:-8000}"
```

| Setting | What to change |
| --- | --- |
| `CHECKPOINT_PATH` | Checkpoint evaluated on all 12 tasks. Use the `.pt` file whose run directory also contains `config.yaml` and `dataset_statistics.json`. |
| `GPUS_STR` | Comma-separated GPU IDs with no spaces. The number of IDs is the number of parallel workers; for example, `0,1` uses two GPUs. |
| `PORT_BASE` | Base WebSocket port. Each worker uses `PORT_BASE + gpu_id`; choose a free range. |

The script discovers the HIVE-Bench root automatically. It also tries to discover Conda with `conda info --base`. If your environment names differ, edit `SERVER_CONDA_ENV` and `CLIENT_CONDA_ENV`; set `CONDA_BASE` only when Conda cannot be discovered automatically.

### Run all evaluations

From the HIVE-Bench repository root, run exactly:

```bash
bash Bench/Robocasa_tabletop/eval_files/run_encoder_eval.sh
```

The default evaluation is:

- 12 built-in RoboCasa tasks;
- 3 independent runs per task;
- 50 episodes per run;
- one server/simulator worker per entry in `GPUS_STR`;
- no forced bf16 and no forced rendering backend.

Each GPU worker repeatedly takes the next available task. The policy server remains alive for that task's three runs and is restarted before the worker moves to another task.

### Outputs

Results are written under:

```text
results/robocasa_eval/<run-tag>_<timestamp>_<pid>/
├── results_raw.csv
├── results_summary.csv
├── logs/<task>/
│   ├── server.log
│   ├── client_run1.log
│   ├── client_run2.log
│   └── client_run3.log
└── videos/<task>/run{1,2,3}/
```

`results_summary.csv` contains `run1`, `run2`, `run3`, and their mean for every completed task. Pressing `Ctrl+C` triggers cleanup of all policy-server and simulator process groups started by the launcher.

## 🧯 Troubleshooting

- **Connection refused:** confirm the policy server is ready and both sides use the same port.
- **Checkpoint statistics missing:** set `CHECKPOINT_PATH` in the batch launcher, or pass the same checkpoint through `--args.pretrained_path` in the manual workflow.
- **Unknown environment ID:** verify the official GR1 tabletop tasks are installed and registered in the active RoboCasa environment.
- **MuJoCo rendering failure:** first repeat the official simulator smoke test. Only set `MUJOCO_GL` when your machine's headless or display setup requires an explicit backend.
- **CUDA out of memory:** reduce `N_ENVS`, use fewer parallel workers, or evaluate a smaller backbone.
- **Dataset not found during training:** set `DATA_ROOT_DIR` or edit `data_root_dir` in the selected YAML.

## Training recipe configuration

The matching shell launcher and YAML in `Bench/Robocasa_tabletop/train_files/` define each training recipe. Edit the paired YAML to change the model configuration, per-device batch size, training steps, save interval, optimizer settings, and other training hyperparameters. Use the shell launcher for run-specific values such as the selected model path, GPUs, dataset root, and run ID.
