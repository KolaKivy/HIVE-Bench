<h1 align="center">🐝 HIVE-Bench</h1>

<p align="center"><strong>Evaluating Patch-Level Visual Representations for Egocentric Robot Manipulation</strong></p>

<p align="center"><em>A controlled, large-scale benchmark for discovering what robot vision encoders really understand.</em></p>

<p align="center">
  <a href="Bench/Robotwin/README.md">🤖 RoboTwin</a> ·
  <a href="Bench/Robocasa_tabletop/README.md">🏠 RoboCasa</a> ·
  <a href="Analyze/README.md">🔬 Analysis</a>
</p>

| **20 benchmarked encoders** | **7 representation families** | **10 VLM adapters** | **24 bimanual tasks** | **1,500+ trained policies** | **≈250K rollouts** | **50+ diagnostics** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |

---

## 🌟 What is HIVE-Bench?

Modern robot policies do not see the world through a single pooled image vector. They consume **dense patch tokens** from moving, partially occluded head and wrist cameras, then use those tokens to recover robot state, object geometry, motion, and action-relevant detail. Yet most representation benchmarks still emphasize global features, static images, or third-person perception.

**HIVE-Bench closes that gap.** To our knowledge, it is the first benchmark built specifically to compare dense patch-level visual representations for egocentric bimanual manipulation under one shared policy architecture.

The benchmark holds the policy architecture, token interface, data protocol, training recipe, and simulator evaluation fixed within each comparison, and swaps the visual representation. This turns encoder comparison from a collection of incompatible policy results into a controlled scientific experiment. The action head is trained; “fixed” here refers to the comparison protocol, not frozen action-head weights.

| Conventional comparison | HIVE-Bench |
| --- | --- |
| Global or class-token features | Dense patch-token representations |
| Static or third-person images | Egocentric head and wrist cameras |
| Different policy heads and recipes | One shared flow-matching DiT policy |
| Success rate alone | Success rate plus 50+ representation diagnostics |
| One dataset or training regime | Two simulators, frozen/fine-tuned, single/multi-task, data/scale/layer sweeps |

HIVE-Bench is both a **benchmark** and a **research toolkit**: train a policy, evaluate it in RoboTwin or RoboCasa, inspect its token geometry and temporal dynamics, and test whether the representation preserves actions, state, objects, and task-relevant variation.

> **The central lesson of the paper:** robot representation quality cannot be inferred reliably from model family, parameter count, or standard vision benchmarks. It has to be measured where it matters—inside closed-loop manipulation—and explained with robot-relevant diagnostics.

## 🔥 What the benchmark reveals

### Dense patch tokens are not optional

Mean-pooling each camera view before the same policy head removes most of the useful control signal. On RoboCasa multi-task evaluation, DINOv2 falls from **32.3% to 9.9%**, while DINOv3 falls from **31.2% to 6.4%**. Across the two encoders, **23 of 24 task-level comparisons** favor dense patch tokens.

### Lightweight readout probes predict manipulation quality

Inverse-dynamics and state-decoding errors are the strongest and most consistent diagnostics in the frozen-encoder study. IDM error reaches a Spearman correlation of **ρ = −0.82** with RoboTwin success and remains at **|ρ| ≥ 0.64** in every frozen benchmark cohort. These probes are especially valuable for screening out weak encoders before paying for full policy training and rollout evaluation.

### Standard vision leaderboards are not robot-control leaderboards

ImageNet, segmentation, depth, and correspondence scores do not significantly track downstream success in the study. A representation can excel on familiar perception benchmarks yet fail to preserve the egocentric state and action information required for control.

### Fine-tuning helps every tested encoder—but does not erase representation quality

All **14 frozen/fine-tuned pairs** improve. Gains reach **26.1 success points** on RoboTwin, where VGGT-Ω rises from **50.0% to 76.1%**. At the same time, the frozen ranking remains highly predictive of the fine-tuned ranking (**ρ = 0.96** on RoboTwin and **ρ = 0.89** on RoboCasa), making frozen evaluation a useful model-selection signal.

### Bigger is not automatically better

Scaling behavior is family-dependent: DINOv2 saturates after Base, DINOv3 Base is strongest on RoboCasa, and V-JEPA2.1 follows a different scaling curve. HIVE-Bench exposes these differences instead of assuming parameter count is a universal proxy for control quality.

### The best VLM layer depends on the embodiment

Intermediate layer-16 features improve the tested VLMs by **6.0–10.7 points on RoboCasa**, while changing RoboTwin performance by less than one point. HIVE-Bench therefore treats feature extraction depth as a benchmark variable rather than a hidden implementation detail.

### There is no universal winner

The strongest and weakest encoders are relatively stable, but the middle of the ranking changes across embodiments and training regimes. VGGT-Ω, DINOv2, and LingBot-Vision remain among the top six in all three frozen cohorts, while VLM-derived encoders lead RoboTwin but sit mid-table on RoboCasa. HIVE-Bench is designed to expose this context dependence.

## 📊 Benchmark at a glance

The paper evaluates **20 pretrained encoders from seven representation families** on **24 manipulation tasks**: 12 RoboCasa-GR1 tasks and 12 RoboTwin 2.0 tasks.

| Protocol | RoboCasa-GR1 | RoboTwin 2.0 |
| --- | --- | --- |
| Embodiment | GR1 humanoid with dexterous hands | Dual-arm robot |
| Tasks | 12 | 12 |
| Demonstrations | Up to 1,000 per task | Up to 500 per task |
| Camera observations | Head camera | Head + two wrist cameras |
| Action space | 29-D | 14-D joint actions |
| Predicted action chunk | 16 steps; execute 12 | 16 steps; execute 16 |
| Training regimes | Single-task and multi-task | Multi-task |
| Encoder regimes | Frozen all 20; fine-tuned subset | Frozen all 20; fine-tuned subset |
| Evaluation | 3 seeds × 50 rollouts per task | 3 seeds × 50 rollouts per task |

The main paper protocol deliberately provides **no proprioceptive observation to the policy**. Robot state must be inferred from images, making the benchmark a direct test of what each visual representation preserves.

### The 20-encoder paper matrix

| Representation family | Evaluated encoders |
| --- | --- |
| Supervised | ViT |
| Self-supervised | MAE, DINOv2, DINOv3, V-JEPA2.1 |
| Geometry-oriented | SPA, VGGT-Ω, LingBot-Vision |
| Vision-language pretraining | CLIP, SigLIP, SigLIP2, InternViT |
| Robot-oriented | VC-1, Voltron |
| Multi-teacher distillation | Theia, RADIOv2.5, C-RADIOv4 |
| VLM-derived representations | Qwen3-VL, DepthVLM, Xiaomi Robotics-1 |

The release is intentionally broader than the paper matrix: it includes additional model sizes, compatibility routes, intermediate-layer extraction, VLM adapters, and analysis tools for extending the benchmark.

## 🧩 A controlled policy backbone

Every encoder is connected to the same patch-conditioned flow-matching policy derived from GR00T:

```text
egocentric RGB views
        │
        ▼
visual encoder / VLM image-position features
        │ dense patch tokens
        ▼
projection + optional language conditioning
        │
        ▼
16-block flow-matching DiT (recipe-configured width)
        │ 32 learned action queries
        ▼
16-step continuous action chunk
```

This shared interface supports three policy designs:

| Framework | Visual pathway | Language pathway | Action prediction |
| --- | --- | --- | --- |
| `DinoGR00T` | Pluggable pure visual encoder | Optional recipe-level conditioning | Flow-matching DiT |
| `Dinov3CLIPGR00T` | Pluggable visual encoder | Frozen CLIP text encoder | Fused visual-language tokens → DiT |
| `QwenVisionGR00T` | Selectable VLM visual layers | Frozen CLIP text encoder | Projected VLM + text tokens → DiT |

Because the policy remains fixed, a performance difference is far easier to attribute to the representation itself.

<a id="visual-encoder-support"></a>

<details>
<summary><strong>👁️ Canonical visual-encoder identifiers in this release</strong></summary>

Compatibility aliases remain accepted by the code, but the list below intentionally shows one canonical spelling per variant.

| Family | Canonical identifiers |
| --- | --- |
| DINOv2 | `dinov2_small`, `dinov2_base`, `dinov2_large`, `dinov2_giant` |
| DINOv3 | `dinov3_small`, `dinov3_base`, `dinov3_large`, `dinov3_vith16plus` |
| CLIP / SigLIP | `clip`, `siglip`, `siglip2` |
| ViT / MAE / SAM | `vit`, `mae`, `sam3` |
| RADIO / C-RADIO | `radio`, `cradio_v4_h`, `cradio_v4_so400m` |
| General representation | `theia`, `internvit`, `mocov3`, `eva`, `eva02` |
| Robot representation | `voltron`, `mcr`, `lingbot_small`, `lingbot_base`, `lingbot_large`, `lingbot_giant` |
| Geometry / 3D | `vggt`, `vggt_omega`, `vggt_omega_register`, `da3` |
| V-JEPA2.1 | `vjepa2.1_base`, `vjepa2.1_large`, `vjepa2.1_giant` |
| LeVJEPA | `levjepa_videomix_large` |

The factory also accepts `spa_<variant>`, `vc1_base`, `vc1_large`, `distill_theia_<checkpoint>`, supported Hugging Face model IDs, and direct DINOv2 Torch Hub identifiers. LingBot-Vision is a **pure visual encoder** in `DinoGR00T`, not a VLM.

</details>

<details>
<summary><strong>💬 VLM adapters and layer-wise identifiers</strong></summary>

The paper evaluates visual representations from **Qwen3-VL, DepthVLM, and Xiaomi Robotics-1**. The release additionally implements a wider adapter catalog:

| Adapter family | Model-name match |
| --- | --- |
| Xiaomi Robotics-1 | `xiaomi-robotics` or `xiaomirobotics` |
| DepthVLM | `depthvlm` |
| Qwen2.5-VL / Nora | `Qwen2.5-VL` or `nora` |
| Qwen3.5 | `Qwen3.5` |
| Qwen3-VL | `Qwen3-VL` |
| Gemma 3 | `gemma-3` or `gemma3` |
| SmolVLM | `smolvlm` or `smol` |
| OpenVLA | `openvla` |
| Florence-2 | `florence` |
| Cosmos-Reason2 | `cosmos-reason2` |

For representation analysis, use `qwen3` / `qwen3_layer16`, `xiaomi` / `xiaomi_layer16`, and `depthvlm` / `depthvlm_layer16`. The `_layer16` form extracts the corresponding intermediate hidden layer; the unsuffixed form uses the adapter's default visual representation.

</details>

## 🔬 50+ diagnostics—not just IDM

HIVE-Bench does more than rank policies by success. It asks **what information is present in the representation, how that information evolves over time, and which measurements actually predict closed-loop manipulation**.

The paper reports **49–51 diagnostics per benchmark cohort**, organized into four complementary families:

| Diagnostic family | What it measures | Representative signals |
| --- | --- | --- |
| Token statistics | Spatial geometry and information distribution within a frame | anisotropy, mean patch cosine, neighbor similarity, token norms, effective rank, norm entropy, within/between-frame variance |
| Temporal diagnostics | Stability, drift, and frequency structure across trajectories | temporal drift, lag-1 autocorrelation, temporal effective rank, spectral entropy, low/mid/high-frequency energy, trajectory variation |
| Readout probes | Whether robot-relevant variables can be decoded | IDM error, state-decoding error, object-position error, plus the release's FDM probe |
| Policy and distribution probes | Whether learned control is robust to nuisance variation and feature shaping | texture sensitivity, action robustness, Gaussian/spherical representation shaping |

The unified analysis runner exposes **22 core operators**—10 single-frame, 10 temporal, and 2 sequence-level operators. Their per-view, per-layer, spectral, lag, trajectory, and aggregate outputs combine with trainable probes and policy diagnostics to produce the paper's 50+ reported measurements. This is why the analysis suite is substantially larger than the four named IDM/FDM/State/Object probe families.

### Included analysis surfaces

- **Spatial/token geometry:** PCA, average token cosine, distance-similarity decay, neighbor similarity, covariance rank, norm entropy, norm variance, token-to-global similarity.
- **Temporal dynamics:** smoothness, cosine shift, lag-distance curves, temporal variance, temporal rank, autocorrelation, spectral entropy, patch dynamics, trajectory variation.
- **Robot readouts:** inverse dynamics, forward dynamics, proprioceptive state, and task-object position.
- **Robustness and structure:** texture/action sensitivity, PCA, and Gaussian/spherical representation-shape analysis.
- **Statistical evidence:** 100,000-shuffle permutation tests, Benjamini–Hochberg correction, and 4,000 bootstrap resamples in the paper protocol.

The result is not merely a larger metric table. It is an evidence stack that connects **representation structure → decodable robot information → policy behavior → closed-loop success**.

For operators, supported models, dataset conventions, probe training, multi-GPU launchers, and output interpretation, continue to the **[🔬 Analyze guide](Analyze/README.md)**.

## 🚀 Installation

HIVE-Bench requires Python 3.10 or newer; Python 3.11 is recommended.

```bash
git clone https://github.com/KolaKivy/HIVE-Bench.git
cd HIVE-Bench

conda create -n hivebench python=3.11 -y
conda activate hivebench

pip install -r requirements.txt
pip install -e .
```

Required research dependencies included with the release are under `third_party/` and are resolved by the code. Pretrained backbone weights, datasets, checkpoints, videos, logs, and generated outputs are intentionally excluded from the repository.

Place local weights at the path selected by your YAML or shell launcher, or use a supported Hugging Face model ID when the adapter permits it. RoboTwin and RoboCasa have independent simulator dependency stacks; follow their official installation instructions and keep those environments separate from the main `hivebench` environment.

Some adapters have additional model-specific requirements. The Qwen-based policy adapters use Flash Attention 2 by default: install `flash-attn` in a compatible CUDA/PyTorch environment (`pip install flash-attn --no-build-isolation`) before using those recipes. The Qwen3.5 adapter requires a separate environment with `transformers>=5.2.0`, rather than the pinned default version. Additional adapter entries are integrations, not a claim that every model was evaluated in the paper or works with the same dependency versions.

## 🧭 Start here

Dataset preparation, training, serving, command-line parameters, and evaluation are documented with each benchmark:

| Goal | Guide |
| --- | --- |
| Download/convert data, train, serve, and evaluate on RoboTwin 2.0 | [🤖 RoboTwin 2.0](Bench/Robotwin/README.md) |
| Download data, train, serve, and evaluate on RoboCasa-GR1 Tabletop | [🏠 RoboCasa-GR1 Tabletop](Bench/Robocasa_tabletop/README.md) |
| Run the 50+ diagnostics, probes, visualizations, and robustness analyses | [🔬 Representation analysis](Analyze/README.md) |

## 📁 Repository layout

```text
HIVE-Bench/
├── Policy/
│   ├── hivebench/
│   │   ├── model/          # Policy frameworks, encoders, VLM adapters, action heads
│   │   ├── dataloader/     # Dataset readers and conversion tools
│   │   ├── training/       # Training entrypoints
│   │   └── config/         # Shared training configuration
│   └── deployment/         # Policy serving
├── Bench/
│   ├── Robotwin/           # RoboTwin data, training, and evaluation workflow
│   └── Robocasa_tabletop/  # RoboCasa data, training, and evaluation workflow
├── Analyze/                # 50+ diagnostics, probes, visualization, and runners
└── third_party/            # Bundled research dependencies
```

## 🤝 Extending HIVE-Bench

HIVE-Bench is designed as a common testbed rather than a closed model list. New visual encoders implement the token interface used by `DinoGR00T`; new VLMs expose visual-token extraction through the shared adapter; new diagnostics plug into the analysis runner while keeping outputs self-describing and reproducible.

Contributions, new encoder integrations, diagnostic ideas, and reproducibility reports are welcome.

## 📄 Paper and license

Paper: **HIVE-Bench: Evaluating Patch-Level Visual Representations for Egocentric Robot Manipulation**. The public paper link will be added when available.

The public citation will be added after the double-blind review period. See [LICENSE](LICENSE) for the repository license, and retain the original notices of bundled or adapted third-party components.
