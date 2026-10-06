<p align="center">
  <img src="assets/teaser.png" width="100%" alt="HIVE-Bench: twenty encoders, one patch-conditioned policy, and four diagnostic families.">
</p>

<p align="center">
  <a href="https://huggingface.co/datasets/zhengtu666/HIVE-Bench-Data"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Data-FFD21E?style=for-the-badge&labelColor=111827" alt="Data"></a>
  <a href="Bench/Robotwin/README.md"><img src="https://img.shields.io/badge/RoboTwin-1D4ED8?style=for-the-badge&labelColor=1D4ED8" alt="RoboTwin"></a>
  <a href="Bench/Robocasa_tabletop/README.md"><img src="https://img.shields.io/badge/RoboCasa-C2410C?style=for-the-badge&labelColor=C2410C" alt="RoboCasa"></a>
  <a href="Analyze/README.md"><img src="https://img.shields.io/badge/Analysis-6D28D9?style=for-the-badge&labelColor=6D28D9" alt="Analysis"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-374151?style=for-the-badge&labelColor=111827" alt="MIT"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/20-encoders-1D4ED8?style=for-the-badge&labelColor=1e3a8a" alt="20 encoders">
  <img src="https://img.shields.io/badge/7-families-0F766E?style=for-the-badge&labelColor=115e59" alt="7 families">
  <img src="https://img.shields.io/badge/24-tasks-C2410C?style=for-the-badge&labelColor=9a3412" alt="24 tasks">
  <img src="https://img.shields.io/badge/2-simulators-B45309?style=for-the-badge&labelColor=92400e" alt="2 simulators">
  <img src="https://img.shields.io/badge/50+-diagnostics-6D28D9?style=for-the-badge&labelColor=5b21b6" alt="50+ diagnostics">
</p>

<p align="center">
  <b>Evaluating patch-level visual representations for egocentric robot manipulation</b>
</p>

HIVE-Bench compares dense patch tokens from pretrained visual encoders on closed-loop bimanual manipulation. The same flow-matching DiT reads every encoder. Architecture, token interface, data protocol, training recipe, and simulator evaluation stay fixed inside each comparison.

Policies see onboard cameras and receive no proprioception. The study covers frozen and fine-tuned encoders, single-task and multi-task training, and sweeps over data, model size, and VLM layer.

## Findings

Numbers are from the paper. Correlations are Spearman ρ with suite success.

<table>
<tr>
<td width="33%" valign="top">
<img src="https://img.shields.io/badge/Patch_tokens-32.3_%E2%86%92_9.9-BE123C?style=for-the-badge&labelColor=881337" alt="Patch tokens"><br><br>
Mean-pooling each view drops frozen DINOv2 from <b>32.3%</b> to <b>9.9%</b> and DINOv3 from <b>31.2%</b> to <b>6.4%</b> on RoboCasa multi-task. <b>23 of 24</b> task comparisons favor the patch tokens.
</td>
<td width="33%" valign="top">
<img src="https://img.shields.io/badge/IDM-rho_%3D_%2D0.82-1D4ED8?style=for-the-badge&labelColor=1e3a8a" alt="IDM"><br><br>
Inverse-dynamics error tracks success on every frozen vision-encoder cohort (<b>|ρ| ≥ 0.64</b>; −0.82 on RoboTwin) and still does after the VLMs are added. State and object probes do not, once those VLMs join.
</td>
<td width="33%" valign="top">
<img src="https://img.shields.io/badge/Vision_benchmarks-no_transfer-0F766E?style=for-the-badge&labelColor=115e59" alt="Vision benchmarks"><br><br>
ImageNet, segmentation, depth, and correspondence scores do not track closed-loop success. A strong perception checkpoint can still drop the action-relevant signal.
</td>
</tr>
<tr>
<td width="33%" valign="top">
<img src="https://img.shields.io/badge/Fine--tuning-%2B26.1_pt-C2410C?style=for-the-badge&labelColor=9a3412" alt="Fine-tuning"><br><br>
All <b>14</b> frozen/fine-tuned pairs improve. On RoboTwin the gains run from 2.2 to 26.1 points; VGGT-Ω goes from <b>50.0%</b> to <b>76.1%</b>. Frozen rank still predicts fine-tuned rank (ρ = 0.96 / 0.89).
</td>
<td width="33%" valign="top">
<img src="https://img.shields.io/badge/Scale-family_specific-B45309?style=for-the-badge&labelColor=92400e" alt="Scale"><br><br>
DINOv2 saturates after Base on RoboCasa. DINOv3 beats DINOv2 on standard vision benchmarks at matched size and trails it on both manipulation suites.
</td>
<td width="33%" valign="top">
<img src="https://img.shields.io/badge/Layer_16-%2B6_to_%2B11-6D28D9?style=for-the-badge&labelColor=5b21b6" alt="Layer 16"><br><br>
Layer 16 gains 6.0–10.7 points on the RoboCasa settings we measured, and changes RoboTwin by less than a point. VLM-derived encoders lead RoboTwin and sit mid-table on RoboCasa.
</td>
</tr>
</table>

## Protocol

The main comparison uses the default checkpoint of each of 20 encoders. Evaluation is 3 seeds × 50 rollouts per task. No proprioception.

| | <img src="https://img.shields.io/badge/RoboCasa--GR1-C2410C?style=flat-square&labelColor=9a3412" alt="RoboCasa"> | <img src="https://img.shields.io/badge/RoboTwin_2.0-1D4ED8?style=flat-square&labelColor=1e3a8a" alt="RoboTwin"> |
| --- | --- | --- |
| Embodiment | GR1 humanoid, dexterous hands | Dual arm, grippers |
| Tasks | 12 | 12 |
| Demonstrations | up to 1,000 / task | up to 500 / task |
| Cameras | head | head + two wrists |
| Action | 29-D | 14-D joint position |
| Chunk | 16 predicted, 12 executed | 16 predicted, 16 executed |
| Training | single-task and multi-task | multi-task |
| Encoder | all 20 frozen; 7 fine-tuned | all 20 frozen; 7 fine-tuned |

Single-task trains one policy per task. Multi-task trains one language-conditioned policy on all twelve. Fine-tuning on RoboCasa is single-task; on RoboTwin it is multi-task.

## Encoders

<p>
<img src="https://img.shields.io/badge/Supervised-B45309?style=flat-square&labelColor=92400e" alt="Supervised">
&nbsp; ViT
</p>
<p>
<img src="https://img.shields.io/badge/Self--supervised-1D4ED8?style=flat-square&labelColor=1e3a8a" alt="Self-supervised">
&nbsp; MAE · DINOv2 · DINOv3 · V-JEPA 2.1
</p>
<p>
<img src="https://img.shields.io/badge/Geometry-BE123C?style=flat-square&labelColor=881337" alt="Geometry">
&nbsp; SPA · VGGT-Ω · LingBot-Vision
</p>
<p>
<img src="https://img.shields.io/badge/Vision--language-0F766E?style=flat-square&labelColor=115e59" alt="Vision-language">
&nbsp; CLIP · SigLIP · SigLIP2 · InternViT
</p>
<p>
<img src="https://img.shields.io/badge/Robot-C2410C?style=flat-square&labelColor=9a3412" alt="Robot">
&nbsp; VC-1 · Voltron
</p>
<p>
<img src="https://img.shields.io/badge/Distillation-1E40AF?style=flat-square&labelColor=1e3a8a" alt="Distillation">
&nbsp; Theia · RADIOv2.5 · C-RADIOv4
</p>
<p>
<img src="https://img.shields.io/badge/VLM--derived-6D28D9?style=flat-square&labelColor=5b21b6" alt="VLM-derived">
&nbsp; Qwen3-VL · DepthVLM · Xiaomi-Robotics-1
</p>

The release also includes other sizes, compatibility aliases, intermediate-layer extraction, and VLM adapters beyond the paper matrix.

<details>
<summary><b>Visual encoder names accepted by DinoGR00T</b></summary>

One canonical name per variant. Aliases still work.

| Family | Names |
| --- | --- |
| DINOv2 | `dinov2_small` `dinov2_base` `dinov2_large` `dinov2_giant` |
| DINOv3 | `dinov3_small` `dinov3_base` `dinov3_large` `dinov3_vith16plus` |
| CLIP / SigLIP | `clip` `siglip` `siglip2` |
| ViT / MAE / SAM | `vit` `mae` `sam3` |
| RADIO | `radio` `cradio_v4_h` `cradio_v4_so400m` |
| Other | `theia` `internvit` `mocov3` `eva` `eva02` |
| Robot | `voltron` `mcr` `lingbot_small` `lingbot_base` `lingbot_large` `lingbot_giant` |
| Geometry | `vggt` `vggt_omega` `vggt_omega_register` `da3` |
| V-JEPA 2.1 | `vjepa2.1_base` `vjepa2.1_large` `vjepa2.1_giant` |
| LeVJEPA | `levjepa_videomix_large` |

Also accepted: `spa_<variant>`, `vc1_base`, `vc1_large`, `distill_theia_<checkpoint>`, supported Hugging Face IDs, and DINOv2 Torch Hub names. LingBot-Vision is a visual encoder, not a VLM.

</details>

<details>
<summary><b>VLM adapters</b></summary>

| Adapter | Name match |
| --- | --- |
| Xiaomi-Robotics-1 | `xiaomi-robotics`, `xiaomirobotics` |
| DepthVLM | `depthvlm` |
| Qwen2.5-VL / Nora | `Qwen2.5-VL`, `nora` |
| Qwen3.5 | `Qwen3.5` |
| Qwen3-VL | `Qwen3-VL` |
| Gemma 3 | `gemma-3`, `gemma3` |
| SmolVLM | `smolvlm`, `smol` |
| OpenVLA | `openvla` |
| Florence-2 | `florence` |
| Cosmos-Reason2 | `cosmos-reason2` |

For probes, `qwen3`, `xiaomi`, and `depthvlm` use the default visual layer. The `_layer16` suffix reads hidden layer 16.

</details>

## Policy

<p align="center">
  <img src="https://img.shields.io/badge/RGB_views-1D4ED8?style=for-the-badge&labelColor=1e3a8a" alt="RGB views">
  <img src="https://img.shields.io/badge/%E2%86%92-patch_tokens-BE123C?style=for-the-badge&labelColor=881337" alt="patch tokens">
  <img src="https://img.shields.io/badge/%E2%86%92-flow--matching_DiT-C2410C?style=for-the-badge&labelColor=9a3412" alt="DiT">
  <img src="https://img.shields.io/badge/%E2%86%92-16--step_chunk-6D28D9?style=for-the-badge&labelColor=5b21b6" alt="action chunk">
</p>

| Framework | Vision | Language | Head |
| --- | --- | --- | --- |
| `DinoGR00T` | pluggable visual encoder | optional, from the recipe | flow-matching DiT |
| `Dinov3CLIPGR00T` | pluggable visual encoder | frozen CLIP text | fused tokens → DiT |
| `QwenVisionGR00T` | selected VLM layers | frozen CLIP text | projected tokens → DiT |

## Diagnostics

The paper reports 49–51 measurements per cohort. The runner exposes 22 operators (10 single-frame, 10 temporal, 2 sequence-level). Per-view, per-layer, spectral, and probe outputs make up the rest.

| | | |
| --- | --- | --- |
| <img src="https://img.shields.io/badge/Token_statistics-1D4ED8?style=flat-square&labelColor=1e3a8a" alt="Token statistics"> | How is information laid out in a frame? | anisotropy, neighbor similarity, effective rank, norm entropy |
| <img src="https://img.shields.io/badge/Temporal-0F766E?style=flat-square&labelColor=115e59" alt="Temporal"> | How do tokens move along a trajectory? | drift, lag-1 autocorrelation, spectral entropy, frequency bands |
| <img src="https://img.shields.io/badge/Readouts-C2410C?style=flat-square&labelColor=9a3412" alt="Readouts"> | Can robot variables be decoded? | inverse dynamics, forward dynamics, joint state, object position |
| <img src="https://img.shields.io/badge/Policy_probes-6D28D9?style=flat-square&labelColor=5b21b6" alt="Policy probes"> | Does the policy ignore nuisance change? | texture sensitivity, action robustness, representation shape |

Associations use 100,000-shuffle permutation tests, Benjamini–Hochberg correction within each cohort, and 4,000 bootstrap resamples. Operators, dataset conventions, and launchers are in the [analysis guide](Analyze/README.md).

## Install

<p>
  <img src="https://img.shields.io/badge/Python-3.11-1D4ED8?style=flat-square&logo=python&logoColor=white&labelColor=1e3a8a" alt="Python 3.11">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white&labelColor=9a3412" alt="PyTorch">
</p>

Python 3.10 or newer. 3.11 is the version we use.

```bash
git clone https://github.com/KolaKivy/HIVE-Bench.git
cd HIVE-Bench

conda create -n hivebench python=3.11 -y
conda activate hivebench
pip install -r requirements.txt
pip install -e .
```

Weights, datasets, checkpoints, and logs are not in the repo. Point YAML or the shell launchers at a local checkpoint, or pass a Hugging Face model id when the adapter supports it. Code under `third_party/` is part of the release.

RoboTwin and RoboCasa keep their own simulator environments, separate from `hivebench`.

Qwen adapters expect FlashAttention 2:

```bash
pip install flash-attn --no-build-isolation
```

Qwen3.5 needs its own environment with `transformers>=5.2.0`. Extra adapters are integrations. They are not a claim that every model was in the paper, or that one dependency pin serves all of them.

## Guides

Run commands from the repository root.

<table>
<tr>
<td width="33%" align="center" valign="top">
<a href="Bench/Robotwin/README.md"><img src="https://img.shields.io/badge/RoboTwin_2.0-1D4ED8?style=for-the-badge&labelColor=1e3a8a" alt="RoboTwin 2.0"></a><br><br>
download, convert, train, serve, evaluate
</td>
<td width="33%" align="center" valign="top">
<a href="Bench/Robocasa_tabletop/README.md"><img src="https://img.shields.io/badge/RoboCasa--GR1-C2410C?style=for-the-badge&labelColor=9a3412" alt="RoboCasa-GR1"></a><br><br>
download, train, serve, evaluate
</td>
<td width="33%" align="center" valign="top">
<a href="Analyze/README.md"><img src="https://img.shields.io/badge/Analysis-6D28D9?style=for-the-badge&labelColor=5b21b6" alt="Analysis"></a><br><br>
probes, token metrics, robustness, figures
</td>
</tr>
</table>

RoboTwin demonstrations:

```bash
pip install -U huggingface_hub
hf download zhengtu666/HIVE-Bench-Data --repo-type dataset \
  --include "RoboTwin_data/**" --local-dir playground/Datasets
```

## Layout

```text
Policy/hivebench/     models, dataloaders, training, configs
Policy/deployment/    policy server
Bench/Robotwin/       RoboTwin workflow
Bench/Robocasa_tabletop/
Analyze/              diagnostics and probes
third_party/          bundled research code
```

A new visual encoder implements the token interface used by `DinoGR00T`. A new VLM exposes visual-token extraction through the shared adapter. A new diagnostic plugs into the analysis runner.

## Citation

```bibtex
@article{liang2026hivebench,
  title   = {HIVE-Bench: Evaluating Patch-Level Visual Representations for Egocentric Robot Manipulation},
  author  = {Liang, Qiwei and Liang, Zhengtu and Lai, Minghao and Chen, Yikeng and Wang, Yiming and Cai, Boyang and Fang, Yuetong and Wang, Taowen and Yin, Baiqiao and Chen, Tianxing and Chen, Yue and Liang, Jiaming and Chen, Guangyu and Zhu, Shaolong and Zhang, Shanghang and Ma, Daolin and Xu, Renjing},
  year    = {2026}
}
```

The arXiv link will replace this note once the paper is public. License: [MIT](LICENSE). Keep the notices on bundled third-party code.
