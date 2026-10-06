<h1 align="center">HIVE-Bench</h1>

<p align="center">
Evaluating patch-level visual representations<br>for egocentric robot manipulation
</p>

<p align="center">
  <a href="https://github.com/KolaKivy/HIVE-Bench"><b>Code</b></a>
  &nbsp;·&nbsp;
  <a href="https://huggingface.co/datasets/zhengtu666/HIVE-Bench-Data"><b>Data</b></a>
  &nbsp;·&nbsp;
  <a href="Bench/Robotwin/README.md"><b>RoboTwin</b></a>
  &nbsp;·&nbsp;
  <a href="Bench/Robocasa_tabletop/README.md"><b>RoboCasa</b></a>
  &nbsp;·&nbsp;
  <a href="Analyze/README.md"><b>Analysis</b></a>
</p>

<p align="center">
  <img src="assets/teaser.png" width="100%" alt="HIVE-Bench: twenty encoders, one patch-conditioned policy, and four diagnostic families.">
</p>

<p align="center">
20 encoders &nbsp;·&nbsp; 7 families &nbsp;·&nbsp; 24 tasks &nbsp;·&nbsp; 2 simulators &nbsp;·&nbsp; 50+ diagnostics
</p>

---

HIVE-Bench compares dense patch tokens from pretrained visual encoders on closed-loop bimanual manipulation. The same flow-matching DiT reads every encoder. Architecture, token interface, data protocol, training recipe, and simulator evaluation stay fixed inside each comparison, so a difference in success is attributable to the representation.

Policies see onboard cameras and receive no proprioception. Robot state has to come from the image tokens. The study covers frozen and fine-tuned encoders, single-task and multi-task training, and sweeps over data, model size, and VLM layer.

## Findings

Numbers below are from the paper. Correlations are Spearman ρ with suite success.

| | |
| --- | --- |
| Patch tokens | Mean-pooling each view drops frozen DINOv2 from 32.3% to 9.9% and DINOv3 from 31.2% to 6.4% on RoboCasa multi-task. 23 of 24 task comparisons favor the patch tokens. |
| Readout probes | Inverse-dynamics error tracks success on every frozen vision-encoder cohort (\|ρ\| ≥ 0.64; −0.82 on RoboTwin). It still tracks success after the VLMs are added. State and object probes do not, once those VLMs join. |
| Vision benchmarks | ImageNet, segmentation, depth, and correspondence scores do not track closed-loop success. |
| Fine-tuning | All 14 frozen/fine-tuned pairs improve. On RoboTwin the gains run from 2.2 to 26.1 points; VGGT-Ω goes from 50.0% to 76.1%. Frozen rank still predicts fine-tuned rank (ρ = 0.96 on RoboTwin, 0.89 on RoboCasa). |
| Scale | Scaling is family-specific. DINOv2 saturates after Base on RoboCasa. DINOv3 beats DINOv2 on standard vision benchmarks at matched size and trails it on both manipulation suites. |
| VLM layer | Layer 16 gains 6.0–10.7 points on the RoboCasa settings we measured, and changes RoboTwin by less than a point. |
| Ranking | The ends of the ranking are more stable than the middle. VLM-derived encoders lead RoboTwin and sit mid-table on RoboCasa. |

## Protocol

The main comparison uses the default checkpoint of each of 20 encoders on 12 RoboCasa-GR1 tasks and 12 RoboTwin 2.0 tasks. Evaluation is 3 seeds × 50 rollouts per task.

| | RoboCasa-GR1 | RoboTwin 2.0 |
| --- | --- | --- |
| Embodiment | GR1 humanoid, dexterous hands | Dual arm, grippers |
| Demonstrations | up to 1,000 / task | up to 500 / task |
| Cameras | head | head + two wrists |
| Action | 29-D | 14-D joint position |
| Chunk | 16 predicted, 12 executed | 16 predicted, 16 executed |
| Training | single-task and multi-task | multi-task |
| Encoder | all 20 frozen; 7 fine-tuned | all 20 frozen; 7 fine-tuned |
| Proprioception | none | none |

Single-task trains one policy per task. Multi-task trains one language-conditioned policy on all twelve. Fine-tuning on RoboCasa is single-task; on RoboTwin it is multi-task.

## Encoders

| Family | Paper checkpoints |
| --- | --- |
| Supervised | ViT |
| Self-supervised | MAE, DINOv2, DINOv3, V-JEPA 2.1 |
| Geometry | SPA, VGGT-Ω, LingBot-Vision |
| Vision–language | CLIP, SigLIP, SigLIP2, InternViT |
| Robot | VC-1, Voltron |
| Distillation | Theia, RADIOv2.5, C-RADIOv4 |
| VLM-derived | Qwen3-VL, DepthVLM, Xiaomi-Robotics-1 |

The release also includes other sizes, compatibility aliases, intermediate-layer extraction, and VLM adapters that were not all in the paper matrix.

<details>
<summary>Visual encoder names accepted by <code>DinoGR00T</code></summary>

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
<summary>VLM adapters</summary>

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

```text
RGB views
    │
    ▼
encoder or VLM image tokens
    │  dense patches, not a pooled vector
    ▼
projection, optional language
    │
    ▼
flow-matching DiT
    │
    ▼
16-step action chunk
```

| Framework | Vision | Language | Head |
| --- | --- | --- | --- |
| `DinoGR00T` | pluggable visual encoder | optional, from the recipe | flow-matching DiT |
| `Dinov3CLIPGR00T` | pluggable visual encoder | frozen CLIP text | fused tokens → DiT |
| `QwenVisionGR00T` | selected VLM layers | frozen CLIP text | projected tokens → DiT |

## Diagnostics

The paper reports 49–51 measurements per cohort. The runner exposes 22 operators (10 single-frame, 10 temporal, 2 sequence-level); per-view, per-layer, spectral, and probe outputs make up the rest.

| Family | Question | Examples |
| --- | --- | --- |
| Token statistics | How is information laid out in a frame? | anisotropy, neighbor similarity, effective rank, norm entropy |
| Temporal | How do tokens move along a trajectory? | drift, lag-1 autocorrelation, spectral entropy, frequency bands |
| Readouts | Can robot variables be decoded? | inverse dynamics, forward dynamics, joint state, object position |
| Policy probes | Does the policy ignore nuisance change? | texture sensitivity, action robustness, representation shape |

Associations in the paper use 100,000-shuffle permutation tests, Benjamini–Hochberg correction within each cohort, and 4,000 bootstrap resamples. Operator lists, dataset conventions, and launchers are in the [analysis guide](Analyze/README.md).

## Install

Python 3.10+. 3.11 is the version we use.

```bash
git clone https://github.com/KolaKivy/HIVE-Bench.git
cd HIVE-Bench

conda create -n hivebench python=3.11 -y
conda activate hivebench
pip install -r requirements.txt
pip install -e .
```

Weights, datasets, checkpoints, and logs are not in the repo. Point YAML or the shell launchers at a local checkpoint, or pass a Hugging Face model id when the adapter supports it. Code under `third_party/` is part of the release.

RoboTwin and RoboCasa keep their own simulator environments, separate from `hivebench`. Follow the benchmark guides rather than installing both simulators into this env.

Qwen adapters expect FlashAttention 2:

```bash
pip install flash-attn --no-build-isolation
```

Qwen3.5 needs its own environment with `transformers>=5.2.0`. Extra adapters are integrations. They are not a claim that every model was in the paper, or that one dependency pin serves all of them.

## Guides

Run commands from the repository root.

| | |
| --- | --- |
| [RoboTwin 2.0](Bench/Robotwin/README.md) | download, convert, train, serve, evaluate |
| [RoboCasa-GR1](Bench/Robocasa_tabletop/README.md) | download, train, serve, evaluate |
| [Analysis](Analyze/README.md) | probes, token metrics, robustness, figures |

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
