<h1 align="center">HIVE-Bench</h1>

<p align="center">
Evaluating patch-level visual representations<br>for egocentric robot manipulation
</p>

<p align="center">
  <a href="https://huggingface.co/datasets/zhengtu666/HIVE-Bench-Data"><b>Data</b></a>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="Bench/Robotwin/README.md"><b>RoboTwin</b></a>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="Bench/Robocasa_tabletop/README.md"><b>RoboCasa</b></a>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="Analyze/README.md"><b>Analysis</b></a>
</p>

<p align="center">
  <img src="assets/teaser.png" width="100%" alt="HIVE-Bench. Twenty encoders share one patch-conditioned policy. Four diagnostic families sit underneath.">
</p>

<p align="center">
<sub>20 encoders · 7 families · 24 tasks · 2 simulators · 50+ diagnostics</sub>
</p>

<br>

One policy, many visual encoders. HIVE-Bench feeds onboard cameras to a flow-matching DiT as dense patch tokens, then asks what those tokens still know about the robot. The action head, data protocol, and evaluation stay fixed, so the comparison is the representation. The policy gets no proprioception.

<table>
<tr>
<td width="33%" align="center" valign="top">
<a href="Bench/Robotwin/README.md"><b>RoboTwin 2.0</b></a>
<br><br>
Dual arm, grippers
<br>
Head and two wrists
<br>
14-D joint actions
<br><br>
<sub>12 tasks · up to 500 demos</sub>
</td>
<td width="33%" align="center" valign="top">
<a href="Bench/Robocasa_tabletop/README.md"><b>RoboCasa-GR1</b></a>
<br><br>
Humanoid, dexterous hands
<br>
Head camera
<br>
29-D actions
<br><br>
<sub>12 tasks · up to 1,000 demos</sub>
</td>
<td width="33%" align="center" valign="top">
<a href="Analyze/README.md"><b>Analysis</b></a>
<br><br>
Token geometry
<br>
Temporal structure
<br>
Readout probes
<br><br>
<sub>22 operators · 50+ measurements</sub>
</td>
</tr>
</table>

<p align="center">
<sub>Each guide is the full path: data, training, serving, evaluation. Run commands from the repository root.</sub>
</p>

## Install

Python 3.10 or newer. We use 3.11. Simulator stacks stay in their own environments.

```bash
git clone https://github.com/KolaKivy/HIVE-Bench.git
cd HIVE-Bench
conda create -n hivebench python=3.11 -y
conda activate hivebench
pip install -r requirements.txt
pip install -e .
```

RoboTwin demonstrations:

```bash
hf download zhengtu666/HIVE-Bench-Data --repo-type dataset \
  --include "RoboTwin_data/**" --local-dir playground/Datasets
```

Weights, checkpoints, and logs are not in the repo. Qwen adapters expect `flash-attn`. Qwen3.5 needs a separate environment with `transformers>=5.2.0`.

## Policy

```text
RGB  →  patch tokens  →  flow-matching DiT  →  16-step action chunk
```

| | Vision | Language |
| --- | --- | --- |
| `DinoGR00T` | a visual encoder | optional |
| `Dinov3CLIPGR00T` | a visual encoder | frozen CLIP |
| `QwenVisionGR00T` | selected VLM layers | frozen CLIP |

The paper matrix is twenty encoders: ViT, MAE, DINOv2, DINOv3, V-JEPA 2.1, SPA, VGGT-Ω, LingBot-Vision, CLIP, SigLIP, SigLIP2, InternViT, VC-1, Voltron, Theia, RADIOv2.5, C-RADIOv4, Qwen3-VL, DepthVLM, Xiaomi-Robotics-1. The release also carries other sizes and adapters.

<details>
<summary>Encoder names</summary>

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

Also accepted: `spa_<variant>`, `vc1_base`, `vc1_large`, `distill_theia_<checkpoint>`, supported Hugging Face ids, and DINOv2 Torch Hub names.

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

`qwen3`, `xiaomi`, and `depthvlm` use the default visual layer. Append `_layer16` to read hidden layer 16.

</details>

## Layout

```text
Policy/hivebench/          models, data, training
Policy/deployment/         policy server
Bench/Robotwin/            RoboTwin workflow
Bench/Robocasa_tabletop/   RoboCasa workflow
Analyze/                   diagnostics
third_party/               bundled research code
```

A new encoder implements the token interface used by `DinoGR00T`. A new VLM exposes its visual tokens through the shared adapter. A new diagnostic plugs into the analysis runner.

## Citation

```bibtex
@article{liang2026hivebench,
  title   = {HIVE-Bench: Evaluating Patch-Level Visual Representations for Egocentric Robot Manipulation},
  author  = {Liang, Qiwei and Liang, Zhengtu and Lai, Minghao and Chen, Yikeng and Wang, Yiming and Cai, Boyang and Fang, Yuetong and Wang, Taowen and Yin, Baiqiao and Chen, Tianxing and Chen, Yue and Liang, Jiaming and Chen, Guangyu and Zhu, Shaolong and Zhang, Shanghang and Ma, Daolin and Xu, Renjing},
  year    = {2026}
}
```

The arXiv link will replace this note once the paper is public. [MIT](LICENSE). Keep the notices on bundled third-party code.
