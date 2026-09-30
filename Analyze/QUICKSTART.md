# 🚀 Encoder Testing Framework - Quick Start

## ⚡ 快速开始

### 1. 🧩 基本命令格式

```bash
python run.py model=<模型> analysis=<分析方法> video_path=<视频路径> [其他参数]
```

### 2. 💡 常用示例

```bash
# PCA可视化（只保存第一帧）
python run.py model=dinov3 analysis=pca_vis video_path=test.mp4

# 单帧分析 - 输出所有帧的平均指标
python run.py model=clip analysis=avg_token_cos video_path=test.mp4

# 时序分析
python run.py model=dinov3 analysis=temporal_smoothness video_path=test.mp4

# 指定帧范围和步长
python run.py model=dinov3 analysis=autocorrelation video_path=test.mp4 frame_start=0 frame_end=100 stride=2

# 减小批处理大小避免内存溢出
python run.py model=sam analysis=dist_sim_decay video_path=test.mp4 batch_size=4
```

### 3. 🤖 可用模型

- `dinov3` - DINOv3
- `clip` - CLIP
- `sam` - Segment Anything Model
- `dinov2` - DINOv2
- `siglip` - SigLIP
- `vit` - Vision Transformer

查看 `configs/model/` 获取完整列表

### 4. 🔬 可用的分析方法

#### 单帧分析（9种）
对每帧单独计算，输出平均值到 `averaged_summary.json`：
- `pca_vis` - PCA特征可视化
- `avg_token_cos` - 平均token余弦相似度
- `dist_sim_decay` - 距离-相似度衰减曲线
- `mean_token_norm` - 平均token范数
- `neighbor_sim` - 邻居相似度
- `token_cov_rank` - Token协方差秩
- `token_norm_entropy` - Token范数熵
- `token_norm_var` - Token范数方差
- `token_to_global` - Token到全局相似度
- `within_between_var` - 样本内/样本间方差比
- `frequency_metrics` - 频域指标（低中高频能量占比，频谱熵，频谱中心，频谱带宽）

#### 时序分析（10种）
需要多帧，分析时间动态：
- `temporal_smoothness` - 时序平滑度
- `temporal_cosine_shift` - 时序余弦偏移
- `lag_distance_curve` - 滞后距离曲线
- `temporal_variance` - 时序方差
- `temporal_effective_rank` - 时序有效秩
- `temporal_spectral_entropy` - 时序谱熵
- `autocorrelation` - 自相关
- `total_trajectory_variation` - 总轨迹变化
- `patch_temporal_smoothness` - patch时序平滑度
- `temporal_token_norm_entropy` - 时序token范数熵
- `trajectory_var_ratio` - 轨迹方差比

### 5. ⚙️ 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `model` | 模型名称 | dinov3 |
| `analysis` | 分析方法 | **必需** |
| `video_path` | 视频路径 | **必需** |
| `frame_start` | 起始帧索引 | 0 |
| `frame_end` | 结束帧索引 | 视频末尾 |
| `stride` | 采样步长 | 1 |
| `batch_size` | 批处理大小 | 8 |
| `output_dir` | 输出目录 | ./output |
| `device` | 计算设备 | cuda |
| `num_bins` | 直方图bin数（仅temporal_token_norm_entropy） | 16 |

### 6. 📂 输出结构

#### 单帧分析
```
output/dinov3_avg_token_cos/
├── frame_0000/
│   ├── reference.png       # 原始帧
│   ├── avg_token_cos.png   # 可视化
│   └── summary.json        # 该帧指标
├── frame_0001/
│   └── ...
└── averaged_summary.json   # 所有帧的平均指标 ⭐
```

#### 时序分析
```
output/dinov3_temporal_smoothness/
├── temporal_smoothness.png  # 时序曲线
└── summary.json             # 详细指标
```

#### PCA特殊处理
```
output/dinov3_pca_vis/
├── frame_0000/
│   ├── reference.png
│   └── pca_vis.png          #  只有第一帧有PCA
├── frame_0001/
│   └── reference.png        #  其他帧无PCA
└── averaged_summary.json
```

### 7. ⚠️ 重要提示

 **必须注意的事项：**

1. **只支持视频输入**：不提供图片目录参数，所有测试都从视频提取帧
2. **每次只运行一种方法**：通过 `analysis` 参数指定
3. **PCA只在第一帧保存**：其他帧会提取特征但不生成PCA图
4. **单帧分析输出平均值**：所有帧的指标会平均后保存到 `averaged_summary.json`
5. **时序分析至少需要2帧**：否则会报错
6. **大视频使用stride**：减少处理的帧数加快速度
7. **内存不足减小batch_size**：默认8，可改为4或2

### 8. ❓ 常见问题

**Q: CUDA out of memory?**
```bash
# 减小批处理大小或使用stride
python run.py model=dinov3 analysis=avg_token_cos video_path=test.mp4 batch_size=4
python run.py model=dinov3 analysis=temporal_smoothness video_path=test.mp4 stride=5
```

**Q: 如何只测试特定时间段？**
```bash
python run.py model=dinov3 analysis=temporal_variance \
    video_path=test.mp4 frame_start=50 frame_end=150
```

**Q: 如何比较不同模型？**
```bash
for model in dinov3 clip sam; do
    python run.py model=$model analysis=temporal_smoothness video_path=test.mp4
done
```

### 9. 📚 完整文档

查看更多详细信息请阅读 [README.md](README.md)
