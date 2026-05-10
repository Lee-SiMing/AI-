# AI-Music-Pipeline
```markdown
# AI-Music-Pipeline
https://github.com/Lee-SiMing/AI_Data_processing-Auto_lableing/blob/main/README.md
> 从原始音频到模型就绪数据的端到端自动化处理流水线，用于构建高质量、结构化标注的音乐数据集。

## 项目结构

AI-Music-Pipeline/
├── scripts/                    # 核心处理脚本
│   ├── dsp_preprocess.py       # 音频DSP预处理（重采样、标准化、切片、去重）
│   ├── audio_analyzer.py       # 音频特征分析（BPM、中国调式识别）
│   ├── panns_feature.py        # PANNs特征提取 + 手工标签合并
│   ├── anchor_tagging.py       # 基于锚点库的标签增强
│   └── prompt_generator.py     # DeepSeek 提示词泛化生成
└── requirements.txt            # Python依赖

## 处理流程

```
原始音频 → DSP预处理 → 特征分析 → PANNs标签 → 锚点增强 → 提示词生成
```

| 脚本 | 功能 |
|------|------|
| `dsp_preprocess.py` | 重采样、声道转换、音量标准化、静音裁剪、切片、去重 |
| `audio_analyzer.py` | 检测BPM（Slow/Medium/Fast）、中国调式（五声/七声） |
| `panns_feature.py` | 提取PANNs Embedding，添加手工标签（乐器/风格/情绪） |
| `anchor_tagging.py` | 与锚点库比对，自动继承标签 |
| `prompt_generator.py` | 调用DeepSeek生成音乐生成提示词 |

## 安装

```bash
pip install -r requirements.txt
```

额外依赖（PANNs模型权重）：下载 [Cnn14_mAP=0.431.pth](https://zenodo.org/record/3987831) 放入指定目录。

## 使用

每个脚本独立运行，使用前需修改脚本末尾的配置参数：

```bash
cd scripts
python dsp_preprocess.py        # 修改 USER_CONFIG 中的路径
python audio_analyzer.py        # 修改 TARGET_FOLDER / OUTPUT_JSON
python panns_feature.py         # 修改 INPUT_JSON / TARGET_FOLDER / OUTPUT_JSON
python anchor_tagging.py        # 修改 ANCHOR_JSON / TARGET_JSON / NEW_AUDIO_DIR
python prompt_generator.py      # 修改 INPUT_JSON_PATH / API_KEY
```

## 配置示例

`dsp_preprocess.py` 核心配置：

```python
USER_CONFIG = {
    "input_dir": "F:/raw/audio",      # 原始音频目录
    "output_dir": "E:/processed",     # 输出目录
    "sample_rate": 32000,             # 采样率
    "channels": 1,                    # 单声道
    "slice_enabled": True,            # 切片开关
    "slice_duration": 15.0,           # 每段15秒
}
```

## 输出

- `metadata.csv` — 所有音频的元数据（时长、响度、信噪比等）
- `report.txt` — 处理统计报告
- `clean/` — 标准化后的音频
- `slices/` — 切片后的音频段

## 许可证

MIT

## 项目目的

解决AI音乐生成模型缺乏大规模、高质量、结构化标注训练样本的问题。
```

---
