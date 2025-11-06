cat << 'EOF' > RUN_COMMANDS.md
# 🚀 和弦识别训练完整命令集合

## 📋 目录
0. [MIDI数据生成](#0-midi数据生成)
1. [环境检查](#1-环境检查)
2. [数据验证](#2-数据验证)
3. [训练命令](#3-训练命令)
4. [测试与预测](#4-测试与预测)
5. [导出模型](#5-导出模型)
6. [特征对比](#6-特征对比)
7. [故障排除](#7-故障排除)

---

## 0. MIDI数据生成

### 0.1 生成单和弦MIDI文件（推荐第一步）
```bash
python generate_single_chords.py -r 10 -o single_chords
```

**参数说明**：
- `-r 10`：每种和弦生成10个重复（带音符变化和声部对调）
- `-o single_chords`：输出到single_chords目录

**生成数据说明**：
- **根音数量**：7个（C, D, E, F, G, A, B）
- **和弦类型**：11种（major, minor, dim, aug, sus2, sus4, maj7, min7, dom7, dim7, hdim7）
- **配器方案**：8种（satb, sat, atb, sa, piano, piano_bass, strings, full）
- **总和弦组合**：7 × 11 × 8 = **616种**
- **文件夹总数**（重复10次）：616 × 10 = **6160个**
- **实际生成**：✅ 已生成6160个MIDI文件夹，6160个WAV文件（约6.1GB）

**生成目录结构示例**：
```
single_chords/
  ├── C_major_satb_01/          # C大三和弦，SATB配置，第1次重复
  │   ├── Soprano.mid
  │   ├── Alto.mid
  │   ├── Tenor.mid
  │   └── Bass.mid
  ├── C_major_satb_02/          # 相同和弦，第2次重复（不同声部排列）
  │   ├── Soprano.mid
  │   ├── Alto.mid
  │   ├── Tenor.mid
  │   └── Bass.mid
  ├── D_min7_piano_01/          # D小七和弦，钢琴配置
  │   ├── Piano_RH.mid
  │   └── Piano_LH.mid
  └── ... (共6160个文件夹)
```

**文件夹命名规则**：`根音_和弦类型_配器方案_序号`

**配器方案详细说明**：
| 方案 | 声部组合 | 说明 |
|------|---------|------|
| satb | Soprano + Alto + Tenor + Bass | 四部合唱（完整） |
| sat | Soprano + Alto + Tenor | 三部合唱（无低音） |
| atb | Alto + Tenor + Bass | 三部合唱（无高音） |
| sa | Soprano + Alto | 二部合唱 |
| piano | Piano_RH + Piano_LH | 钢琴左右手 |
| piano_bass | Piano_RH + Piano_LH + Bass | 钢琴+低音 |
| strings | Strings + Bass | 弦乐+低音 |
| full | Soprano + Alto + Tenor + Bass + Strings | 完整编制 |

**音乐性变化机制**：
- ✅ 智能音符省略（根据和弦重要性）
- ✅ 音符加倍（根音、五音优先）
- ✅ 八度调整（适应声部音域）
- ✅ 声部对调（S-A对调、T-B对调、完全重排等）
- ✅ 随机力度和时间微调（增加表现力）

### 0.2 批量渲染MIDI为WAV（使用Reaper + Lua脚本）

1. 打开Reaper DAW
2. 加载与配器方案匹配的轨道模板（至少8轨）
3. 运行`midi_render.lua`脚本
4. 脚本会自动渲染混音为WAV并输出到`single_chords_output/`

**预期WAV文件数量**：与MIDI文件夹数相同（如重复10次则6160个WAV）

---

## 1. 环境检查

### 1.1 激活 conda 环境
```bash
conda activate librosa
```

### 1.2 检查所有依赖包
```bash
conda list | grep -E "torch|audio|librosa|numpy|scipy"
```

### 1.3 完整环境诊断
```bash
python -c "
import sys, torch, torchaudio, librosa, numpy
print('='*60)
print('环境诊断报告')
print('='*60)
print(f'Python: {sys.version.split()[0]}')
print(f'PyTorch: {torch.__version__}')
print(f'torchaudio: {torchaudio.__version__}')
print(f'librosa: {librosa.__version__}')
print(f'numpy: {numpy.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print('='*60)
"
```

### 1.4 测试音频加载（重要！）
```bash
python -c "
import torchaudio, os
test_file = [f for f in os.listdir('single_chords_output') if f.endswith('.wav')][0]
wav_path = os.path.join('single_chords_output', test_file)
try:
    waveform, sr = torchaudio.load(wav_path)
    print(f'✓ 音频加载成功: {waveform.shape}, SR={sr}')
except Exception as e:
    print(f'✗ 音频加载失败: {e}')
    print('解决方案: conda install -c conda-forge ffmpeg')
"
```

---

## 2. 数据验证

### 2.1 检查 WAV 文件数量
```bash
ls single_chords_output/*.wav | wc -l
```

### 2.2 检查文件命名格式
```bash
ls single_chords_output/*.wav | head -10
```

### 2.3 验证文件完整性
```bash
python -c "
from pathlib import Path
import os

wav_dir = Path('single_chords_output')
wav_files = list(wav_dir.glob('*.wav'))

print(f'总文件数: {len(wav_files)}')
print(f'总大小: {sum(f.stat().st_size for f in wav_files) / 1024**2:.2f} MB')

# 检查文件名格式
from train_chord_stft import LabelExtractor
valid = 0
invalid = []
for wav_file in wav_files[:50]:  # 检查前50个
    try:
        LabelExtractor.parse_filename(wav_file.name)
        valid += 1
    except Exception as e:
        invalid.append(wav_file.name)

print(f'文件名格式检查: {valid}/50 有效')
if invalid:
    print(f'无效文件: {invalid[:5]}')
"
```

---

## 3. 训练命令

### 3.1 训练根音识别（Root - 7 类，最简单）⭐ 推荐先运行
```bash
python train_chord_stft.py \
    --data_dir single_chords_output \
    --task root \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.001 \
    --output_dir models_root_stft
```

**预期效果**:
- 训练样本: ~4928 (80%分割)
- 验证样本: ~1232 (20%分割)
- 类别数: 7 (A, B, C, D, E, F, G)
- 预期准确率: 85-95%

### 3.2 训练和弦类型识别（Chord - 11 类）
```bash
python train_chord_stft.py \
    --data_dir single_chords_output \
    --task chord \
    --epochs 300 \
    --batch_size 32 \
    --lr 0.001 \
    --output_dir models_chord_stft
```

**预期效果**:
- 训练样本: ~4928
- 验证样本: ~1232
- 类别数: 11 (aug, dim, dim7, dom7, hdim7, maj7, major, min7, minor, sus2, sus4)
- 预期准确率: 75-85%

### 3.3 训练完整和弦识别（Full - 77 类，最难）
```bash
python train_chord_stft.py \
    --data_dir single_chords_output \
    --task full \
    --epochs 1000 \
    --batch_size 32 \
    --lr 0.001 \
    --output_dir models_full_stft
```

**预期效果**:
- 训练样本: ~4928
- 验证样本: ~1232
- 类别数: 77 (7 roots × 11 chord types)
- 预期准确率: 60-75%
- 需要更多 epochs 和更小学习率

### 3.4 使用 CQT 特征训练（最佳音乐识别效果）⭐⭐⭐
```bash
python train_chord_cqt.py \
    --data_dir single_chords_output \
    --task root \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001 \
    --output_dir models_cqt
```

**预期效果**:
- 比 STFT 准确率提升 5-10%
- 训练速度较慢（CPU-based librosa）
- 对移调更鲁棒

---

## 4. 测试与预测

### 4.1 单文件预测（STFT）
```bash
python predict_chord_stft.py \
    --wav_file single_chords_output/C_major_satb_01.wav \
    --model models_root_stft/chord_model_root.pth \
    --mappings models_root_stft/label_mappings_root.json
```

### 4.2 单文件预测（CQT）
```bash
python predict_chord_cqt.py \
    --wav_file single_chords_output/C_major_satb_01.wav \
    --model models_cqt/chord_model_cqt_root.pth \
    --mappings models_cqt/label_mappings_root.json
```

### 4.3 批量测试模型
```bash
python test_model.py \
    --data_dir single_chords_output \
    --model models_root_stft/chord_model_root.pth \
    --mappings models_root_stft/label_mappings_root.json \
    --output_dir test_results
```

**输出**:
- `classification_report.txt` - 详细分类报告
- `confusion_matrix.png` - 混淆矩阵
- `confidence_distribution.png` - 置信度分布
- `error_samples.json` - 错误样本分析

---

## 5. 导出模型（用于JUCE插件）

### 5.1 导出所有模型（推荐）
```bash
python export_models_for_juce.py --export_all
```

**输出文件**：
- `root_model.pt` - 根音识别模型（7类）
- `chord_model.pt` - 和弦类型识别模型（11类）
- `full_model.pt` - 完整和弦识别模型（77类）

**模型输入格式**：
- 输入形状：`[1, 1, 1025, T]`（批次大小，通道数，频率bin数，时间帧数）
- 数据类型：`torch.float32`
- 频谱类型：原始STFT频谱（20*log10转换为dB）

### 5.2 单独导出模型
```bash
# 导出根音识别模型
python export_models_for_juce.py \
    --model_type root \
    --model_path models_full_stft/chord_model_full_20251028_113548.pth \
    --output_path root_model.pt

# 导出和弦类型识别模型
python export_models_for_juce.py \
    --model_type chord \
    --model_path models_chord_stft/chord_model_chord_20251028_114129.pth \
    --output_path chord_model.pt

# 导出完整和弦识别模型
python export_models_for_juce.py \
    --model_type full \
    --model_path models_full_stft/chord_model_full_20251028_114225.pth \
    --output_path full_model.pt
```

**注意事项**：
- 导出的模型为TorchScript格式（.pt），可在JUCE插件中通过LibTorch加载
- 确保模型路径正确指向训练好的.pth文件
- 导出后的模型应放置在JUCE插件的Resources目录中

---

## 6. 特征对比

### 6.1 可视化对比 STFT vs Mel vs CQT
```bash
python compare_features.py \
    --wav_file single_chords_output/C_major_satb_01.wav \
    --output feature_comparison.png
```

**查看图片**:
```bash
open feature_comparison.png
```

### 6.2 自定义参数对比
```bash
python compare_features.py \
    --wav_file single_chords_output/G_dom7_satb_01.wav \
    --n_fft 4096 \
    --n_mels 256 \
    --n_bins 96 \
    --output comparison_custom.png
```

---

## 7. 故障排除

### 7.1 如果提示 "FFmpeg not found"
```bash
# 安装 FFmpeg
conda install -c conda-forge ffmpeg

# 验证安装
which ffmpeg
ffmpeg -version | head -1
```

### 7.2 如果提示 "MPS not available"
```bash
# 检查 MPS 支持
python -c "
import torch
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
"

# 如果不支持，使用 CPU
python train_chord_stft.py \
    --data_dir single_chords_output \
    --task root \
    --epochs 30 \
    --device cpu \
    --output_dir models_stft
```

### 7.3 如果内存不足（OOM）
```bash
# 减小 batch size
python train_chord_stft.py \
    --data_dir single_chords_output \
    --task root \
    --batch_size 16 \
    --epochs 30 \
    --output_dir models_stft
```

### 7.4 如果训练太慢
```bash
# 减少 epochs 快速验证
python train_chord_stft.py \
    --data_dir single_chords_output \
    --task root \
    --epochs 5 \
    --output_dir models_test
```

### 7.5 清理旧模型
```bash
# 删除旧训练结果
rm -rf models_root_stft models_chord_stft models_full_stft models_cqt test_results

# 重新创建目录
mkdir -p models_root_stft models_chord_stft models_full_stft models_cqt test_results
```

---

## 📊 训练监控

### 实时查看训练日志
```bash
# 如果训练在后台运行
tail -f training.log
```

### 检查 GPU/MPS 使用率
```bash
# macOS MPS 监控
while true; do 
    python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
    sleep 5
done
```

### 查看模型文件
```bash
ls -lh models_root_stft/
ls -lh models_chord_stft/
ls -lh models_full_stft/
```

---

## 🎯 推荐工作流

### 快速验证流程（5 分钟）
```bash
# 1. 环境检查
conda activate librosa
python -c "import torch, torchaudio; print('✓ 环境正常')"

# 2. 快速训练（5 epochs）
python train_chord_stft.py \
    --data_dir single_chords_output \
    --task root \
    --epochs 5 \
    --output_dir models_test

# 3. 测试预测
python predict_chord.py \
    --wav_file single_chords_output/C_major_satb_01.wav \
    --model models_test/chord_model_root.pth \
    --mappings models_test/label_mappings_root.json
```

### 完整训练流程（30-60 分钟）
```bash
# 1. STFT 训练（推荐）
python train_chord_stft.py \
    --data_dir single_chords_output \
    --task root \
    --epochs 30 \
    --output_dir models_stft

# 2. CQT 训练（最佳效果）
python train_chord_cqt.py \
    --data_dir single_chords_output \
    --task root \
    --epochs 50 \
    --output_dir models_cqt

# 3. 对比测试
python test_model.py \
    --data_dir single_chords_output \
    --model models_root_stft/chord_model_root.pth \
    --mappings models_root_stft/label_mappings_root.json \
    --output_dir test_results_stft

python test_model.py \
    --data_dir single_chords_output \
    --model models_cqt/chord_model_cqt_root.pth \
    --mappings models_cqt/label_mappings_root.json \
    --output_dir test_results_cqt

# 4. 特征对比
python compare_features.py --wav_file single_chords_output/C_major_satb_01.wav --output comparison.png
open comparison.png
```

---

## 📝 命令参数说明

### train_chord_stft.py
- `--data_dir`: WAV 文件目录
- `--task`: 任务类型 (root|chord|full)
- `--epochs`: 训练轮数 (推荐 30-100)
- `--batch_size`: 批次大小 (推荐 16-32)
- `--lr`: 学习率 (推荐 0.0005-0.001)
- `--n_fft`: FFT 窗口大小 (默认 2048)
- `--device`: 设备 (cuda|mps|cpu)
- `--output_dir`: 输出目录

### train_chord_cqt.py
- 参数同上，但使用 CQT 特征
- `--n_bins`: CQT bins 数 (默认 84)
- `--bins_per_octave`: 每八度 bins (默认 12)

### predict_chord.py / predict_chord_cqt.py
- `--wav_file`: 要预测的 WAV 文件
- `--model`: 模型文件路径 (.pth)
- `--mappings`: 标签映射文件 (.json)

### test_model.py
- `--model_path`: 模型路径
- `--data_dir`: 测试数据目录
- `--output_dir`: 结果输出目录

### compare_features.py
- `--wav_file`: 音频文件路径
- `--n_fft`: STFT FFT 大小
- `--n_mels`: Mel bins 数
- `--n_bins`: CQT bins 数
- `--output`: 输出图片路径

---

## ✅ 成功标志

训练成功后你应该看到：

```
============================================================
Training Completed! Best Val Acc: 95.23%
============================================================

✓ 标签映射保存到: models_root_stft/label_mappings_root.json
✓ 模型保存到: models_root_stft/chord_model_root_20251103_123456.pth
✓ 训练历史保存到: models_root_stft/training_history_root.png

最佳验证准确率: 95.23%
训练样本数: 4928
验证样本数: 1232
```

---

**最后更新**: 2025-11-03  
**作者**: GitHub Copilot  
**项目**: AutoChord 和弦识别系统

---

## 📊 实际数据统计

### MIDI生成统计
- **文件夹总数**: 6160个
- **和弦类型**: 11种（aug, dim, dim7, dom7, hdim7, maj7, major, min7, minor, sus2, sus4）
- **每种和弦**: 560个样本（7根音 × 8配器 × 10重复）

### WAV渲染统计
- **WAV文件总数**: 6160个
- **总大小**: 6.1GB
- **采样率**: 48000 Hz
- **声道**: 单声道（Mono）
- **文件命名**: `根音_和弦类型_配器_序号.wav`（如 `C_major_satb_01.wav`）

### 训练数据分割
- **训练集**: 4928个样本（80%）
- **验证集**: 1232个样本（20%）
- **类别分布**: 每类样本数量均衡
EOF

echo "✅ 命令集合已保存到 RUN_COMMANDS.md"
