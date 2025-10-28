# 🚀 和弦识别训练完整命令集合（Windows 版本）

## 📋 目录
1. [环境检查](#1-环境检查)
2. [数据验证](#2-数据验证)
3. [训练命令](#3-训练命令)
4. [测试与预测](#4-测试与预测)
5. [特征对比](#5-特征对比)
6. [故障排除](#6-故障排除)

---

## 1. 环境检查

### 1.1 激活 conda 环境
```powershell
conda activate librosa
```

### 1.2 检查所有依赖包
```powershell
conda list | Select-String -Pattern "torch|audio|librosa|numpy|scipy"
```

### 1.3 完整环境诊断
```powershell
python -c "import sys, torch, torchaudio, librosa, numpy; print('='*60); print('环境诊断报告'); print('='*60); print(f'Python: {sys.version.split()[0]}'); print(f'PyTorch: {torch.__version__}'); print(f'torchaudio: {torchaudio.__version__}'); print(f'librosa: {librosa.__version__}'); print(f'numpy: {numpy.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print('='*60)"
```

### 1.4 测试音频加载（重要！）
```powershell
python -c "import torchaudio, os; test_files = [f for f in os.listdir('single_chords_output') if f.endswith('.wav')]; test_file = test_files[0] if test_files else None; wav_path = os.path.join('single_chords_output', test_file) if test_file else None; waveform, sr = torchaudio.load(wav_path) if wav_path else (None, None); print(f'✓ 音频加载成功: {waveform.shape}, SR={sr}') if waveform is not None else print('✗ 音频加载失败')"
```

**注意**: 如果上述命令报错，请确保安装了 ffmpeg：
```powershell
conda install -c conda-forge ffmpeg
```

---

## 2. 数据验证

### 2.1 检查 WAV 文件数量
```powershell
(Get-ChildItem single_chords_output\*.wav).Count
```

### 2.2 检查文件命名格式
```powershell
Get-ChildItem single_chords_output\*.wav | Select-Object -First 10 | Select-Object Name
```

### 2.3 验证文件完整性
```powershell
python -c "from pathlib import Path; wav_dir = Path('single_chords_output'); wav_files = list(wav_dir.glob('*.wav')); print(f'总文件数: {len(wav_files)}'); print(f'总大小: {sum(f.stat().st_size for f in wav_files) / 1024**2:.2f} MB'); from train_chord_recognition import LabelExtractor; valid = 0; invalid = []; [valid := valid + 1 if not invalid.append(wav_file.name) and LabelExtractor.parse_filename(wav_file.name) else invalid.append(wav_file.name) for wav_file in wav_files[:50]]; print(f'文件名格式检查: {valid}/50 有效'); print(f'无效文件: {invalid[:5]}') if invalid else None"
```

---

## 3. 训练命令

### 3.1 训练根音识别（Root - 7 类，最简单）⭐ 推荐先运行
```powershell
python train_chord_recognition.py `
    --data_dir single_chords_output `
    --task root `
    --epochs 300 `
    --batch_size 32 `
    --lr 0.001 `
    --output_dir models_root_stft
```

**预期效果**:
- 训练样本: ~1568
- 验证样本: ~392
- 类别数: 7 (A, B, C, D, E, F, G)
- 预期准确率: 85-95%

### 3.2 训练和弦类型识别（Chord - 11 类）
```powershell
python train_chord_recognition.py `
    --data_dir single_chords_output `
    --task chord `
    --epochs 300 `
    --batch_size 32 `
    --lr 0.001 `
    --output_dir models_chord_stft
```

**预期效果**:
- 类别数: 14 (major, minor, dim, aug, dom7, maj7, min7, dim7, hdi, sus2, sus4, 6, 9, add9)
- 预期准确率: 75-85%

### 3.3 训练完整和弦识别（Full - 77 类，最难）
```powershell
python train_chord_recognition.py `
    --data_dir single_chords_output `
    --task full `
    --epochs 500 `
    --batch_size 32 `
    --lr 0.001 `
    --output_dir models_full_stft
```

**预期效果**:
- 类别数: 98 (7 roots × 14 chord types)
- 预期准确率: 60-75%
- 需要更多 epochs 和更小学习率

### 3.4 使用 CQT 特征训练（最佳音乐识别效果）⭐⭐⭐
```powershell
python train_chord_cqt.py `
    --data_dir single_chords_output `
    --task root `
    --epochs 50 `
    --batch_size 32 `
    --lr 0.001 `
    --output_dir models_cqt
```

**预期效果**:
- 比 STFT 准确率提升 5-10%
- 训练速度较慢（CPU-based librosa）
- 对移调更鲁棒

---

## 4. 测试与预测

### 4.1 单文件预测（STFT）
```powershell
python predict_chord.py `
    --wav_file single_chords_output\C_maj_satb_0001.wav `
    --model models_stft\chord_model_root.pth `
    --mappings models_stft\label_mappings_root.json
```

### 4.2 单文件预测（CQT）
```powershell
python predict_chord_cqt.py `
    --wav_file single_chords_output\C_maj_satb_0001.wav `
    --model models_cqt\chord_model_cqt_root.pth `
    --mappings models_cqt\label_mappings_root.json
```

### 4.3 批量测试模型
```powershell
python test_model.py `
    --model_path models_stft\chord_model_root.pth `
    --data_dir single_chords_output `
    --output_dir test_results
```

**输出**:
- `classification_report.txt` - 详细分类报告
- `confusion_matrix.png` - 混淆矩阵
- `confidence_distribution.png` - 置信度分布
- `error_samples.json` - 错误样本分析

---

## 5. 特征对比

### 5.1 可视化对比 STFT vs Mel vs CQT
```powershell
python compare_features.py `
    --wav_file single_chords_output\C_maj_satb_0001.wav `
    --output feature_comparison.png
```

**查看图片**:
```powershell
Start-Process feature_comparison.png
```

### 5.2 自定义参数对比
```powershell
python compare_features.py `
    --wav_file single_chords_output\G_dom_satb_0001.wav `
    --n_fft 4096 `
    --n_mels 256 `
    --n_bins 96 `
    --output comparison_custom.png
```

---

## 6. 故障排除

### 6.1 如果提示 "FFmpeg not found"
```powershell
# 安装 FFmpeg
conda install -c conda-forge ffmpeg

# 验证安装（PowerShell）
Get-Command ffmpeg
ffmpeg -version
```

### 6.2 如果提示 "CUDA not available" 且有 NVIDIA GPU
```powershell
# 检查 CUDA 支持
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}') if torch.cuda.is_available() else print('CUDA not built')"

# 如果不支持，使用 CPU
python train_chord_recognition.py `
    --data_dir single_chords_output `
    --task root `
    --epochs 30 `
    --device cpu `
    --output_dir models_stft
```

### 6.3 如果内存不足（OOM）
```powershell
# 减小 batch size
python train_chord_recognition.py `
    --data_dir single_chords_output `
    --task root `
    --batch_size 16 `
    --epochs 30 `
    --output_dir models_stft
```

### 6.4 如果训练太慢
```powershell
# 减少 epochs 快速验证
python train_chord_recognition.py `
    --data_dir single_chords_output `
    --task root `
    --epochs 5 `
    --output_dir models_test
```

### 6.5 清理旧模型
```powershell
# 删除旧训练结果
Remove-Item -Path models_stft, models_cqt, test_results -Recurse -Force -ErrorAction SilentlyContinue

# 重新创建目录
New-Item -ItemType Directory -Path models_stft, models_cqt, test_results -Force
```

---

## 📊 训练监控

### 实时查看训练日志
```powershell
# 如果训练在后台运行
Get-Content training.log -Wait -Tail 50
```

### 检查 GPU 使用率
```powershell
# NVIDIA GPU 监控（如果有 nvidia-smi）
nvidia-smi

# 或使用任务管理器（性能 -> GPU）
taskmgr
```

### 查看模型文件
```powershell
Get-ChildItem models_stft\ | Format-Table Name, Length, LastWriteTime
```

---

## 🎯 推荐工作流

### 快速验证流程（5 分钟）
```powershell
# 1. 环境检查
conda activate librosa
python -c "import torch, torchaudio; print('✓ 环境正常')"

# 2. 快速训练（5 epochs）
python train_chord_recognition.py `
    --data_dir single_chords_output `
    --task root `
    --epochs 5 `
    --output_dir models_test

# 3. 测试预测
python predict_chord.py `
    --wav_file single_chords_output\C_maj_satb_0001.wav `
    --model models_test\chord_model_root.pth `
    --mappings models_test\label_mappings_root.json
```

### 完整训练流程（30-60 分钟）
```powershell
# 1. STFT 训练（推荐）
python train_chord_recognition.py `
    --data_dir single_chords_output `
    --task root `
    --epochs 30 `
    --output_dir models_stft

# 2. CQT 训练（最佳效果）
python train_chord_cqt.py `
    --data_dir single_chords_output `
    --task root `
    --epochs 50 `
    --output_dir models_cqt

# 3. 对比测试
python test_model.py --model_path models_stft\chord_model_root.pth --data_dir single_chords_output
python test_model.py --model_path models_cqt\chord_model_cqt_root.pth --data_dir single_chords_output

# 4. 特征对比
python compare_features.py --wav_file single_chords_output\C_maj_satb_0001.wav --output comparison.png
Start-Process comparison.png
```

---

## 📝 命令参数说明

### train_chord_recognition.py
- `--data_dir`: WAV 文件目录
- `--task`: 任务类型 (root|chord|full)
- `--epochs`: 训练轮数 (推荐 30-100)
- `--batch_size`: 批次大小 (推荐 16-32)
- `--lr`: 学习率 (推荐 0.0005-0.001)
- `--n_fft`: FFT 窗口大小 (默认 2048)
- `--device`: 设备 (cuda|cpu)
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

## 💡 Windows 特定提示

### 1. PowerShell 多行命令
在 PowerShell 中，使用反引号（`` ` ``）来续行，而不是 bash 的反斜杠（`\`）。

### 2. 路径分隔符
Windows 使用反斜杠（`\`）作为路径分隔符，但 Python 也接受正斜杠（`/`）。

### 3. 查找命令
- `which` → `Get-Command`
- `ls` → `Get-ChildItem` 或 `ls`（PowerShell 别名）
- `grep` → `Select-String`
- `wc -l` → `(Get-ChildItem).Count`
- `open` → `Start-Process`
- `rm -rf` → `Remove-Item -Recurse -Force`

### 4. 环境变量
```powershell
# 查看环境变量
$env:PATH

# 临时设置环境变量
$env:CUDA_VISIBLE_DEVICES = "0"
```

---

## ✅ 成功标志

训练成功后你应该看到：

```
============================================================
Training Completed! Best Val Acc: 95.23%
============================================================

✓ 标签映射保存到: models_stft\label_mappings_root.json
✓ 模型保存到: models_stft\chord_model_root.pth
✓ 训练历史保存到: models_stft\training_history_root.png

最佳验证准确率: 95.23%
```

---

**最后更新**: 2025-10-27  
**作者**: GitHub Copilot  
**项目**: AutoChord 和弦识别系统（Windows 版本）
