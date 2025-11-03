# AutoChord JUCE Plugin - 构建指南

## 项目概述

AutoChord是一个实时和弦识别JUCE插件，使用LibTorch加载训练好的模型，对音频信号进行实时推理。

## 功能特性

- ✅ 实时音频信号分析（非MIDI）
- ✅ 三种识别模式：
  - Root（7类根音识别）
  - Chord（11类和弦类型识别）
  - Full（77类完整和弦识别）
- ✅ 实时显示识别结果和置信度
- ✅ 可视化概率分布
- ✅ VST3 和 Standalone 格式

## 系统要求

### Windows
- Visual Studio 2019 或更新版本
- CMake 3.15+
- JUCE 7.x
- LibTorch (CPU或CUDA版本)

### 依赖项
- JUCE框架（已包含在`../JUCE`目录）
- LibTorch（已包含在`../libtorch`目录）
- 训练好的模型文件（`.pt`格式）

## 构建步骤

### 1. 准备模型文件

首先需要将训练好的PyTorch模型转换为TorchScript格式：

```python
# 在项目根目录运行
import torch
from train_chord_recognition import ChordCNN

# 加载并导出Root模型
model = ChordCNN(num_classes=7)
checkpoint = torch.load('models_root_stft/chord_model_root_xxx.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 创建示例输入（根据实际输入维度调整）
example_input = torch.randn(1, 1, 128, 173)  # [batch, channel, freq, time]
traced_model = torch.jit.trace(model, example_input)
traced_model.save('models_root_stft/root_model.pt')

# 同样方式导出Chord和Full模型
```

### 2. 配置CMake

```powershell
# 进入插件目录
cd AutoChordPlugin

# 创建构建目录
mkdir build
cd build

# 配置CMake（Windows + Visual Studio）
cmake .. -G "Visual Studio 16 2019" -A x64
```

### 3. 编译插件

```powershell
# 使用CMake编译
cmake --build . --config Release

# 或者打开生成的 .sln 文件在Visual Studio中编译
```

### 4. 安装插件

编译完成后，插件会自动复制到：
- VST3: `C:\Program Files\Common Files\VST3\AutoChordPlugin.vst3`
- Standalone: `build\AutoChordPlugin_artefacts\Release\Standalone\AutoChordPlugin.exe`

## 使用方法

### 在DAW中使用

1. 打开你的DAW（Reaper、Ableton Live、FL Studio等）
2. 加载AutoChordPlugin VST3插件到音频轨道
3. 插件会自动加载模型文件
4. 使用下拉菜单选择识别模式（Root/Chord/Full）
5. 播放音频，实时查看和弦识别结果

### Standalone模式

1. 直接运行`AutoChordPlugin.exe`
2. 选择音频输入设备
3. 选择识别模式
4. 实时监测和弦

## 插件界面说明

- **标题栏**：显示插件名称
- **模型选择器**：切换Root/Chord/Full识别模式
- **Load Models按钮**：手动重新加载模型
- **和弦显示区**：大字体显示当前识别的和弦
- **置信度显示**：显示识别置信度（0-100%）
- **概率分布条**：显示Top 10类别的概率分布

### 技术细节

### 音频处理流程

1. **音频采集**：插件接收实时音频流（44.1kHz采样率）
2. **缓冲**：维持2秒音频缓冲区
3. **STFT计算**：使用2048点FFT，512点hop length
4. **频谱处理**：使用原始STFT频谱（1025 bins），转换为dB scale
5. **模型推理**：LibTorch实时推理
6. **结果显示**：30Hz更新UI

**重要**：本插件使用**原始STFT频谱**（不使用Mel频谱），与训练代码保持完全一致。

### 模型架构

使用与训练相同的CNN架构：
```
Input [1, 1, 1025, T]  (原始STFT频谱)
  → Conv1 (32 filters) → ReLU → MaxPool
  → Conv2 (64 filters) → ReLU → MaxPool
  → Conv3 (128 filters) → ReLU → AdaptiveAvgPool
  → FC1 (256) → Dropout → FC2 (num_classes)
  → Softmax
```

**特征说明**：
- 使用原始STFT频谱（1025个频率bins）
- 不进行Mel频率转换
- dB scale归一化

### 性能优化

- 使用循环缓冲区减少内存分配
- 异步推理避免阻塞音频线程
- 使用LibTorch CPU优化（可选CUDA加速）

## 故障排除

### 模型加载失败

检查模型文件路径：
- 确保`.pt`文件在正确目录
- 检查文件权限
- 查看调试输出（DBG消息）

### 推理结果不准确

- 确保输入音频清晰，无过多噪音
- 检查采样率匹配（44.1kHz）
- 验证STFT参数与训练一致

### 性能问题

- 降低推理频率（修改`sampleCounter`阈值）
- 使用较小的模型（Root而非Full）
- 考虑使用CUDA版本LibTorch

## 开发调试

启用调试输出：

```cpp
// 在ChordRecognizer.cpp中添加
DBG("Current chord: " + currentChord_);
DBG("Confidence: " + juce::String(currentConfidence_));
```

## 后续改进

- [ ] 添加和弦历史记录
- [ ] 支持MIDI输出
- [ ] 添加更多可视化（波形、频谱图）
- [ ] 支持自定义模型路径
- [ ] 添加性能监控
- [ ] 支持macOS和Linux

## 许可证

本项目基于JUCE框架开发，请遵守JUCE许可证条款。

## 联系方式

如有问题或建议，请提交Issue或联系开发者。
