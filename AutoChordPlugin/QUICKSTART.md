# AutoChord JUCE插件 - 快速开始指南

## 🎯 目标

创建一个实时和弦识别JUCE插件，读取音频信号并使用训练好的LibTorch模型进行推理。

## 📋 准备工作

### 1. 导出模型为TorchScript格式

```powershell
# 导出所有模型（root, chord, full）
python export_models_for_juce.py --export_all
```

这将在各模型目录下生成`.pt`文件：
- `models_root_stft/root_model.pt`
- `models_chord_stft/chord_model.pt`
- `models_full_stft/full_model.pt`

**输入格式**：`[1, 1, 1025, T]`
- 1025：原始STFT频谱bins数（FFT_SIZE/2 + 1）
- T：时间帧数（取决于音频长度）

### 2. 检查依赖

确保以下目录存在：
- `JUCE/` - JUCE框架
- `libtorch/` - LibTorch库
- 各模型目录下的`.pt`文件

## 🔨 构建插件

### Windows (Visual Studio)

```powershell
cd AutoChordPlugin

# 创建构建目录
mkdir build
cd build

# 配置CMake
cmake .. -G "Visual Studio 16 2019" -A x64

# 编译（Release模式）
cmake --build . --config Release

# 或在Visual Studio中打开.sln文件编译
```

### 编译选项

- **Debug模式**：`cmake --build . --config Debug`
- **仅VST3**：修改CMakeLists.txt中的`FORMATS VST3`
- **仅Standalone**：修改为`FORMATS Standalone`

## 🎵 使用插件

### 在DAW中使用

1. 编译完成后，插件自动安装到VST3目录
2. 打开DAW（Reaper/Ableton/FL Studio等）
3. 在音频轨道上加载"AutoChordPlugin"
4. 选择识别模式（Root/Chord/Full）
5. 播放音频查看实时识别结果

### Standalone模式

```powershell
# 直接运行可执行文件
.\build\AutoChordPlugin_artefacts\Release\Standalone\AutoChordPlugin.exe
```

## 🎨 插件界面

```
┌────────────────────────────────────────┐
│   AutoChord - Real-time Recognition   │
├────────────────────────────────────────┤
│  [Model: Full ▼]  [Load Models]       │
├────────────────────────────────────────┤
│                                        │
│           C_major                      │
│         (大字体显示)                    │
│                                        │
│      Confidence: 87.5%                 │
├────────────────────────────────────────┤
│  C_major     ████████████ 87.5%       │
│  C_minor     ████ 5.2%                │
│  G_major     ███ 3.1%                 │
│  ... (Top 10)                          │
└────────────────────────────────────────┘
```

## 🔧 技术架构

### 音频处理流程

```
音频输入 (44.1kHz)
    ↓
循环缓冲区 (2秒)
    ↓
STFT (FFT=2048, Hop=512)
    ↓
原始频谱 (1025 bins) + dB转换
    ↓
LibTorch推理
    ↓
Softmax → 和弦标签 + 置信度
    ↓
GUI显示 (30Hz刷新)
```

**注意**：本项目使用**原始STFT频谱**（不转换为Mel频谱），与训练代码保持一致。

### 三种识别模式

| 模式 | 类别数 | 说明 | 训练样本 | 准确率 |
|------|--------|------|---------|--------|
| Root | 7 | 识别根音（C,D,E,F,G,A,B） | 每类880个 | 85-95% |
| Chord | 11 | 和弦类型（major, minor, dim, aug, sus2, sus4, maj7, min7, dom7, dim7, hdim7） | 每类560个 | 60-75% |
| Full | 77 | 完整和弦识别（7根音×11类型） | 每类80个 | 60-70% |

## 📊 性能优化

### CPU优化
- 使用LibTorch CPU版本（默认）
- 降低推理频率（修改HOP_LENGTH）
- 减小缓冲区大小

### GPU加速（可选）
1. 下载LibTorch CUDA版本
2. 修改`ChordRecognizer.cpp`添加CUDA支持：
```cpp
if (torch::cuda::is_available()) {
    model.to(torch::kCUDA);
}
```

## 🐛 故障排除

### 模型加载失败
```
错误：Failed to load model
解决：检查.pt文件路径，确保使用export_models_for_juce.py导出
```

### 编译错误：找不到LibTorch
```
错误：Could NOT find Torch
解决：确保CMAKE_PREFIX_PATH指向正确的libtorch目录
```

### 推理无结果
```
问题：GUI显示"--"
解决：
1. 检查音频输入是否有信号
2. 查看Debug输出（DBG消息）
3. 确认模型已成功加载
```

## 📊 数据集说明

使用`generate_single_chords.py`生成的数据集：
- **7个根音** × **11种和弦** × **8种配器** × **10次重复** = **6160个样本**
- 每个样本包含智能音符省略、加倍、声部对调等音乐性变化
- 训练集/验证集划分：80% / 20%

## 🎓 代码结构

```
AutoChordPlugin/
├── CMakeLists.txt          # 构建配置
├── Source/
│   ├── ChordRecognizer.h/cpp   # 核心识别引擎
│   ├── PluginProcessor.h/cpp   # 音频处理器
│   └── PluginEditor.h/cpp      # GUI界面
└── README.md               # 详细文档
```

## 🚀 后续开发

可扩展功能：
- [ ] 添加和弦进行记录
- [ ] MIDI输出功能
- [ ] 频谱图可视化
- [ ] 自定义模型加载
- [ ] 多语言支持

## 📝 关键代码片段

### 加载模型
```cpp
chordRecognizer_.loadModel(
    ChordRecognizer::ModelType::FULL,
    juce::File("models_full_stft/full_model.pt")
);
```

### 实时处理
```cpp
void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) override {
    chordRecognizer_.processAudioBuffer(buffer, getSampleRate());
}
```

### 获取结果
```cpp
auto chord = chordRecognizer_.getCurrentChord();
auto confidence = chordRecognizer_.getCurrentConfidence();
```

## ✅ 验证清单

构建前检查：
- [ ] 已运行`export_models_for_juce.py --export_all`
- [ ] JUCE路径正确
- [ ] LibTorch路径正确
- [ ] Visual Studio已安装

构建后检查：
- [ ] VST3文件生成
- [ ] Standalone可执行文件运行
- [ ] 模型文件自动复制
- [ ] DAW中可加载插件

## 📞 支持

遇到问题？
1. 查看`AutoChordPlugin/README.md`详细文档
2. 检查CMake配置输出
3. 启用Debug模式查看DBG日志
4. 提交Issue附带错误信息

---

**祝你构建成功！🎉**
