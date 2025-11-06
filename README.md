# 📚 AutoChord MIDI 数据集生成工具 - 文档索引

欢迎使用 AutoChord MIDI 数据集生成和渲染工具包！

---

## 🚀 快速开始

**第一次使用？正在导入 MIDI 文件？**

👉 **直接看这个:** [QUICK_START.md](./QUICK_START.md) - 手把手教你现在就要做什么!

**想了解完整流程？**

1. 📖 **[WORKFLOW.md](./WORKFLOW.md)** ← 从这里开始！
   - 完整工作流程概览
   - 端到端步骤说明
   - 时间估算和优化建议

2. 📝 **[REAPER_SETUP_DETAILED.md](./REAPER_SETUP_DETAILED.md)**
   - Reaper 模板详细设置步骤
   - 每一步都有截图和说明
   - 故障排查指南

3. ▶️ **开始生成数据**
   ```bash
   python3 generate_single_chords_multitrack.py -r 10  # 测试
   ```

---

## 📑 文档列表

### 🎵 生成器工具

| 文件 | 功能 | 输出 | 文档 |
|------|------|------|------|
| **generate_single_chords_multitrack.py** | 生成单和弦多轨道 MIDI | 700+ 个多声部和弦文件 | - |
| **generate_chord_progressions.py** | 生成和弦进行 MIDI（可自定义时长） | 100-2000+ 个和弦序列文件 | **[PROGRESSION_USAGE.md](./PROGRESSION_USAGE.md)** ⭐ |
| **generate_single_chords.py** | 旧版单和弦生成器 | 简化版单轨道文件 | - |

### 🎚️ 渲染工具

| 文件 | 功能 | 用途 |
|------|------|------|
| **batch_render_simple.lua** | 简化版批量渲染（推荐） | Reaper Lua 脚本 |
| **batch_render_midi.lua** | 完整版批量渲染 | 高级功能版本 |

### 📖 文档

| 文档 | 适合人群 | 特点 |
|------|---------|------|
| **[QUICK_START.md](./QUICK_START.md)** | 正在导入 MIDI | 🚀 立即行动指南,解决当前问题 |
| **[WORKFLOW.md](./WORKFLOW.md)** | 所有人 | 完整流程，一站式指南 |
| **[PROGRESSION_USAGE.md](./PROGRESSION_USAGE.md)** | 生成和弦进行 | 📚 完整使用指南：时长/模板/声部配置 |
| **[REAPER_SETUP_DETAILED.md](./REAPER_SETUP_DETAILED.md)** | 第一次设置 | 详细步骤，有检查清单 |
| **[MIDI_CHANNEL_EXPLAINED.md](./MIDI_CHANNEL_EXPLAINED.md)** | 有疑问 | ⚠️ 必须设置通道！解答核心疑问 |
| **[WHY_LUA_NO_CHANNEL_SETUP.md](./WHY_LUA_NO_CHANNEL_SETUP.md)** | 有困惑 | 🤔 为什么 Lua 脚本不设置通道？ |
| **[REAPER_QUICK_REF.md](./REAPER_QUICK_REF.md)** | 已经设置过 | 快速查询，速查表 |
| **[REAPER_VISUAL_GUIDE.md](./REAPER_VISUAL_GUIDE.md)** | 喜欢看图 | ASCII 图解，直观 |
| **[REAPER_SETUP_GUIDE.md](./REAPER_SETUP_GUIDE.md)** | 了解原理 | 技术背景，两种方法对比 |

---

## 🗺️ 使用路线图

### 路线 A：快速开始（推荐新手）

```
1. 阅读 WORKFLOW.md
   ↓
2. 按 REAPER_SETUP_DETAILED.md 设置 Reaper
   ↓
3. 生成测试数据（3 个文件）
   ↓
4. 手动导入 Reaper 验证
   ↓
5. 批量生成和渲染
```

**时间**：2-3 小时（包括学习）

### 路线 B：深入理解（推荐进阶）

```
1. 阅读 REAPER_SETUP_GUIDE.md（了解原理）
   ↓
2. 阅读 REAPER_VISUAL_GUIDE.md（理解架构）
   ↓
3. 按 REAPER_SETUP_DETAILED.md 设置
   ↓
4. 测试和优化
```

**时间**：3-4 小时

### 路线 C：极速模式（老手）

```
1. 看 REAPER_QUICK_REF.md
   ↓
2. 30 分钟设置完成
   ↓
3. 直接批量渲染
```

**时间**：1 小时

---

## 🎓 按问题查找

### "我想了解整体流程"
→ **[WORKFLOW.md](./WORKFLOW.md)**

### "我要设置 Reaper 模板"
→ **[REAPER_SETUP_DETAILED.md](./REAPER_SETUP_DETAILED.md)**

### "我忘了通道怎么设置"
→ **[REAPER_QUICK_REF.md](./REAPER_QUICK_REF.md)** - 第一页有表格

### "为什么要这样设置？"
→ **[REAPER_SETUP_GUIDE.md](./REAPER_SETUP_GUIDE.md)** - 原理说明

### "我看不懂文字，有图吗？"
→ **[REAPER_VISUAL_GUIDE.md](./REAPER_VISUAL_GUIDE.md)**

### "导入 MIDI 后没声音"
→ **[REAPER_SETUP_DETAILED.md](./REAPER_SETUP_DETAILED.md)** - 故障排查章节

### "Lua 脚本怎么用？"
→ **[WORKFLOW.md](./WORKFLOW.md)** - 阶段 3：批量渲染

### "想生成更多类型的数据"
→ **[WORKFLOW.md](./WORKFLOW.md)** - 扩展方向章节

---

## 🔧 工具功能对比

### MIDI 生成器对比

| 功能 | multitrack 版本 | 旧版本 | 和弦进行 |
|------|----------------|--------|---------|
| 多轨道 | ✅ 2-8 轨 | ❌ 1-2 轨 | ✅ 多轨 |
| 省略音 | ✅ | ❌ | ✅ |
| 多配器 | ✅ 6 种 | ❌ | ✅ 8 种 |
| 自定义时长 | ❌ 固定4秒 | ❌ 固定4秒 | ✅ 可自定义(默认2秒) |
| 和弦数量 | 1个 | 1个 | 可自定义(默认4个) |
| 进行模板 | ❌ | ❌ | ✅ 8种常见进行 |
| 适合训练 | ✅ 阶段一 | ⚠️ 简化版 | ✅ 阶段二(真实音乐场景) |
| 推荐使用 | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |

### Lua 渲染脚本对比

| 功能 | simple 版本 | 完整版 |
|------|------------|--------|
| 易用性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 自动路由 | ✅ | ✅ |
| 错误处理 | 基础 | 详细 |
| 推荐使用 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

**建议**：使用 `batch_render_simple.lua`！

---

## 📊 数据集规模参考

### 小规模测试（快速验证）

```bash
# 单和弦：140 个文件（14 种 × 10 次）
python3 generate_single_chords_multitrack.py -r 10

# 和弦进行：100 个文件
python3 generate_chord_progressions.py -n 100

# 渲染时间：约 20 分钟
```

### 中等规模（正常训练）

```bash
# 单和弦：700 个文件（默认）
python3 generate_single_chords_multitrack.py -r 50

# 和弦进行：500 个文件（默认）
python3 generate_chord_progressions.py -n 500

# 渲染时间：约 60-90 分钟
```

### 大规模（充足数据）

```bash
# 单和弦：1400 个文件
python3 generate_single_chords_multitrack.py -r 100

# 和弦进行：2000 个文件
python3 generate_chord_progressions.py -n 2000

# 渲染时间：2-4 小时
```

---

## 🎯 核心概念

### MIDI 通道路由

```
MIDI 文件 → 包含多个轨道 → 每个轨道使用不同 MIDI 通道（0-7）
                                          ↓
Reaper 音源轨道 → 设置监听特定通道 → 自动接收对应音符
```

**关键**：
- MIDI 通道 0 = Reaper Channel 1
- MIDI 通道 1 = Reaper Channel 2
- 依此类推...

### 多轨道的意义

```
单轨道：所有音符混在一起
    → 模型只能学习整体音响
    
多轨道：每个声部独立
    → 模型可以学习：
       • 声部分离
       • 音域特征
       • 配器差异
       → 更强大的识别能力
```

### 训练策略

```
阶段 1（基础）：单和弦
    → 学习 14 种和弦的基本特征
    → 数据：single_chord_multitrack/
    
阶段 2（进阶）：和弦进行
    → 学习和弦转换和时间序列
    → 数据：chord_progression_dataset/
    
阶段 3（混合）：所有数据
    → 综合能力
```

---

## ⚡ 常见问题 FAQ

### Q1: 需要什么软件？
- Python 3.x（你已有 3.13.2）
- Reaper（你已安装）
- 音源插件（可以用 Reaper 自带的 ReaSynth）

### Q2: 要多少硬盘空间？
- MIDI 文件：~100 MB（1000 个文件）
- WAV 文件：~2-5 GB（1000 个文件，取决于时长和质量）

### Q3: 渲染需要多久？
- 测试（100 个文件）：10-20 分钟
- 正常（1000 个文件）：1-2 小时
- 取决于音源复杂度和电脑性能

### Q4: 必须用 8 个轨道吗？
不必须，但建议：
- 最少 4 个（SATB）
- 推荐 8 个（覆盖多种配器）
- 可以都用同一个音源

### Q5: 可以只生成某些和弦吗？
可以！修改生成器代码，注释掉不需要的和弦类型。

### Q6: 支持 Windows 吗？
支持！但需要修改 Lua 脚本中的路径格式（使用反斜杠或双反斜杠）。

### Q7: 可以用 Logic Pro / Cubase 吗？
原理相同，但：
- 需要手动设置
- 没有现成的 Lua 脚本
- 建议用 Reaper（自动化更容易）

### Q8: 已有 MIDI 文件，能直接用吗？
可以，但需要确保：
- MIDI 文件有正确的轨道名称或通道分配
- 对应 Reaper 模板的轨道设置

---

## 🔗 外部资源

### Reaper 相关
- [Reaper 官方网站](https://www.reaper.fm/)
- [ReaScript API 文档](https://www.reaper.fm/sdk/reascript/reascript.php)
- [Reaper 中文论坛](https://www.reaperch.com/)

### Python MIDI 库
- [MIDIUtil 文档](https://midiutil.readthedocs.io/)
- [mido 库](https://mido.readthedocs.io/)（备选）

### 音乐理论
- 和弦进行理论
- 四部和声规则
- MIDI 协议标准

---

## 📝 版本历史

### v1.0 (2025-01-25)
- ✅ 多轨道单和弦生成器
- ✅ 和弦进行生成器
- ✅ Reaper 批量渲染脚本
- ✅ 完整文档体系

### 计划中
- [ ] 支持更多调性（12 调）
- [ ] 更多和弦类型（20+ 种）
- [ ] 节奏变化
- [ ] 转调支持
- [ ] GUI 界面

---

## 🤝 贡献

欢迎改进：
- 添加新的和弦类型
- 优化渲染脚本
- 补充文档
- 报告 bug

---

## 📄 许可

MIT License - 自由使用和修改

---

## 🎉 开始使用

准备好了吗？

### 方案 A: 单和弦训练数据（基础）

```bash
# 第一步：生成测试数据
python3 generate_single_chords_multitrack.py -r 3 -o ./test_midi

# 第二步：设置 Reaper
# 参考：REAPER_SETUP_DETAILED.md

# 第三步：测试渲染
# 手动拖入 Reaper 测试

# 第四步：批量渲染
# 修改 batch_render_simple.lua 路径
# 在 Reaper 中运行脚本

# 第五步：训练模型
# 使用生成的 WAV + labels.txt
```

### 方案 B: 和弦进行数据（真实音乐场景）✨ 新功能!

```bash
# 第一步：生成和弦进行 MIDI（100个文件，每个4个和弦，每个和弦2秒）
python generate_chord_progressions.py -n 100 -c 4 -d 2.0

# 第二步：使用特定进行模板
python generate_chord_progressions.py -n 50 -t I-V-vi-IV  # 流行四和弦

# 第三步：调整和弦时长
python generate_chord_progressions.py -n 50 -c 6 -d 3.0  # 每个和弦3秒

# 第四步：在 Reaper 中批量渲染
# 使用相同的 batch_render_simple.lua

# 第五步：用和弦分析器测试
python chord_analyzer_simple.py output.wav --window 2.0 --hop 2.0
```

**详细文档**: [PROGRESSION_USAGE.md](./PROGRESSION_USAGE.md) 📚

**祝你的 AutoChord 项目成功！** 🎵🚀

---

最后更新：2024年11月4日
```
