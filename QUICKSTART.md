# 快速开始 - 分析音乐的和弦进行

## 🚀 5分钟上手

### 1. 准备音频文件
- 支持格式：WAV, MP3, FLAC
- 推荐：干净的音频（伴奏或钢琴独奏效果最好）
- 复杂混音可能影响识别准确率

### 2. 运行分析（最简单）

```bash
# 使用自动化脚本（会自动查找模型）
bash analyze_music.sh your_song.wav
```

### 3. 查看结果
- 终端显示实时分析过程
- 自动生成 `your_song_chord_analysis.txt` 文件
- 包含完整和弦进行和统计信息

---

## 📝 示例：分析一首歌

假设你有一首歌 `test_song.wav`：

```bash
# 方法 1：使用脚本（推荐）
bash analyze_music.sh test_song.wav

# 方法 2：手动指定参数
python3 chord_analyzer_simple.py test_song.wav \
    --model models_full_stft/chord_model_full_xxx.pth \
    --hop 0.5 \
    --device mps
```

**输出预览：**
```
开始分析（每 0.5 秒滑动窗口，分析窗口 2.0 秒）...

    时间范围      |         和弦         | 置信度 | Top-3 预测
-------------------------------------------------------------------------------------
  0.00- 2.00s | ✓      C_major        |  85.2% | C_major(85%), C_maj7(8%), G_major(4%)
  0.50- 2.50s | ✓      C_major        |  87.3% | C_major(87%), C_maj7(7%), F_major(3%)
  1.00- 3.00s | ✓      F_major        |  78.9% | F_major(79%), F_maj7(12%), C_major(5%) ← 和弦变化！
  1.50- 3.50s | ✓      F_major        |  82.1% | F_major(82%), Bb_major(10%), F_maj7(4%)
  ...

和弦进行序列（按时间顺序）:
-------------------------------------------------------------------------------------
  0.00-  1.00s |     1.00s | C_major
  1.00-  4.50s |     3.50s | F_major
  4.50-  8.00s |     3.50s | G_major
  8.00- 12.00s |     4.00s | C_major
  ...
```

---

## 🎯 不同音乐类型的推荐设置

### 流行歌曲（Pop/Rock）
```bash
bash analyze_music.sh song.wav --hop 0.5
```
- hop=0.5 秒：捕捉正常节奏的和弦变化
- 适合大多数4/4拍的流行音乐

### 古典/爵士（复杂和声）
```bash
bash analyze_music.sh song.wav --hop 0.25
```
- hop=0.25 秒：更精细的分析
- 适合频繁转调的音乐

### 民谣/慢歌
```bash
bash analyze_music.sh song.wav --hop 1.0
```
- hop=1.0 秒：和弦变化较慢，1秒足够

### 快速测试
```bash
bash analyze_music.sh song.wav --hop 2.0
```
- hop=2.0 秒：快速扫描整首歌的大致和弦

---

## 📊 理解输出结果

### 1. 实时分析显示

```
  1.00- 3.00s | ✓      F_major        |  78.9% | F_major(79%), F_maj7(12%), C_major(5%)
  ^^^^^^^^^^^^   ^      ^^^^^^^          ^^^^^    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  时间范围       符号    识别和弦        置信度    Top-3候选和弦及概率
```

**符号含义：**
- `✓` 高置信度（>70%）：可靠
- `~` 中等置信度（40-70%）：可能有歧义
- `?` 低置信度（<40%）：不确定

### 2. 和弦进行序列

显示合并后的和弦段落，更易读：
```
  0.00-  4.50s |     4.50s | C_major    ← 开头4.5秒都是C大调
  4.50-  8.00s |     3.50s | F_major    ← 然后转到F大调
  8.00- 12.00s |     4.00s | G_major    ← 再转到G大调
```

### 3. 和弦统计

```
     C_major       |          4 |    12.50s |   27.6%
     ^^^^^^^^^^^      ^^^^^^^^      ^^^^^^      ^^^^^
     和弦名称         出现次数      总时长      占比
```

帮助你了解：
- 歌曲的主和弦（占比最高）
- 和弦进行的复杂度（和弦种类数量）
- 各和弦的重要性

---

## 🔧 常见问题

### Q: 识别结果不准确？
**A:** 可能的原因和解决方案：
1. **音频质量差**：使用干净的伴奏或钢琴版
2. **多乐器混音**：尝试提取主旋律或使用伴奏轨
3. **模型未训练充分**：检查模型准确率是否 >75%
4. **和弦太复杂**：模型只训练了11种基础和弦类型

### Q: 为什么同一位置显示多次？
**A:** 这是滑动窗口的特点：
- 每个2秒窗口都会输出一次识别结果
- hop=0.5 意味着窗口有1.5秒重叠
- "和弦进行序列"部分会自动合并相同和弦

### Q: 如何加快分析速度？
**A:** 
- 增大 `--hop` 参数（如 1.0 或 2.0）
- 使用 `--device mps`（Mac）或 `cuda`（NVIDIA GPU）
- 使用较小的模型（chord 任务而非 full 任务）

### Q: 可以分析哪些音乐？
**A:** 
- ✅ 流行歌曲伴奏
- ✅ 钢琴独奏
- ✅ 吉他弹唱
- ✅ 简单编曲的音乐
- ⚠️ 复杂交响乐（效果可能不佳）
- ⚠️ 重金属/电子（和声复杂）

---

## 📈 下一步

1. **测试不同 hop 值**：找到适合你音乐风格的设置
2. **对比人工标注**：检查模型准确率
3. **训练更好的模型**：使用 CQT 特征或增加训练数据
4. **导出为 MIDI/MusicXML**：进一步开发导出功能

---

## 💡 实用技巧

### 批量处理多个文件
```bash
for file in *.wav; do
    echo "分析: $file"
    bash analyze_music.sh "$file" --hop 0.5
done
```

### 只分析前30秒（快速测试）
```bash
# 先用 ffmpeg 截取
ffmpeg -i full_song.wav -t 30 -c copy preview.wav
bash analyze_music.sh preview.wav
```

### 对比不同模型
```bash
# Full 任务（77类）
python3 chord_analyzer_simple.py song.wav \
    --model models_full_stft/chord_model_full_xxx.pth \
    --hop 0.5

# Chord 任务（11类）
python3 chord_analyzer_simple.py song.wav \
    --model models_chord_stft/chord_model_chord_xxx.pth \
    --hop 0.5
```

---

需要更多帮助？查看 `PLAYER_USAGE.md` 完整文档！
