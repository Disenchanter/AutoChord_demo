#!/usr/bin/env python3
"""
和弦识别播放器 - 简化命令行版本
实时分析 WAV 文件并显示和弦预测

依赖最小化：只需 torch, torchaudio, numpy
"""

import sys
import time
from pathlib import Path
from typing import Tuple, List
import json

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T


# ==================== 模型定义 ====================

class ChordCNN(nn.Module):
    """和弦识别 CNN 模型"""
    
    def __init__(self, num_classes: int, input_channels: int = 1):
        super(ChordCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
            
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# ==================== 和弦识别器 ====================

class SimpleChordRecognizer:
    """简化和弦识别器"""
    
    def __init__(self, model_path: str, label_mapping_path: str, device: str = 'cpu'):
        self.device = device
        self.target_sr = 22050
        self.n_fft = 2048
        self.hop_length = 512
        self.window_duration = 2.0
        self.window_samples = int(self.target_sr * self.window_duration)
        
        # 加载标签
        with open(label_mapping_path, 'r') as f:
            mapping_data = json.load(f)
        
        self.task = mapping_data['task']
        if self.task == 'full':
            self.idx_to_label = {
                int(v): k for k, v in mapping_data['mappings']['full_label_to_idx'].items()
            }
        elif self.task == 'root':
            self.idx_to_label = {
                int(v): k for k, v in mapping_data['mappings']['root_to_idx'].items()
            }
        elif self.task == 'chord':
            self.idx_to_label = {
                int(v): k for k, v in mapping_data['mappings']['chord_to_idx'].items()
            }
        
        # 加载模型
        num_classes = len(self.idx_to_label)
        self.model = ChordCNN(num_classes=num_classes)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # STFT 变换
        self.spectrogram_transform = T.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            power=2.0,
        )
        self.amplitude_to_db = T.AmplitudeToDB()
        
        print(f"✓ 模型加载: {num_classes} 类, 任务: {self.task}")
    
    def predict(self, waveform: torch.Tensor, sr: int) -> Tuple[str, float, List[Tuple[str, float]]]:
        """预测和弦"""
        # 重采样
        if sr != self.target_sr:
            resampler = T.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
        
        # 裁剪/填充
        if waveform.shape[1] < self.window_samples:
            padding = self.window_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            waveform = waveform[:, :self.window_samples]
        
        # STFT
        spec = self.spectrogram_transform(waveform)
        spec_db = self.amplitude_to_db(spec)
        spec_db = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min() + 1e-8)
        
        # 预测
        with torch.no_grad():
            spec_db = spec_db.unsqueeze(0).to(self.device)
            outputs = self.model(spec_db)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        
        # Top-5
        top_5_idx = np.argsort(probs)[-5:][::-1]
        top_5 = [(self.idx_to_label[idx], float(probs[idx])) for idx in top_5_idx]
        
        return self.idx_to_label[top_5_idx[0]], float(probs[top_5_idx[0]]), top_5


# ==================== 主函数 ====================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='和弦识别播放器 - 简化版')
    parser.add_argument('wav_file', type=str, help='WAV 文件路径')
    parser.add_argument('--model', type=str, required=True, help='模型文件路径 (.pth)')
    parser.add_argument('--mapping', type=str, help='标签映射文件路径 (.json)')
    parser.add_argument('--hop', type=float, default=1.0, help='分析间隔（秒）')
    parser.add_argument('--device', type=str, default='cpu', help='设备 (cpu/cuda/mps)')
    
    args = parser.parse_args()
    
    # 检查文件
    if not Path(args.wav_file).exists():
        print(f"错误: 文件不存在 - {args.wav_file}")
        return
    
    if not Path(args.model).exists():
        print(f"错误: 模型不存在 - {args.model}")
        return
    
    # 自动查找标签映射
    if args.mapping is None:
        model_dir = Path(args.model).parent
        mapping_files = list(model_dir.glob("label_mappings_*.json"))
        if not mapping_files:
            print("错误: 未找到标签映射文件")
            return
        args.mapping = str(mapping_files[0])
    
    print("\n" + "="*70)
    print("和弦识别播放器 - 简化命令行版")
    print("="*70 + "\n")
    
    # 加载音频
    print(f"加载音频: {args.wav_file}")
    waveform, sr = torchaudio.load(args.wav_file)
    
    # 转为单声道
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    duration = waveform.shape[1] / sr
    print(f"  时长: {duration:.2f} 秒")
    print(f"  采样率: {sr} Hz")
    
    # 加载识别器
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    if device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'
    
    recognizer = SimpleChordRecognizer(args.model, args.mapping, device=device)
    
    # 逐段分析（滑动窗口）
    print(f"\n开始分析（每 {args.hop} 秒滑动窗口，分析窗口 {recognizer.window_duration} 秒）...\n")
    print(f"{'时间范围':^18s} | {'和弦':^20s} | {'置信度':>8s} | {'Top-3 预测':^45s}")
    print("-" * 95)
    
    hop_samples = int(sr * args.hop)
    window_samples_sr = int(sr * recognizer.window_duration)
    
    position = 0
    results = []
    last_chord = None
    chord_start_time = 0
    chord_segments = []  # [(start_time, end_time, chord, avg_confidence), ...]
    
    while position + window_samples_sr <= waveform.shape[1]:
        # 提取片段
        segment = waveform[:, position:position + window_samples_sr]
        
        # 识别
        chord, confidence, top_5 = recognizer.predict(segment, sr)
        
        # 显示
        current_time = position / sr
        end_time = (position + window_samples_sr) / sr
        top_3_str = " | ".join([f"{c}({p*100:.1f}%)" for c, p in top_5[:3]])
        
        # 根据置信度选择符号
        if confidence > 0.7:
            symbol = "✓"
        elif confidence > 0.4:
            symbol = "~"
        else:
            symbol = "?"
        
        # 检测和弦变化
        change_marker = ""
        if last_chord is not None and chord != last_chord:
            change_marker = " ← 和弦变化！"
            # 保存上一个和弦段
            chord_segments.append((chord_start_time, current_time, last_chord, []))
            chord_start_time = current_time
        
        print(f"{current_time:6.2f}-{end_time:5.2f}s | {symbol} {chord:^18s} | {confidence*100:6.1f}% | {top_3_str}{change_marker}")
        
        # 保存详细结果（包括top-3）
        results.append((current_time, chord, confidence, top_5[:3]))
        
        last_chord = chord
        
        # 移动窗口
        position += hop_samples
    
    # 保存最后一个和弦段
    if last_chord is not None:
        final_time = waveform.shape[1] / sr
        chord_segments.append((chord_start_time, final_time, last_chord, []))
    
    # 统计和总结
    print("\n" + "="*85)
    print("分析完成！")
    print("="*85)
    print(f"总时长: {duration:.2f} 秒")
    print(f"分析窗口数: {len(results)}")
    print(f"检测到的和弦段数: {len(chord_segments)}")
    
    if results:
        avg_conf = np.mean([r[2] for r in results])
        print(f"平均置信度: {avg_conf*100:.1f}%")
        
        # 和弦进行总结
        print(f"\n{'='*85}")
        print("和弦进行序列（按时间顺序）:")
        print(f"{'='*85}")
        print(f"{'时间范围':^18s} | {'持续时长':>10s} | {'和弦':^20s}")
        print("-" * 85)
        
        for start, end, chord, _ in chord_segments:
            duration_seg = end - start
            print(f"{start:6.2f}-{end:6.2f}s | {duration_seg:8.2f}s | {chord:^20s}")
        
        # 和弦统计
        chord_counts = {}
        chord_durations = {}
        for start, end, chord, _ in chord_segments:
            chord_counts[chord] = chord_counts.get(chord, 0) + 1
            chord_durations[chord] = chord_durations.get(chord, 0) + (end - start)
        
        print(f"\n{'='*85}")
        print("和弦统计:")
        print(f"{'='*85}")
        print(f"{'和弦':^20s} | {'出现次数':>10s} | {'总时长':>10s} | {'占比':>8s}")
        print("-" * 85)
        
        for chord in sorted(chord_counts.keys(), key=lambda x: -chord_durations[x])[:15]:
            count = chord_counts[chord]
            total_dur = chord_durations[chord]
            percentage = (total_dur / duration) * 100
            print(f"{chord:^20s} | {count:10d} | {total_dur:9.2f}s | {percentage:7.1f}%")
        
        # 保存结果到文件
        output_dir = Path("analysis_test")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / (Path(args.wav_file).stem + "_chord_analysis.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*85 + "\n")
            f.write(f"和弦识别分析报告 - {Path(args.wav_file).name}\n")
            f.write("="*85 + "\n\n")
            f.write(f"总时长: {duration:.2f} 秒\n")
            f.write(f"分析窗口数: {len(results)}\n")
            f.write(f"检测到的和弦段数: {len(chord_segments)}\n")
            f.write(f"平均置信度: {avg_conf*100:.1f}%\n\n")
            
            # 写入详细的逐窗口分析（包括Top-3）
            f.write("="*95 + "\n")
            f.write("详细分析（每个窗口的Top-3预测）:\n")
            f.write("="*95 + "\n")
            f.write(f"{'时间范围':^18s} | {'第1名':^25s} | {'第2名':^25s} | {'第3名':^25s}\n")
            f.write("-" * 95 + "\n")
            for time_pos, chord, conf, top_3 in results:
                end_t = time_pos + recognizer.window_duration
                top_3_formatted = [f"{c}({p*100:.1f}%)" for c, p in top_3]
                if len(top_3_formatted) < 3:
                    top_3_formatted.extend(['---'] * (3 - len(top_3_formatted)))
                f.write(f"{time_pos:6.2f}-{end_t:5.2f}s | {top_3_formatted[0]:^25s} | {top_3_formatted[1]:^25s} | {top_3_formatted[2]:^25s}\n")
            
            f.write("\n" + "="*85 + "\n")
            f.write("和弦进行序列（按时间顺序）:\n")
            f.write("="*85 + "\n")
            f.write(f"{'时间范围':^18s} | {'持续时长':>10s} | {'和弦':^20s}\n")
            f.write("-" * 85 + "\n")
            for start, end, chord, _ in chord_segments:
                duration_seg = end - start
                f.write(f"{start:6.2f}-{end:6.2f}s | {duration_seg:8.2f}s | {chord:^20s}\n")
            
            f.write("\n" + "="*85 + "\n")
            f.write("和弦统计:\n")
            f.write("="*85 + "\n")
            f.write(f"{'和弦':^20s} | {'出现次数':>10s} | {'总时长':>10s} | {'占比':>8s}\n")
            f.write("-" * 85 + "\n")
            for chord in sorted(chord_counts.keys(), key=lambda x: -chord_durations[x])[:15]:
                count = chord_counts[chord]
                total_dur = chord_durations[chord]
                percentage = (total_dur / duration) * 100
                f.write(f"{chord:^20s} | {count:10d} | {total_dur:9.2f}s | {percentage:7.1f}%\n")
        
        print(f"\n✓ 分析结果已保存到: {output_file}")


if __name__ == '__main__':
    main()
