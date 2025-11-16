#!/usr/bin/env python3
"""
和弦进行分析器 - CQT 版本
支持使用 CQT 模型分析和弦进行
"""

import sys
import argparse
from pathlib import Path
from typing import List, Tuple
import json

import numpy as np
import torch
import torch.nn as nn
import librosa

from chord_analyzer_stft import ChordCNN


class CQTChordAnalyzer:
    """基于 CQT 的和弦分析器"""
    
    def __init__(
        self, 
        model_path: str, 
        label_mapping_path: str, 
        device: str = 'cpu',
        n_bins: int = 84,
        bins_per_octave: int = 12
    ):
        self.device = device
        self.target_sr = 22050
        self.hop_length = 512
        self.window_duration = 2.0
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        
        # 加载标签映射
        with open(label_mapping_path, 'r') as f:
            mapping_data = json.load(f)
        
        self.task = mapping_data['task']
        self.num_classes = mapping_data['num_classes']
        
        # 获取 CQT 参数
        self.n_bins = mapping_data.get('n_bins', n_bins)
        self.bins_per_octave = mapping_data.get('bins_per_octave', bins_per_octave)
        
        # 构建标签映射
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
        self.model = ChordCNN(num_classes=self.num_classes)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        print(f"✓ CQT 模型已加载")
        print(f"  任务类型: {self.task}")
        print(f"  类别数: {self.num_classes}")
        print(f"  CQT bins: {self.n_bins}")
        print(f"  Bins/octave: {self.bins_per_octave}")
        print(f"  设备: {device}\n")
    
    def extract_cqt_features(self, audio_segment: np.ndarray) -> torch.Tensor:
        """提取 CQT 特征"""
        # 裁剪或填充到目标长度
        target_length = int(self.target_sr * self.window_duration)
        if len(audio_segment) < target_length:
            audio_segment = np.pad(audio_segment, (0, target_length - len(audio_segment)))
        else:
            audio_segment = audio_segment[:target_length]
        
        # 提取 CQT
        C = librosa.cqt(
            audio_segment,
            sr=self.target_sr,
            hop_length=self.hop_length,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave
        )
        
        # 转换为 dB
        C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)
        
        # 归一化
        C_norm = (C_db - C_db.min()) / (C_db.max() - C_db.min() + 1e-8)
        
        # 转换为 tensor
        cqt_tensor = torch.FloatTensor(C_norm).unsqueeze(0).unsqueeze(0)
        
        return cqt_tensor
    
    def predict_chord(self, audio_segment: np.ndarray) -> Tuple[str, float]:
        """预测单个音频片段的和弦"""
        # 提取特征
        features = self.extract_cqt_features(audio_segment)
        features = features.to(self.device)
        
        # 预测
        with torch.no_grad():
            output = self.model(features)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_idx = probabilities.max(1)
        
        predicted_label = self.idx_to_label[predicted_idx.item()]
        return predicted_label, confidence.item()
    
    def analyze_progression(
        self, 
        wav_path: str, 
        overlap: float = 0.5
    ) -> List[Tuple[float, float, str, float]]:
        """分析整个和弦进行
        
        返回: List[(start_time, end_time, chord_label, confidence)]
        """
        # 加载音频
        y, sr = librosa.load(wav_path, sr=self.target_sr, mono=True)
        
        # 计算滑动窗口参数
        window_samples = int(self.target_sr * self.window_duration)
        hop_samples = int(window_samples * (1 - overlap))
        
        results = []
        
        # 滑动窗口分析
        for start_sample in range(0, len(y) - window_samples + 1, hop_samples):
            end_sample = start_sample + window_samples
            audio_segment = y[start_sample:end_sample]
            
            # 预测
            chord_label, confidence = self.predict_chord(audio_segment)
            
            # 计算时间
            start_time = start_sample / self.target_sr
            end_time = end_sample / self.target_sr
            
            results.append((start_time, end_time, chord_label, confidence))
        
        return results
    
    def smooth_predictions(
        self, 
        results: List[Tuple[float, float, str, float]], 
        min_duration: float = 0.5
    ) -> List[Tuple[float, float, str, float]]:
        """平滑预测结果，合并相同的连续和弦"""
        if not results:
            return []
        
        smoothed = []
        current_chord = results[0][2]
        current_start = results[0][0]
        current_end = results[0][1]
        current_confidences = [results[0][3]]
        
        for start, end, chord, conf in results[1:]:
            if chord == current_chord:
                # 继续累积相同和弦，更新结束时间
                current_end = end
                current_confidences.append(conf)
            else:
                # 保存前一个和弦
                avg_confidence = np.mean(current_confidences)
                duration = current_end - current_start
                
                if duration >= min_duration:
                    smoothed.append((
                        current_start,
                        current_end,
                        current_chord,
                        avg_confidence
                    ))
                
                # 开始新和弦
                current_chord = chord
                current_start = start
                current_end = end
                current_confidences = [conf]
        
        # 添加最后一个和弦
        if current_confidences:
            avg_confidence = np.mean(current_confidences)
            duration = current_end - current_start
            if duration >= min_duration:
                smoothed.append((
                    current_start,
                    current_end,
                    current_chord,
                    avg_confidence
                ))
        
        return smoothed


def format_time(seconds: float) -> str:
    """格式化时间显示"""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:05.2f}"


def print_analysis_results(
    results: List[Tuple[float, float, str, float]], 
    output_file: str = None
):
    """打印分析结果"""
    print("\n" + "="*80)
    print("和弦进行分析结果 (CQT)")
    print("="*80)
    
    output_lines = []
    output_lines.append("="*80)
    output_lines.append("和弦进行分析结果 (CQT)")
    output_lines.append("="*80)
    output_lines.append("")
    output_lines.append(f"{'时间范围':<20} {'和弦':<15} {'置信度':<10} {'时长':<10}")
    output_lines.append("-"*80)
    
    print(f"\n{'时间范围':<20} {'和弦':<15} {'置信度':<10} {'时长':<10}")
    print("-"*80)
    
    for start, end, chord, conf in results:
        duration = end - start
        time_range = f"{format_time(start)} - {format_time(end)}"
        conf_str = f"{conf*100:.1f}%"
        duration_str = f"{duration:.1f}s"
        
        line = f"{time_range:<20} {chord:<15} {conf_str:<10} {duration_str:<10}"
        print(line)
        output_lines.append(line)
    
    print("="*80)
    output_lines.append("="*80)
    
    # 统计信息
    total_duration = results[-1][1] if results else 0
    unique_chords = len(set(r[2] for r in results))
    avg_confidence = np.mean([r[3] for r in results]) if results else 0
    
    stats = f"\n统计信息:"
    stats += f"\n  总时长: {total_duration:.1f}s"
    stats += f"\n  和弦数量: {len(results)}"
    stats += f"\n  不同和弦: {unique_chords}"
    stats += f"\n  平均置信度: {avg_confidence*100:.1f}%"
    
    print(stats)
    output_lines.append("")
    output_lines.extend(stats.split('\n'))
    
    # 保存到文件
    if output_file:
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        print(f"\n✓ 结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='和弦进行分析器 - CQT 版本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 分析和弦进行
  python chord_analyzer_cqt.py \\
      long_progressions_piano_midi_output/prog_0001.wav \\
      --model models_full_cqt/chord_model_full_20251105_211615.pth \\
      --mappings models_full_cqt/label_mappings_full_20251105_211615.json
  
  # 保存结果到文件
  python chord_analyzer_cqt.py \\
      progressions_chords_output/prog_0001_satb_C_major-F_major-G_major-C_major.wav \\
      --model models_full_cqt/chord_model_full_20251105_211615.pth \\
      --mappings models_full_cqt/label_mappings_full_20251105_211615.json \\
      --output analysis_cqt/prog_0001_analysis.txt
        """
    )
    
    parser.add_argument('wav_file', type=str,
                        help='WAV 文件路径')
    parser.add_argument('--model', type=str, required=True,
                        help='CQT 模型文件路径 (.pth)')
    parser.add_argument('--mappings', type=str, required=True,
                        help='标签映射文件路径 (.json)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='计算设备 (cpu/cuda/mps)')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='滑动窗口重叠比例 (0-1, 默认 0.5)')
    parser.add_argument('--min_duration', type=float, default=0.5,
                        help='最小和弦持续时间 (秒, 默认 0.5)')
    parser.add_argument('--output', type=str, default=None,
                        help='结果输出文件路径')
    parser.add_argument('--no_smooth', action='store_true',
                        help='不进行平滑处理')
    
    args = parser.parse_args()
    
    # 检测设备
    if args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    # 检查文件是否存在
    wav_path = Path(args.wav_file)
    if not wav_path.exists():
        print(f"错误: 文件不存在: {wav_path}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"CQT 和弦进行分析器")
    print(f"{'='*80}")
    print(f"音频文件: {wav_path.name}")
    print(f"模型: {Path(args.model).name}")
    print(f"{'='*80}\n")
    
    # 创建分析器
    analyzer = CQTChordAnalyzer(
        model_path=args.model,
        label_mapping_path=args.mappings,
        device=device
    )
    
    # 分析和弦进行
    print("正在分析和弦进行...")
    results = analyzer.analyze_progression(
        str(wav_path),
        overlap=args.overlap
    )
    
    # 平滑处理
    if not args.no_smooth:
        print(f"正在平滑结果 (最小时长: {args.min_duration}s)...")
        results = analyzer.smooth_predictions(results, min_duration=args.min_duration)
    
    # 打印结果
    print_analysis_results(results, output_file=args.output)


if __name__ == '__main__':
    main()
