#!/usr/bin/env python3
"""
实时和弦识别系统
支持音频文件播放和实时分析,带调性检测
"""

import sys
import time
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from collections import deque
import argparse

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
            
            nn.AdaptiveAvgPool2d((1, 1))
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


# ==================== KS调性检测算法 ====================

class RealtimeKeyDetector:
    """实时调性检测器 (Krumhansl-Schmuckler算法)"""
    
    # Krumhansl-Kessler音高轮廓 (基于认知心理学实验)
    MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # 和弦音符映射 (只列出11种和弦类型)
    CHORD_NOTES = {
        'major': [0, 4, 7],           # 大三和弦
        'minor': [0, 3, 7],           # 小三和弦
        'dim': [0, 3, 6],             # 减三和弦
        'aug': [0, 4, 8],             # 增三和弦
        'sus2': [0, 2, 7],            # 挂二和弦
        'sus4': [0, 5, 7],            # 挂四和弦
        'maj7': [0, 4, 7, 11],        # 大七和弦
        'min7': [0, 3, 7, 10],        # 小七和弦
        'dom7': [0, 4, 7, 10],        # 属七和弦
        'hdim7': [0, 3, 6, 10],       # 半减七和弦
        'dim7': [0, 3, 6, 9],         # 减七和弦
    }
    
    def __init__(self, window_size: int = 16, min_chords: int = 4, 
                 confidence_threshold: float = 0.65):
        """
        Args:
            window_size: 滑动窗口大小(保留最近N个和弦)
            min_chords: 开始判断所需的最少和弦数
            confidence_threshold: 输出调性的最低置信度
        """
        self.window_size = window_size
        self.min_chords = min_chords
        self.confidence_threshold = confidence_threshold
        
        self.chord_buffer = deque(maxlen=window_size)
        self.current_key = None
        self.confidence = 0.0
        self.key_history = []  # 记录调性变化历史
    
    def add_chord(self, chord_name: str) -> Dict:
        """
        添加新识别的和弦并更新调性判断
        
        Args:
            chord_name: 和弦名称,格式如 "C_major", "A_minor"等
        
        Returns:
            调性信息字典: {
                'key': 当前调性 (如"C_major"),
                'confidence': 置信度 (0-1),
                'status': 状态 ('confirmed'/'analyzing'/'insufficient_data')
            }
        """
        self.chord_buffer.append(chord_name)
        
        if len(self.chord_buffer) >= self.min_chords:
            self._update_key()
        
        return self.get_current_key()
    
    def _update_key(self):
        """基于当前缓冲区更新调性判断"""
        # 1. 计算音高类权重
        pitch_weights = self._calculate_pitch_weights()
        
        # 2. 对所有24个调性计算相关性
        key_scores = {}
        
        for root_pc in range(12):
            # 旋转大调模板
            major_profile = self._rotate_profile(self.MAJOR_PROFILE, root_pc)
            corr_major = self._pearson_correlation(pitch_weights, major_profile)
            key_name = f"{self.NOTE_NAMES[root_pc]}_major"
            key_scores[key_name] = corr_major
            
            # 旋转小调模板
            minor_profile = self._rotate_profile(self.MINOR_PROFILE, root_pc)
            corr_minor = self._pearson_correlation(pitch_weights, minor_profile)
            key_name = f"{self.NOTE_NAMES[root_pc]}_minor"
            key_scores[key_name] = corr_minor
        
        # 3. 找出最佳调性
        sorted_keys = sorted(key_scores.items(), key=lambda x: x[1], reverse=True)
        best_key, best_score = sorted_keys[0]
        second_score = sorted_keys[1][1] if len(sorted_keys) > 1 else 0
        
        # 4. 计算置信度 (使用绝对相关性值)
        # Pearson相关系数范围是[-1, 1],但KS算法通常给出正值
        # 当相关性 > 0.7 时认为高置信度
        confidence = max(0.0, best_score)  # 直接使用相关性作为置信度
        
        # 5. 更新当前调性 (使用滞后机制避免频繁跳变)
        if self.current_key is None:
            # 首次判断,置信度达到阈值即可
            if confidence >= self.confidence_threshold:
                self.current_key = best_key
                self.confidence = confidence
                self.key_history.append((best_key, confidence))
        else:
            # 已有调性,切换需要更高置信度
            if best_key != self.current_key:
                if confidence >= 0.75:  # 切换阈值更高
                    self.current_key = best_key
                    self.confidence = confidence
                    self.key_history.append((best_key, confidence))
            else:
                self.confidence = confidence
    
    def _calculate_pitch_weights(self) -> List[float]:
        """计算当前缓冲区中每个音高类的权重"""
        weights = [0.0] * 12
        
        for i, chord_name in enumerate(self.chord_buffer):
            # 解析和弦
            root_str, chord_type = self._parse_chord(chord_name)
            root_pc = self.NOTE_NAMES.index(root_str)
            
            # 获取和弦音符
            if chord_type in self.CHORD_NOTES:
                intervals = self.CHORD_NOTES[chord_type]
            else:
                # 未知和弦类型,只用根音
                intervals = [0]
            
            # 时间衰减权重 (越新的和弦权重越高)
            recency_weight = 0.5 + 0.5 * (i / len(self.chord_buffer))
            
            for interval in intervals:
                pc = (root_pc + interval) % 12
                if interval == 0:  # 根音权重加倍
                    weights[pc] += 2.0 * recency_weight
                else:
                    weights[pc] += 1.0 * recency_weight
        
        return weights
    
    @staticmethod
    def _parse_chord(chord_name: str) -> Tuple[str, str]:
        """解析和弦名称"""
        parts = chord_name.split('_')
        if len(parts) == 2:
            return parts[0], parts[1]
        else:
            return parts[0], 'major'  # 默认大三和弦
    
    @staticmethod
    def _rotate_profile(profile: List[float], steps: int) -> List[float]:
        """旋转音高轮廓"""
        return profile[steps:] + profile[:steps]
    
    @staticmethod
    def _pearson_correlation(x: List[float], y: List[float]) -> float:
        """计算皮尔逊相关系数"""
        n = len(x)
        if n == 0:
            return 0.0
        
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        
        sum_sq_x = sum((x[i] - mean_x) ** 2 for i in range(n))
        sum_sq_y = sum((y[i] - mean_y) ** 2 for i in range(n))
        
        denominator = (sum_sq_x * sum_sq_y) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def get_current_key(self) -> Dict:
        """获取当前调性信息"""
        if len(self.chord_buffer) < self.min_chords:
            return {
                'key': None,
                'confidence': 0.0,
                'status': 'insufficient_data',
                'buffer_size': len(self.chord_buffer)
            }
        
        if self.confidence >= self.confidence_threshold:
            return {
                'key': self.current_key,
                'confidence': self.confidence,
                'status': 'confirmed',
                'buffer_size': len(self.chord_buffer)
            }
        else:
            return {
                'key': self.current_key,
                'confidence': self.confidence,
                'status': 'analyzing',
                'buffer_size': len(self.chord_buffer)
            }
    
    def reset(self):
        """重置检测器"""
        self.chord_buffer.clear()
        self.current_key = None
        self.confidence = 0.0
        self.key_history.clear()


# ==================== 实时和弦分析器 ====================

class RealtimeChordAnalyzer:
    """实时和弦分析器 (使用SimpleChordRecognizer保证一致性)"""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Args:
            model_path: 模型文件路径
            device: 计算设备 ('cpu' 或 'cuda')
        """
        # 导入SimpleChordRecognizer
        from chord_analyzer_simple import SimpleChordRecognizer
        
        # 查找标签映射文件
        model_dir = Path(model_path).parent
        mapping_files = list(model_dir.glob("label_mappings_*.json"))
        if not mapping_files:
            raise FileNotFoundError(f"未找到标签映射文件在 {model_dir}")
        
        mapping_path = str(mapping_files[0])
        
        # 使用SimpleChordRecognizer确保预处理一致性
        print(f"正在加载模型: {model_path}")
        self.recognizer = SimpleChordRecognizer(model_path, mapping_path, device=device)
        
        # 参数
        self.window_duration = self.recognizer.window_duration
        self.sample_rate = self.recognizer.target_sr
        
        # 调性检测器
        self.key_detector = RealtimeKeyDetector()

    
    def analyze_audio_file(self, audio_path: str, display_realtime: bool = True):
        """
        分析音频文件,模拟实时播放效果
        
        Args:
            audio_path: 音频文件路径
            display_realtime: 是否实时显示(带延迟模拟)
        """
        print(f"\n{'='*70}")
        print(f"正在分析: {Path(audio_path).name}")
        print(f"{'='*70}\n")
        
        # 加载音频
        waveform, sr = torchaudio.load(audio_path)
        
        # 转单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 重采样
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # 计算窗口参数
        window_samples = int(self.window_duration * self.sample_rate)
        hop_samples = window_samples // 2  # 50% 重叠
        total_duration = waveform.shape[1] / self.sample_rate
        
        print(f"音频时长: {total_duration:.1f}秒")
        print(f"分析窗口: {self.window_duration}秒, 步长: {self.window_duration/2}秒")
        print(f"\n{'时间':<10} {'和弦':<20} {'置信度':<10} {'调性':<15} {'调性置信度':<12}")
        print(f"{'-'*70}")
        
        # 重置调性检测器
        self.key_detector.reset()
        
        # 滑动窗口分析
        results = []
        position = 0
        
        while position + window_samples <= waveform.shape[1]:
            # 提取窗口
            window = waveform[:, position:position + window_samples]
            
            # 预测和弦 (传入采样率)
            chord, confidence = self._predict_chord(window, self.sample_rate)
            
            # 更新调性检测
            key_info = self.key_detector.add_chord(chord)
            
            # 当前时间
            current_time = position / self.sample_rate
            
            # 格式化输出
            time_str = f"{current_time:>6.1f}s"
            chord_str = f"{chord:<20}"
            conf_str = f"{confidence:>5.1f}%"
            
            if key_info['status'] == 'confirmed':
                key_str = f"✓ {key_info['key']:<15}"
                key_conf_str = f"{key_info['confidence']*100:>5.1f}%"
            elif key_info['status'] == 'analyzing':
                key_str = f"? {key_info['key'] or '分析中...':<15}"
                key_conf_str = f"{key_info['confidence']*100:>5.1f}%"
            else:
                key_str = f"  {'[数据不足]':<15}"
                key_conf_str = "-"
            
            print(f"{time_str:<10} {chord_str} {conf_str:<10} {key_str} {key_conf_str:<12}")
            
            results.append({
                'time': current_time,
                'chord': chord,
                'confidence': confidence,
                'key': key_info['key'],
                'key_confidence': key_info['confidence'],
                'key_status': key_info['status']
            })
            
            # 模拟实时延迟
            if display_realtime:
                time.sleep(self.window_duration / 4)  # 加速4倍播放
            
            position += hop_samples
        
        # 输出最终调性
        final_key = self.key_detector.get_current_key()
        print(f"\n{'='*70}")
        if final_key['status'] == 'confirmed':
            print(f"✓ 最终调性: {final_key['key']} (置信度: {final_key['confidence']*100:.1f}%)")
        else:
            print(f"⚠ 调性不确定: {final_key['key'] or '无法判断'} (置信度: {final_key['confidence']*100:.1f}%)")
        
        # 显示调性变化历史
        if len(self.key_detector.key_history) > 1:
            print(f"\n调性变化历史:")
            for i, (key, conf) in enumerate(self.key_detector.key_history):
                print(f"  {i+1}. {key} (置信度: {conf*100:.1f}%)")
        
        print(f"{'='*70}\n")
        
        return results
    
    def _predict_chord(self, waveform: torch.Tensor, sr: int) -> Tuple[str, float]:
        """预测和弦 (直接调用SimpleChordRecognizer保证一致性)"""
        chord, confidence, _ = self.recognizer.predict(waveform, sr)
        return chord, confidence * 100


# ==================== 命令行接口 ====================

def main():
    parser = argparse.ArgumentParser(
        description='实时和弦识别系统 (带调性检测)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分析单个文件
  python realtime_chord_analyzer.py audio.wav
  
  # 指定模型路径
  python realtime_chord_analyzer.py audio.wav --model models_full_stft/chord_model_full_20251103_130224.pth
  
  # 快速模式(不模拟实时延迟)
  python realtime_chord_analyzer.py audio.wav --no-realtime
  
  # 使用GPU
  python realtime_chord_analyzer.py audio.wav --device cuda
        """
    )
    
    parser.add_argument('audio_file', type=str, help='音频文件路径')
    parser.add_argument('--model', type=str, 
                        default='models_full_stft/chord_model_full_20251103_130224.pth',
                        help='模型文件路径')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='计算设备')
    parser.add_argument('--no-realtime', action='store_true',
                        help='禁用实时延迟模拟(快速模式)')
    
    args = parser.parse_args()
    
    # 检查文件
    if not Path(args.audio_file).exists():
        print(f"错误: 音频文件不存在: {args.audio_file}")
        sys.exit(1)
    
    if not Path(args.model).exists():
        print(f"错误: 模型文件不存在: {args.model}")
        sys.exit(1)
    
    # 创建分析器
    analyzer = RealtimeChordAnalyzer(args.model, args.device)
    
    # 分析音频
    try:
        analyzer.analyze_audio_file(
            args.audio_file,
            display_realtime=not args.no_realtime
        )
    except KeyboardInterrupt:
        print("\n\n用户中断分析")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
