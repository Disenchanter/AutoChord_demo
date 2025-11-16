#!/usr/bin/env python3
"""
使用鲁棒调性检测器重新分析WAV文件
对比原始方法和改进方法的效果
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict
from collections import deque
import sys

import librosa
import torch
import numpy as np

# 导入分析器和鲁棒检测器
from key_detector_robust import RobustKeyDetector
from chord_analyzer_cqt import CQTChordAnalyzer


# ==================== 原始调性检测器 ====================

class OriginalKeyDetector:
    """原始调性检测器 (基于 Krumhansl-Schmuckler 算法)"""
    
    # Krumhansl-Kessler 音高轮廓
    MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # 和弦音符映射
    CHORD_NOTES = {
        'major': [0, 4, 7],
        'minor': [0, 3, 7],
        'dim': [0, 3, 6],
        'aug': [0, 4, 8],
        'sus2': [0, 2, 7],
        'sus4': [0, 5, 7],
        'maj7': [0, 4, 7, 11],
        'min7': [0, 3, 7, 10],
        'dom7': [0, 4, 7, 10],
        'hdim7': [0, 3, 6, 10],
        'dim7': [0, 3, 6, 9],
    }
    
    def __init__(
        self, 
        window_size: int = 12, 
        min_chords: int = 4,
        confidence_threshold: float = 0.60,
        use_confidence_weighting: bool = True
    ):
        self.window_size = window_size
        self.min_chords = min_chords
        self.confidence_threshold = confidence_threshold
        self.use_confidence_weighting = use_confidence_weighting
        
        self.chord_buffer = deque(maxlen=window_size)
        self.confidence_buffer = deque(maxlen=window_size)
        self.current_key = None
        self.confidence = 0.0
    
    def add_chord(self, chord_name: str, confidence: float = 1.0) -> Dict:
        """添加新识别的和弦并更新调性判断"""
        self.chord_buffer.append(chord_name)
        self.confidence_buffer.append(confidence)
        
        if len(self.chord_buffer) >= self.min_chords:
            self._update_key()
        
        return self.get_current_key()
    
    def _update_key(self):
        """基于当前缓冲区更新调性判断"""
        # 计算音高类权重
        pitch_weights = self._calculate_pitch_weights()
        
        # 对所有24个调性计算相关性
        key_scores = {}
        
        for root_pc in range(12):
            # 大调
            major_profile = self._rotate_profile(self.MAJOR_PROFILE, root_pc)
            corr_major = self._pearson_correlation(pitch_weights, major_profile)
            key_scores[f"{self.NOTE_NAMES[root_pc]}_major"] = max(0.0, corr_major)
            
            # 小调
            minor_profile = self._rotate_profile(self.MINOR_PROFILE, root_pc)
            corr_minor = self._pearson_correlation(pitch_weights, minor_profile)
            key_scores[f"{self.NOTE_NAMES[root_pc]}_minor"] = max(0.0, corr_minor)
        
        # 找出最佳调性
        best_key = max(key_scores, key=key_scores.get)
        best_score = key_scores[best_key]
        
        # 更新当前调性
        if self.current_key is None:
            if best_score >= self.confidence_threshold:
                self.current_key = best_key
                self.confidence = best_score
        else:
            if best_key != self.current_key:
                if best_score >= 0.70:  # 切换阈值更高
                    self.current_key = best_key
                    self.confidence = best_score
            else:
                self.confidence = best_score
    
    def _calculate_pitch_weights(self) -> List[float]:
        """计算当前缓冲区中每个音高类的权重"""
        weights = [0.0] * 12
        
        for i, chord_name in enumerate(self.chord_buffer):
            root_str, chord_type = self._parse_chord(chord_name)
            if root_str is None:
                continue
                
            try:
                root_pc = self.NOTE_NAMES.index(root_str)
            except ValueError:
                continue
            
            chord_notes = self.CHORD_NOTES.get(chord_type, [0])
            
            # 时间权重
            time_weight = (i + 1) / len(self.chord_buffer)
            
            # 置信度加权
            if self.use_confidence_weighting:
                conf_weight = self.confidence_buffer[i]
                weight = time_weight * conf_weight
            else:
                weight = time_weight
            
            # 累加音高类权重
            for note_offset in chord_notes:
                pc = (root_pc + note_offset) % 12
                weights[pc] += weight
        
        return weights
    
    def _parse_chord(self, chord_name: str) -> Tuple[str, str]:
        """解析和弦名称"""
        parts = chord_name.split()
        if not parts:
            return None, None
        
        root_and_type = parts[0]
        
        # 分离根音和类型
        for chord_type in ['hdim7', 'dim7', 'maj7', 'min7', 'dom7', 'aug', 'dim', 'sus2', 'sus4', 'major', 'minor']:
            if chord_type in root_and_type:
                root = root_and_type.replace(chord_type, '').strip()
                return root, chord_type
        
        return root_and_type, 'major'
    
    def _rotate_profile(self, profile: List[float], steps: int) -> List[float]:
        """旋转音高轮廓"""
        return profile[-steps:] + profile[:-steps] if steps else profile
    
    def _pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """计算Pearson相关系数"""
        n = len(x)
        if n == 0:
            return 0.0
        
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        sum_y2 = sum(yi * yi for yi in y)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def get_current_key(self) -> Dict:
        """获取当前调性"""
        return {
            'key': self.current_key,
            'confidence': self.confidence,
            'num_chords': len(self.chord_buffer)
        }
    
    def reset(self):
        """重置检测器"""
        self.chord_buffer.clear()
        self.confidence_buffer.clear()
        self.current_key = None
        self.confidence = 0.0
    
    def get_all_key_scores(self) -> Dict[str, float]:
        """获取所有调性的得分"""
        if len(self.chord_buffer) < self.min_chords:
            return {}
        
        pitch_weights = self._calculate_pitch_weights()
        key_scores = {}
        
        for root_pc in range(12):
            major_profile = self._rotate_profile(self.MAJOR_PROFILE, root_pc)
            corr_major = self._pearson_correlation(pitch_weights, major_profile)
            key_scores[f"{self.NOTE_NAMES[root_pc]}_major"] = max(0.0, corr_major)
            
            minor_profile = self._rotate_profile(self.MINOR_PROFILE, root_pc)
            corr_minor = self._pearson_correlation(pitch_weights, minor_profile)
            key_scores[f"{self.NOTE_NAMES[root_pc]}_minor"] = max(0.0, corr_minor)
        
        return key_scores


def analyze_with_robust_detector(
    audio_path: str,
    model_path: str,
    mapping_path: str,
    device: str = 'cpu'
) -> dict:
    """使用鲁棒检测器分析音频"""
    
    print(f"\n{'='*80}")
    print(f"分析文件: {Path(audio_path).name}")
    print(f"{'='*80}\n")
    
    # 1. 使用CQT分析器提取和弦
    print("步骤 1: 提取和弦进行...")
    analyzer = CQTChordAnalyzer(
        model_path=model_path,
        label_mapping_path=mapping_path,
        device=device
    )
    
    chord_progression = analyzer.analyze_progression(
        str(audio_path),
        overlap=0.5
    )
    
    # 平滑
    chord_progression = analyzer.smooth_predictions(
        chord_progression,
        min_duration=0.5
    )
    
    print(f"  ✓ 检测到 {len(chord_progression)} 个和弦段\n")
    
    # 2. 使用原始检测器
    print("步骤 2: 原始调性检测...")
    original_detector = OriginalKeyDetector(
        window_size=16,
        min_chords=6,
        confidence_threshold=0.65
    )
    
    for start, end, chord, conf in chord_progression:
        original_detector.add_chord(chord, conf)
    
    original_result = original_detector.get_current_key()
    original_top3 = original_detector.get_top_keys(3)
    
    if original_result['key']:
        print(f"  原始方法: {original_result['key']} "
              f"(置信度: {original_result['confidence']:.1%})")
    else:
        print(f"  原始方法: 无法确定")
    
    # 3. 使用鲁棒检测器
    print("\n步骤 3: 鲁棒调性检测...")
    robust_detector = RobustKeyDetector(
        window_size=16,
        min_chords=6,
        confidence_threshold=0.60,
        use_pattern_analysis=True,
        use_interval_analysis=True,
        pitch_class_weight=0.3
    )
    
    for start, end, chord, conf in chord_progression:
        robust_detector.add_chord(chord, conf)
    
    robust_result = robust_detector.get_current_key()
    robust_top3 = robust_detector.get_top_keys(3)
    
    if robust_result['key']:
        print(f"  鲁棒方法: {robust_result['key']} "
              f"(置信度: {robust_result['confidence']:.1%})")
    else:
        print(f"  鲁棒方法: 无法确定")
    
    # 4. 显示详细对比
    print(f"\n{'='*80}")
    print("详细对比")
    print(f"{'='*80}\n")
    
    print("原始方法 Top 3:")
    for i, (key, score) in enumerate(original_top3, 1):
        print(f"  {i}. {key:<15} {score*100:>5.1f}%")
    
    print("\n鲁棒方法 Top 3:")
    for i, (key, score) in enumerate(robust_top3, 1):
        print(f"  {i}. {key:<15} {score*100:>5.1f}%")
    
    # 5. 和弦统计
    print(f"\n{'='*80}")
    print("和弦统计")
    print(f"{'='*80}\n")
    
    from collections import Counter
    chord_types = Counter()
    for _, _, chord, _ in chord_progression:
        if '_' in chord:
            chord_type = chord.split('_')[1]
            chord_types[chord_type] += 1
    
    print("和弦类型分布:")
    for chord_type, count in chord_types.most_common(10):
        percentage = (count / len(chord_progression)) * 100
        print(f"  {chord_type:<12} {count:>3} 次  ({percentage:>5.1f}%)")
    
    # 6. 鲁棒检测器分析详情
    print(f"\n{'='*80}")
    print("鲁棒检测器分析详情")
    print(f"{'='*80}\n")
    
    details = robust_detector.get_analysis_details()
    if details['status'] == 'analyzed':
        print("和弦质量分布:")
        for quality, count in sorted(details['chord_qualities'].items(), 
                                     key=lambda x: -x[1]):
            print(f"  {quality:<18} {count:>3} 次")
        
        print("\n音程分布 (半音):")
        interval_names = {
            0: '同根音', 1: '小二度', 2: '大二度', 3: '小三度',
            4: '大三度', 5: '纯四度', 6: '增四度/减五度', 7: '纯五度',
            8: '小六度', 9: '大六度', 10: '小七度', 11: '大七度'
        }
        for interval, count in sorted(details['intervals'].items(), 
                                      key=lambda x: -x[1])[:8]:
            name = interval_names.get(interval, f'{interval}半音')
            print(f"  {name:<20} {count:>3} 次")
    
    return {
        'audio_path': audio_path,
        'original_method': {
            'key': original_result['key'],
            'confidence': original_result['confidence'],
            'top3': original_top3
        },
        'robust_method': {
            'key': robust_result['key'],
            'confidence': robust_result['confidence'],
            'top3': robust_top3
        },
        'chord_progression': chord_progression,
        'statistics': {
            'total_chords': len(chord_progression),
            'chord_types': dict(chord_types)
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description='使用鲁棒调性检测器分析音频',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 分析单个文件
  python analyze_with_robust_key.py \\
      tonal_progressions_midi_output/tonal_001_satb_F_major_*.wav \\
      --model models_full_cqt/chord_model_full_20251116_005133.pth
  
  # 批量分析并保存结果
  python analyze_with_robust_key.py \\
      tonal_progressions_midi_output/tonal_002_*.wav \\
      --model models_full_cqt/chord_model_full_20251116_005133.pth \\
      --output robust_results.json
        """
    )
    
    parser.add_argument('audio_path', type=str, help='音频文件路径')
    parser.add_argument('--model', type=str, required=True, help='CQT模型路径')
    parser.add_argument('--mappings', type=str, help='标签映射路径（可选，自动查找）')
    parser.add_argument('--device', type=str, default='cpu', help='设备 (cpu/cuda/mps)')
    parser.add_argument('--output', type=str, help='输出JSON文件路径（可选）')
    
    args = parser.parse_args()
    
    # 检查文件
    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        print(f"错误: 文件不存在 - {audio_path}")
        sys.exit(1)
    
    # 查找映射文件
    if args.mappings is None:
        model_dir = Path(args.model).parent
        mapping_files = list(model_dir.glob('label_mappings_*.json'))
        if not mapping_files:
            print("错误: 未找到标签映射文件")
            sys.exit(1)
        args.mappings = str(mapping_files[0])
    
    # 分析
    result = analyze_with_robust_detector(
        str(audio_path),
        args.model,
        args.mappings,
        args.device
    )
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为可序列化格式
        serializable = {
            'audio_path': result['audio_path'],
            'original_method': {
                'key': result['original_method']['key'],
                'confidence': float(result['original_method']['confidence']),
                'top3': [(k, float(s)) for k, s in result['original_method']['top3']]
            },
            'robust_method': {
                'key': result['robust_method']['key'],
                'confidence': float(result['robust_method']['confidence']),
                'top3': [(k, float(s)) for k, s in result['robust_method']['top3']]
            },
            'statistics': result['statistics']
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 结果已保存到: {output_path}")
    
    print(f"\n{'='*80}")
    print("分析完成！")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
