#!/usr/bin/env python3
"""
鲁棒调性检测器
降低对和弦根音识别误差的敏感度，更关注和弦类型模式和音程关系
"""

from collections import deque, Counter
from typing import List, Tuple, Dict, Optional
import numpy as np


class RobustKeyDetector:
    """
    改进的调性检测器 - 对根音识别误差更鲁棒
    
    策略：
    1. 使用和弦类型序列模式（major/minor分布）
    2. 分析和弦进行的音程关系（四度、五度进行）
    3. 统计音高类出现频率但降低权重
    4. 识别功能和声模式（I-IV-V等）
    """
    
    # Krumhansl-Kessler 音高轮廓（保留用于辅助判断）
    MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # 和弦类型到质量的映射
    CHORD_QUALITY = {
        'major': 'major',
        'minor': 'minor',
        'dim': 'diminished',
        'aug': 'augmented',
        'sus2': 'suspended',
        'sus4': 'suspended',
        'maj7': 'major7',
        'min7': 'minor7',
        'dom7': 'dominant7',
        'hdim7': 'half_diminished',
        'dim7': 'diminished7',
    }
    
    # 和弦音符（用于音高类统计）
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
    
    # 大调特征：major和弦占优，dom7常见
    MAJOR_CHORD_PATTERN = {
        'major': 0.50,      # 50% 大三和弦
        'dominant7': 0.15,  # 15% 属七
        'minor': 0.20,      # 20% 小三（ii, iii, vi）
        'major7': 0.10,     # 10% 大七
        'others': 0.05
    }
    
    # 小调特征：minor和弦占优
    MINOR_CHORD_PATTERN = {
        'minor': 0.45,      # 45% 小三和弦
        'major': 0.30,      # 30% 大三（III, VI, VII）
        'diminished': 0.10, # 10% 减和弦（ii°）
        'minor7': 0.10,     # 10% 小七
        'others': 0.05
    }
    
    def __init__(
        self,
        min_chords: int = 6,
        confidence_threshold: float = 0.60,
        use_pattern_analysis: bool = True,
        use_interval_analysis: bool = True,
        pitch_class_weight: float = 0.3,
        decay_factor: float = 0.98
    ):
        """
        Args:
            min_chords: 最少和弦数
            confidence_threshold: 置信度阈值
            use_pattern_analysis: 是否使用和弦类型模式分析
            use_interval_analysis: 是否使用音程关系分析
            pitch_class_weight: 音高类统计的权重（0-1）
            decay_factor: 历史衰减因子(0-1)，越大历史影响越持久
        """
        self.min_chords = min_chords
        self.confidence_threshold = confidence_threshold
        self.use_pattern_analysis = use_pattern_analysis
        self.use_interval_analysis = use_interval_analysis
        self.pitch_class_weight = pitch_class_weight
        self.decay_factor = decay_factor
        
        # 改用列表记录所有历史和弦
        self.chord_history = []
        self.confidence_history = []
        self.current_key = None
        self.confidence = 0.0
    
    def add_chord(self, chord_name: str, confidence: float = 1.0) -> Dict:
        """添加和弦并更新调性"""
        self.chord_history.append(chord_name)
        self.confidence_history.append(confidence)
        
        if len(self.chord_history) >= self.min_chords:
            self._update_key()
        
        return self.get_current_key()
    
    def _update_key(self):
        """更新调性判断"""
        scores = {}
        
        # 1. 和弦类型模式分析（权重最高）
        if self.use_pattern_analysis:
            pattern_scores = self._analyze_chord_pattern()
            for key, score in pattern_scores.items():
                scores[key] = scores.get(key, 0) + score * 0.5
        
        # 2. 音程关系分析
        if self.use_interval_analysis:
            interval_scores = self._analyze_intervals()
            for key, score in interval_scores.items():
                scores[key] = scores.get(key, 0) + score * 0.2
        
        # 3. 音高类统计（降低权重）
        pitch_scores = self._analyze_pitch_classes()
        for key, score in pitch_scores.items():
            scores[key] = scores.get(key, 0) + score * self.pitch_class_weight
        
        # 找出最佳调性
        if not scores:
            return
        
        best_key = max(scores, key=scores.get)
        best_score = scores[best_key]
        
        # 归一化分数
        if best_score > 0:
            best_score = min(1.0, best_score)
        
        # 更新调性（使用滞后机制）
        if self.current_key is None:
            if best_score >= self.confidence_threshold:
                self.current_key = best_key
                self.confidence = best_score
        else:
            if best_key != self.current_key:
                if best_score >= 0.75:  # 切换需要更高置信度
                    self.current_key = best_key
                    self.confidence = best_score
            else:
                self.confidence = best_score
    
    def _analyze_chord_pattern(self) -> Dict[str, float]:
        """
        分析和弦类型模式（基于所有历史和弦，带时间衰减）
        大调：大三和弦多，属七和弦常见
        小调：小三和弦多，减和弦常见
        """
        scores = {}
        
        # 统计和弦质量（带时间衰减）
        quality_counts = Counter()
        total_weight = 0.0
        total_chords = len(self.chord_history)
        
        for i, chord_name in enumerate(self.chord_history):
            _, chord_type = self._parse_chord(chord_name)
            if chord_type:
                quality = self._get_chord_quality(chord_type)
                
                # 时间衰减权重
                time_decay = self.decay_factor ** (total_chords - i - 1)
                # 置信度加权
                weight = time_decay * self.confidence_history[i]
                
                quality_counts[quality] += weight
                total_weight += weight
        
        if total_weight == 0:
            return scores
        
        # 归一化
        quality_dist = {q: c / total_weight for q, c in quality_counts.items()}
        
        # 对每个调性计算匹配度
        for root_pc in range(12):
            root_name = self.NOTE_NAMES[root_pc]
            
            # 大调匹配度
            major_score = 0.0
            major_score += quality_dist.get('major', 0) * 2.0  # 大三和弦
            major_score += quality_dist.get('dominant7', 0) * 1.5  # 属七
            major_score += quality_dist.get('major7', 0) * 1.0
            major_score += quality_dist.get('minor', 0) * 0.5  # 有些小三和弦
            major_score -= quality_dist.get('diminished', 0) * 0.5  # 减和弦少
            
            scores[f"{root_name}_major"] = major_score
            
            # 小调匹配度
            minor_score = 0.0
            minor_score += quality_dist.get('minor', 0) * 2.0  # 小三和弦
            minor_score += quality_dist.get('minor7', 0) * 1.5  # 小七
            minor_score += quality_dist.get('major', 0) * 0.8  # 也有大三和弦
            minor_score += quality_dist.get('diminished', 0) * 1.0  # 减和弦常见
            minor_score += quality_dist.get('half_diminished', 0) * 1.0
            
            scores[f"{root_name}_minor"] = minor_score
        
        return scores
    
    def _analyze_intervals(self) -> Dict[str, float]:
        """
        分析和弦进行的音程关系（基于所有历史和弦，带时间衰减）
        常见进行：四度上行（I-IV）、五度下行（V-I）
        """
        scores = {f"{self.NOTE_NAMES[i]}_{mode}": 0.0 
                  for i in range(12) for mode in ['major', 'minor']}
        
        if len(self.chord_history) < 2:
            return scores
        
        # 统计根音间的音程（带时间衰减）
        interval_counts = Counter()
        total_weight = 0.0
        total_chords = len(self.chord_history)
        
        for i in range(len(self.chord_history) - 1):
            root1, _ = self._parse_chord(self.chord_history[i])
            root2, _ = self._parse_chord(self.chord_history[i + 1])
            
            if root1 and root2:
                try:
                    pc1 = self.NOTE_NAMES.index(root1)
                    pc2 = self.NOTE_NAMES.index(root2)
                    interval = (pc2 - pc1) % 12
                    
                    # 时间衰减权重（对转换i->i+1使用第i+1个和弦的时间点）
                    time_decay = self.decay_factor ** (total_chords - i - 2)
                    interval_counts[interval] += time_decay
                    total_weight += time_decay
                except ValueError:
                    continue
        
        if total_weight == 0:
            return scores
        
        # 常见功能和声音程
        # 五度上行（7半音）：I-V, IV-I
        # 四度上行（5半音）：I-IV
        # 二度上行（2半音）：I-ii, IV-V
        
        fifth_up = interval_counts[7] / total_weight if total_weight > 0 else 0
        fourth_up = interval_counts[5] / total_weight if total_weight > 0 else 0
        second_up = interval_counts[2] / total_weight if total_weight > 0 else 0
        
        # 大调和小调都喜欢这些音程，轻微加分
        boost = (fifth_up * 0.5 + fourth_up * 0.3 + second_up * 0.2)
        
        for key in scores:
            scores[key] += boost
        
        return scores
    
    def _analyze_pitch_classes(self) -> Dict[str, float]:
        """传统的音高类统计分析（Krumhansl-Schmuckler，基于所有历史和弦）"""
        pitch_weights = [0.0] * 12
        total_chords = len(self.chord_history)
        
        for i, chord_name in enumerate(self.chord_history):
            root_str, chord_type = self._parse_chord(chord_name)
            if not root_str or not chord_type:
                continue
            
            try:
                root_pc = self.NOTE_NAMES.index(root_str)
            except ValueError:
                continue
            
            chord_notes = self.CHORD_NOTES.get(chord_type, [0])
            
            # 时间衰减权重
            time_decay = self.decay_factor ** (total_chords - i - 1)
            weight = time_decay * self.confidence_history[i]
            
            for note_offset in chord_notes:
                pc = (root_pc + note_offset) % 12
                pitch_weights[pc] += weight
        
        # 计算相关性
        scores = {}
        for root_pc in range(12):
            major_profile = self._rotate_profile(self.MAJOR_PROFILE, root_pc)
            minor_profile = self._rotate_profile(self.MINOR_PROFILE, root_pc)
            
            scores[f"{self.NOTE_NAMES[root_pc]}_major"] = max(0.0, 
                self._pearson_correlation(pitch_weights, major_profile))
            scores[f"{self.NOTE_NAMES[root_pc]}_minor"] = max(0.0,
                self._pearson_correlation(pitch_weights, minor_profile))
        
        return scores
    
    def _get_chord_quality(self, chord_type: str) -> str:
        """获取和弦质量"""
        return self.CHORD_QUALITY.get(chord_type, 'other')
    
    def _parse_chord(self, chord_name: str) -> Tuple[Optional[str], Optional[str]]:
        """解析和弦名称"""
        if '_' not in chord_name:
            return None, None
        
        parts = chord_name.split('_')
        if len(parts) != 2:
            return None, None
        
        root, chord_type = parts
        root = root.replace('sharp', '#').replace('flat', 'b')
        
        return root, chord_type
    
    def _rotate_profile(self, profile: List[float], shift: int) -> List[float]:
        """旋转音高轮廓"""
        return profile[shift:] + profile[:shift]
    
    def _pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """Pearson相关系数"""
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi ** 2 for xi in x)
        sum_y2 = sum(yi ** 2 for yi in y)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def get_current_key(self) -> Dict:
        """获取当前调性"""
        if self.current_key is None:
            return {
                'key': None,
                'confidence': 0.0,
                'chords_analyzed': len(self.chord_history)
            }
        else:
            return {
                'key': self.current_key,
                'confidence': self.confidence,
                'chords_analyzed': len(self.chord_history)
            }
    
    def get_top_keys(self, top_n: int = 3) -> List[Tuple[str, float]]:
        """获取最可能的前N个调性"""
        if len(self.chord_history) < self.min_chords:
            return []
        
        scores = {}
        
        # 综合三种分析方法
        if self.use_pattern_analysis:
            pattern_scores = self._analyze_chord_pattern()
            for key, score in pattern_scores.items():
                scores[key] = scores.get(key, 0) + score * 0.5
        
        if self.use_interval_analysis:
            interval_scores = self._analyze_intervals()
            for key, score in interval_scores.items():
                scores[key] = scores.get(key, 0) + score * 0.2
        
        pitch_scores = self._analyze_pitch_classes()
        for key, score in pitch_scores.items():
            scores[key] = scores.get(key, 0) + score * self.pitch_class_weight
        
        # 归一化并排序
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: v / max_score for k, v in scores.items()}
        
        sorted_keys = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_keys[:top_n]
    
    def reset(self):
        """重置状态"""
        self.chord_history.clear()
        self.confidence_history.clear()
        self.current_key = None
        self.confidence = 0.0
    
    def get_analysis_details(self) -> Dict:
        """获取详细分析信息（用于调试）"""
        if len(self.chord_history) < self.min_chords:
            return {'status': 'insufficient_data'}
        
        # 和弦质量分布
        quality_counts = Counter()
        for chord_name in self.chord_history:
            _, chord_type = self._parse_chord(chord_name)
            if chord_type:
                quality = self._get_chord_quality(chord_type)
                quality_counts[quality] += 1
        
        # 音程分布
        interval_counts = Counter()
        for i in range(len(self.chord_history) - 1):
            root1, _ = self._parse_chord(self.chord_history[i])
            root2, _ = self._parse_chord(self.chord_history[i + 1])
            if root1 and root2:
                try:
                    pc1 = self.NOTE_NAMES.index(root1)
                    pc2 = self.NOTE_NAMES.index(root2)
                    interval = (pc2 - pc1) % 12
                    interval_counts[interval] += 1
                except ValueError:
                    pass
        
        return {
            'status': 'analyzed',
            'chord_qualities': dict(quality_counts),
            'intervals': dict(interval_counts),
            'pattern_scores': self._analyze_chord_pattern(),
            'interval_scores': self._analyze_intervals(),
            'pitch_class_scores': self._analyze_pitch_classes()
        }


# 测试代码
if __name__ == '__main__':
    # 测试用例：C大调进行（但根音可能有误差）
    detector = RobustKeyDetector(window_size=12, min_chords=4)
    
    # 模拟和弦序列（有些根音识别错误）
    # 真实: C-F-G-C, 识别为: C-F-A-C (G被错认为A)
    test_chords = [
        ('C_major', 0.8),
        ('F_major', 0.7),
        ('A_major', 0.6),  # 实际是G major
        ('C_major', 0.8),
        ('D_minor', 0.7),
        ('F_major', 0.75),
        ('A_major', 0.65),  # 实际是G major
        ('C_major', 0.85),
    ]
    
    print("测试鲁棒调性检测器")
    print("=" * 60)
    
    for i, (chord, conf) in enumerate(test_chords, 1):
        result = detector.add_chord(chord, conf)
        print(f"\n步骤 {i}: 添加和弦 {chord} (置信度: {conf:.1%})")
        
        if result['key']:
            print(f"  检测到调性: {result['key']} (置信度: {result['confidence']:.1%})")
            top_keys = detector.get_top_keys(3)
            print(f"  备选调性: {', '.join([f'{k}({s:.1%})' for k, s in top_keys[:3]])}")
        else:
            print(f"  状态: 正在分析... ({result['chords_analyzed']} 和弦)")
    
    print("\n" + "=" * 60)
    print("分析详情:")
    details = detector.get_analysis_details()
    if details['status'] == 'analyzed':
        print(f"\n和弦质量分布: {details['chord_qualities']}")
        print(f"音程分布: {details['intervals']}")
