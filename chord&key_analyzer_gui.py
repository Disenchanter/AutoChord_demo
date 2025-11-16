#!/usr/bin/env python3
"""
实时和弦分析GUI - Tkinter版本
加载音频文件,播放并显示置信度前三的和弦预测 + 实时调性检测
"""

import tkinter as tk
from tkinter import filedialog, ttk
import json
from pathlib import Path
from collections import deque
from typing import List, Tuple, Dict
import numpy as np
import torch
import librosa
import sounddevice as sd
import threading
import time
from train_chord_cqt import ChordCNN


# ==================== 调性检测器 ====================

class KeyDetector:
    """调性检测器 (基于 Krumhansl-Schmuckler 算法)"""
    
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
        min_chords: int = 4,
        confidence_threshold: float = 0.60,
        use_confidence_weighting: bool = True,
        decay_factor: float = 0.98
    ):
        """
        Args:
            min_chords: 开始判断所需的最少和弦数
            confidence_threshold: 输出调性的最低置信度
            use_confidence_weighting: 是否使用和弦置信度加权
            decay_factor: 历史衰减因子(0-1)，越大历史影响越持久
        """
        self.min_chords = min_chords
        self.confidence_threshold = confidence_threshold
        self.use_confidence_weighting = use_confidence_weighting
        self.decay_factor = decay_factor
        
        # 改用列表记录所有历史和弦
        self.chord_history = []
        self.confidence_history = []
        self.current_key = None
        self.confidence = 0.0
    
    def add_chord(self, chord_name: str, confidence: float = 1.0) -> Dict:
        """添加新识别的和弦并更新调性判断"""
        self.chord_history.append(chord_name)
        self.confidence_history.append(confidence)
        
        if len(self.chord_history) >= self.min_chords:
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
        
        # 更新当前调性 (使用滞后机制避免频繁跳变)
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
        """计算所有历史和弦中每个音高类的权重（带时间衰减）"""
        weights = [0.0] * 12
        total_chords = len(self.chord_history)
        
        for i, chord_name in enumerate(self.chord_history):
            root_str, chord_type = self._parse_chord(chord_name)
            if root_str is None:
                continue
                
            try:
                root_pc = self.NOTE_NAMES.index(root_str)
            except ValueError:
                continue
            
            chord_notes = self.CHORD_NOTES.get(chord_type, [0])
            
            # 时间衰减权重：越新的和弦权重越高
            # 使用指数衰减：decay_factor^(total-i-1)
            time_decay = self.decay_factor ** (total_chords - i - 1)
            
            # 置信度加权
            if self.use_confidence_weighting:
                conf_weight = self.confidence_history[i]
                weight = time_decay * conf_weight
            else:
                weight = time_decay
            
            # 累加音高类权重
            for note_offset in chord_notes:
                pc = (root_pc + note_offset) % 12
                weights[pc] += weight
        
        return weights
    
    def _parse_chord(self, chord_name: str) -> Tuple[str, str]:
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
        """计算 Pearson 相关系数"""
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
        """获取当前调性信息"""
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
        
        pitch_weights = self._calculate_pitch_weights()
        key_scores = {}
        
        for root_pc in range(12):
            major_profile = self._rotate_profile(self.MAJOR_PROFILE, root_pc)
            key_scores[f"{self.NOTE_NAMES[root_pc]}_major"] = max(0.0, self._pearson_correlation(pitch_weights, major_profile))
            
            minor_profile = self._rotate_profile(self.MINOR_PROFILE, root_pc)
            key_scores[f"{self.NOTE_NAMES[root_pc]}_minor"] = max(0.0, self._pearson_correlation(pitch_weights, minor_profile))
        
        sorted_keys = sorted(key_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_keys[:top_n]
    
    def reset(self):
        """重置检测器"""
        self.chord_history.clear()
        self.confidence_history.clear()
        self.current_key = None
        self.confidence = 0.0


class ChordAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("实时和弦与调性分析器")
        self.root.geometry("700x650")
        
        # 状态变量
        self.model = None
        self.idx_to_label = None
        self.audio_data = None
        self.sr = 22050
        self.is_playing = False
        self.current_position = 0
        self.play_thread = None
        
        # 参数
        self.window_size = 2.0  # 秒
        self.hop_length = 512
        self.n_bins = 84
        self.bins_per_octave = 12
        self.device = 'cpu'
        
        # 调性检测器
        self.key_detector = KeyDetector(
            min_chords=4,
            confidence_threshold=0.60,
            decay_factor=0.98  # 历史衰减因子，0.98表示100个和弦后影响降至13%
        )
        
        self.create_widgets()
        
    def create_widgets(self):
        # 标题
        title_label = tk.Label(
            self.root, 
            text="🎵 实时和弦分析器", 
            font=("Arial", 20, "bold"),
            pady=10
        )
        title_label.pack()
        
        # 模型加载区域
        model_frame = tk.LabelFrame(self.root, text="模型设置", padx=10, pady=10)
        model_frame.pack(fill="x", padx=20, pady=5)
        
        tk.Button(
            model_frame, 
            text="加载模型", 
            command=self.load_model,
            width=15
        ).pack(side="left", padx=5)
        
        self.model_status = tk.Label(model_frame, text="未加载模型", fg="red")
        self.model_status.pack(side="left", padx=10)
        
        # 音频文件区域
        audio_frame = tk.LabelFrame(self.root, text="音频文件", padx=10, pady=10)
        audio_frame.pack(fill="x", padx=20, pady=5)
        
        tk.Button(
            audio_frame, 
            text="选择音频", 
            command=self.load_audio,
            width=15
        ).pack(side="left", padx=5)
        
        self.audio_status = tk.Label(audio_frame, text="未加载音频", fg="gray")
        self.audio_status.pack(side="left", padx=10)
        
        # 控制按钮
        control_frame = tk.Frame(self.root, pady=10)
        control_frame.pack()
        
        self.play_button = tk.Button(
            control_frame,
            text="▶ 播放分析",
            command=self.toggle_playback,
            state="disabled",
            font=("Arial", 12),
            width=15,
            bg="#4CAF50",
            fg="white"
        )
        self.play_button.pack()
        
        # 进度条
        progress_frame = tk.Frame(self.root, padx=20)
        progress_frame.pack(fill="x", pady=10)
        
        tk.Label(progress_frame, text="播放进度:").pack()
        self.progress_bar = ttk.Progressbar(
            progress_frame, 
            length=500, 
            mode='determinate'
        )
        self.progress_bar.pack(pady=5)
        
        self.time_label = tk.Label(progress_frame, text="0.00s / 0.00s")
        self.time_label.pack()
        
        # 调性检测显示区域 (新增)
        key_frame = tk.LabelFrame(
            self.root,
            text="🎼 实时调性检测",
            padx=10,
            pady=10,
            font=("Arial", 11, "bold")
        )
        key_frame.pack(fill="x", padx=20, pady=5)
        
        # 当前调性显示
        key_display_frame = tk.Frame(key_frame)
        key_display_frame.pack(fill="x")
        
        tk.Label(key_display_frame, text="当前调性:", font=("Arial", 10)).pack(side="left", padx=5)
        self.key_label = tk.Label(
            key_display_frame,
            text="分析中...",
            font=("Arial", 14, "bold"),
            fg="blue",
            width=15
        )
        self.key_label.pack(side="left", padx=5)
        
        tk.Label(key_display_frame, text="置信度:", font=("Arial", 10)).pack(side="left", padx=5)
        self.key_confidence_label = tk.Label(
            key_display_frame,
            text="0.0%",
            font=("Arial", 12, "bold"),
            width=6
        )
        self.key_confidence_label.pack(side="left")
        
        # 备选调性显示 (Top 3)
        alt_keys_frame = tk.Frame(key_frame)
        alt_keys_frame.pack(fill="x", pady=(5, 0))
        
        tk.Label(alt_keys_frame, text="备选:", font=("Arial", 9), fg="gray").pack(side="left", padx=5)
        self.alt_keys_label = tk.Label(
            alt_keys_frame,
            text="---",
            font=("Arial", 9),
            fg="gray"
        )
        self.alt_keys_label.pack(side="left")
        
        # 和弦预测显示区域
        prediction_frame = tk.LabelFrame(
            self.root, 
            text="实时和弦预测 (置信度Top 3)", 
            padx=10, 
            pady=10,
            font=("Arial", 11, "bold")
        )
        prediction_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Top 3和弦显示
        self.chord_labels = []
        self.confidence_bars = []
        
        for i in range(3):
            # 创建每个和弦的显示行
            chord_frame = tk.Frame(prediction_frame)
            chord_frame.pack(fill="x", pady=5)
            
            # 排名标签
            rank_label = tk.Label(
                chord_frame, 
                text=f"#{i+1}", 
                font=("Arial", 14, "bold"),
                width=3
            )
            rank_label.pack(side="left")
            
            # 和弦名称
            chord_label = tk.Label(
                chord_frame,
                text="---",
                font=("Arial", 16, "bold"),
                width=12,
                anchor="w"
            )
            chord_label.pack(side="left", padx=5)
            self.chord_labels.append(chord_label)
            
            # 置信度进度条
            conf_bar = ttk.Progressbar(
                chord_frame,
                length=200,
                mode='determinate'
            )
            conf_bar.pack(side="left", padx=5)
            self.confidence_bars.append(conf_bar)
            
            # 置信度百分比
            conf_label = tk.Label(chord_frame, text="0.0%", width=6)
            conf_label.pack(side="left")
            self.chord_labels.append(conf_label)  # 复用列表存储
            
    def load_model(self):
        """加载模型和映射文件"""
        model_path = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("PyTorch模型", "*.pth"), ("所有文件", "*.*")]
        )
        
        if not model_path:
            return
            
        # 自动查找对应的映射文件
        model_dir = Path(model_path).parent
        model_name = Path(model_path).stem
        
        # 尝试找到对应的映射文件
        mapping_files = list(model_dir.glob("label_mappings*.json"))
        if not mapping_files:
            tk.messagebox.showerror("错误", "未找到标签映射文件")
            return
            
        # 使用最新的映射文件
        mapping_path = str(sorted(mapping_files)[-1])
        
        try:
            # 加载映射
            with open(mapping_path, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)
            
            task = mapping_data['task']
            num_classes = mapping_data['num_classes']
            mappings = mapping_data['mappings']
            
            # 构建反向映射
            if task == 'full':
                self.idx_to_label = {v: k for k, v in mappings['full_label_to_idx'].items()}
            elif task == 'root':
                self.idx_to_label = {v: k for k, v in mappings['root_to_idx'].items()}
            else:
                self.idx_to_label = {v: k for k, v in mappings['chord_to_idx'].items()}
            
            # 加载模型
            self.model = ChordCNN(num_classes=num_classes)
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.model.to(self.device)
            
            self.model_status.config(
                text=f"✓ 已加载 ({num_classes}类)", 
                fg="green"
            )
            
            # 如果已加载音频,启用播放按钮
            if self.audio_data is not None:
                self.play_button.config(state="normal")
                
        except Exception as e:
            tk.messagebox.showerror("加载模型失败", str(e))
            
    def load_audio(self):
        """加载音频文件"""
        audio_path = filedialog.askopenfilename(
            title="选择音频文件",
            filetypes=[
                ("音频文件", "*.wav *.mp3 *.flac"),
                ("所有文件", "*.*")
            ]
        )
        
        if not audio_path:
            return
            
        try:
            # 加载音频
            self.audio_data, self.sr = librosa.load(audio_path, sr=22050, mono=True)
            
            duration = len(self.audio_data) / self.sr
            filename = Path(audio_path).name
            
            self.audio_status.config(
                text=f"✓ {filename} ({duration:.1f}s)",
                fg="green"
            )
            
            self.time_label.config(text=f"0.00s / {duration:.2f}s")
            self.current_position = 0
            self.progress_bar['value'] = 0
            
            # 如果已加载模型,启用播放按钮
            if self.model is not None:
                self.play_button.config(state="normal")
                
        except Exception as e:
            tk.messagebox.showerror("加载音频失败", str(e))
            
    def extract_cqt_feature(self, audio_segment):
        """提取CQT特征"""
        C = librosa.cqt(
            audio_segment,
            sr=self.sr,
            hop_length=self.hop_length,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave
        )
        
        # 转换为dB
        C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)
        
        # 归一化
        C_norm = (C_db - C_db.min()) / (C_db.max() - C_db.min() + 1e-8)
        
        # 转换为tensor
        cqt_tensor = torch.FloatTensor(C_norm).unsqueeze(0).unsqueeze(0)
        
        return cqt_tensor
        
    def predict_chord(self, audio_segment):
        """预测和弦,返回Top 3结果"""
        if self.model is None:
            return None
            
        try:
            # 提取特征
            features = self.extract_cqt_feature(audio_segment)
            features = features.to(self.device)
            
            # 预测
            with torch.no_grad():
                output = self.model(features)
                probs = torch.softmax(output, dim=1)
                
            # 获取Top 3
            top3_probs, top3_indices = torch.topk(probs[0], 3)
            
            results = []
            for prob, idx in zip(top3_probs, top3_indices):
                chord_name = self.idx_to_label.get(idx.item(), "Unknown")
                confidence = prob.item()
                results.append((chord_name, confidence))
                
            return results
            
        except Exception as e:
            print(f"预测错误: {e}")
            return None
            
    def update_display(self, predictions, key_info=None):
        """更新显示的和弦预测和调性信息"""
        if predictions is None:
            return
            
        for i, (chord, confidence) in enumerate(predictions):
            # 更新和弦名称 (偶数索引)
            self.chord_labels[i*2].config(text=chord)
            
            # 更新进度条
            self.confidence_bars[i]['value'] = confidence * 100
            
            # 更新百分比标签 (奇数索引)
            self.chord_labels[i*2+1].config(text=f"{confidence*100:.1f}%")
            
            # 根据置信度改变颜色
            if i == 0:  # Top 1
                if confidence > 0.7:
                    color = "green"
                elif confidence > 0.4:
                    color = "orange"
                else:
                    color = "red"
                self.chord_labels[i*2].config(fg=color)
        
        # 更新调性显示 (新增)
        if key_info:
            if key_info['key']:
                key_display = key_info['key'].replace('_', ' ').title()
                self.key_label.config(text=key_display, fg="blue")
                self.key_confidence_label.config(
                    text=f"{key_info['confidence']*100:.1f}%"
                )
            else:
                chords_needed = 4  # min_chords
                current = key_info['chords_analyzed']
                self.key_label.config(
                    text=f"分析中 ({current}/{chords_needed})",
                    fg="gray"
                )
                self.key_confidence_label.config(text="--")
                
    def play_and_analyze(self):
        """播放音频并实时分析"""
        window_samples = int(self.window_size * self.sr)
        hop_samples = int(0.5 * self.sr)  # 0.5秒更新一次
        
        self.current_position = 0
        total_samples = len(self.audio_data)
        duration = total_samples / self.sr
        
        # 播放音频
        sd.play(self.audio_data, self.sr)
        
        while self.is_playing and self.current_position < total_samples:
            start_time = time.time()
            
            # 提取当前窗口
            end_pos = min(self.current_position + window_samples, total_samples)
            
            if end_pos - self.current_position < window_samples:
                # 最后一个窗口,填充
                segment = np.pad(
                    self.audio_data[self.current_position:end_pos],
                    (0, window_samples - (end_pos - self.current_position))
                )
            else:
                segment = self.audio_data[self.current_position:end_pos]
            
            # 预测和弦
            predictions = self.predict_chord(segment)
            
            # 更新调性检测器 (新增)
            key_info = None
            if predictions and len(predictions) > 0:
                top_chord, top_confidence = predictions[0]
                key_info = self.key_detector.add_chord(top_chord, top_confidence)
                
                # 获取备选调性
                top_keys = self.key_detector.get_top_keys(top_n=3)
                if top_keys:
                    alt_keys_text = " | ".join([
                        f"{k.replace('_', ' ').title()}({s*100:.0f}%)" 
                        for k, s in top_keys[1:]  # 跳过第1名（已在主显示）
                    ])
                    self.root.after(0, lambda txt=alt_keys_text: self.alt_keys_label.config(text=txt))
            
            # 更新显示（包含和弦和调性）
            if predictions:
                self.root.after(0, self.update_display, predictions, key_info)
            
            # 更新进度
            current_time = self.current_position / self.sr
            progress = (self.current_position / total_samples) * 100
            
            self.root.after(
                0, 
                lambda: self.progress_bar.config(value=progress)
            )
            self.root.after(
                0,
                lambda: self.time_label.config(
                    text=f"{current_time:.2f}s / {duration:.2f}s"
                )
            )
            
            # 移动到下一个窗口
            self.current_position += hop_samples
            
            # 控制更新频率
            elapsed = time.time() - start_time
            if elapsed < 0.5:
                time.sleep(0.5 - elapsed)
                
        # 播放结束
        sd.stop()
        self.is_playing = False
        self.root.after(0, lambda: self.play_button.config(text="▶ 播放分析"))
        
    def toggle_playback(self):
        """切换播放/停止"""
        if not self.is_playing:
            # 开始播放 - 重置调性检测器
            self.key_detector.reset()
            self.is_playing = True
            self.play_button.config(text="⏸ 停止")
            self.play_thread = threading.Thread(target=self.play_and_analyze)
            self.play_thread.daemon = True
            self.play_thread.start()
        else:
            # 停止播放
            self.is_playing = False
            sd.stop()
            self.play_button.config(text="▶ 播放分析")


def main():
    root = tk.Tk()
    app = ChordAnalyzerGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
