#!/usr/bin/env python3
"""
和弦识别播放器 - 独立应用程序
加载 WAV 文件，实时播放并显示和弦预测结果

功能：
- 加载音频文件（WAV, MP3, FLAC 等）
- 实时播放音频
- 滑动窗口和弦识别（2秒窗口）
- 可视化显示：波形、频谱、和弦序列
- 支持暂停/播放/跳转
"""

import sys
import os
from pathlib import Path
from typing import Optional, List, Tuple
import json

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from scipy.signal import resample

# GUI 库
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QSlider, QFileDialog, QComboBox, QGroupBox
    )
    from PyQt5.QtCore import QTimer, Qt, pyqtSignal
    from PyQt5.QtGui import QFont
    GUI_AVAILABLE = True
except ImportError:
    print("警告: PyQt5 未安装，将使用命令行模式")
    GUI_AVAILABLE = False

# 音频播放库
try:
    import pyaudio
    AUDIO_AVAILABLE = True
except ImportError:
    print("警告: pyaudio 未安装，音频播放功能不可用")
    AUDIO_AVAILABLE = False

# 可视化库
try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False


# ==================== 模型定义 ====================

class ChordCNN(nn.Module):
    """和弦识别 CNN 模型（与训练脚本一致）"""
    
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


# ==================== 和弦识别引擎 ====================

class ChordRecognitionEngine:
    """和弦识别引擎"""
    
    def __init__(
        self,
        model_path: str,
        label_mapping_path: str,
        target_sr: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        window_duration: float = 2.0,
        device: str = 'cpu'
    ):
        self.target_sr = target_sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_duration = window_duration
        self.window_samples = int(target_sr * window_duration)
        self.device = device
        
        # 加载标签映射
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
        
        num_classes = len(self.idx_to_label)
        
        # 加载模型
        self.model = ChordCNN(num_classes=num_classes)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # STFT 变换
        self.spectrogram_transform = T.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=2.0,
        )
        self.amplitude_to_db = T.AmplitudeToDB()
        
        print(f"✓ 模型加载成功: {num_classes} 类 ({self.task} 任务)")
        print(f"✓ 采样率: {target_sr} Hz, 窗口: {window_duration} 秒")
    
    def predict(self, audio_segment: np.ndarray, sr: int) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        预测音频片段的和弦
        
        Args:
            audio_segment: 音频片段 (numpy array)
            sr: 采样率
            
        Returns:
            (predicted_chord, confidence, top_5_predictions)
        """
        # 转换为 tensor
        waveform = torch.from_numpy(audio_segment).float().unsqueeze(0)
        
        # 重采样
        if sr != self.target_sr:
            resampler = T.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
        
        # 裁剪或填充
        if waveform.shape[1] < self.window_samples:
            padding = self.window_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            waveform = waveform[:, :self.window_samples]
        
        # 提取 STFT 特征
        spec = self.spectrogram_transform(waveform)
        spec_db = self.amplitude_to_db(spec)
        
        # 归一化
        spec_db = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min() + 1e-8)
        
        # 预测
        with torch.no_grad():
            spec_db = spec_db.unsqueeze(0).to(self.device)  # [1, 1, freq, time]
            outputs = self.model(spec_db)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        
        # 获取 top-5 预测
        top_5_idx = np.argsort(probs)[-5:][::-1]
        top_5 = [(self.idx_to_label[idx], float(probs[idx])) for idx in top_5_idx]
        
        predicted_idx = top_5_idx[0]
        predicted_chord = self.idx_to_label[predicted_idx]
        confidence = float(probs[predicted_idx])
        
        return predicted_chord, confidence, top_5


# ==================== 音频播放器 ====================

class AudioPlayer:
    """音频播放器"""
    
    def __init__(self, wav_path: str, target_sr: int = 22050):
        self.wav_path = wav_path
        self.target_sr = target_sr
        
        # 加载音频
        waveform, sr = torchaudio.load(wav_path)
        
        # 转换为单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 重采样
        if sr != target_sr:
            resampler = T.Resample(sr, target_sr)
            waveform = resampler(waveform)
        
        self.audio_data = waveform.squeeze().numpy()
        self.sr = target_sr
        self.duration = len(self.audio_data) / self.sr
        
        # PyAudio 初始化
        if AUDIO_AVAILABLE:
            self.p = pyaudio.PyAudio()
            self.stream = None
        
        self.current_position = 0
        self.is_playing = False
        
        print(f"✓ 音频加载成功: {self.duration:.2f} 秒, {self.sr} Hz")
    
    def play(self):
        """开始播放"""
        if not AUDIO_AVAILABLE:
            print("音频播放功能不可用（pyaudio 未安装）")
            return
        
        if self.stream is None:
            self.stream = self.p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sr,
                output=True
            )
        
        self.is_playing = True
    
    def pause(self):
        """暂停播放"""
        self.is_playing = False
    
    def stop(self):
        """停止播放"""
        self.is_playing = False
        self.current_position = 0
        if self.stream:
            self.stream.stop_stream()
    
    def seek(self, position: float):
        """跳转到指定位置（秒）"""
        self.current_position = int(position * self.sr)
        self.current_position = max(0, min(self.current_position, len(self.audio_data)))
    
    def get_current_audio(self, duration: float = 2.0) -> np.ndarray:
        """获取当前位置的音频片段"""
        samples = int(duration * self.sr)
        end_pos = min(self.current_position + samples, len(self.audio_data))
        
        segment = self.audio_data[self.current_position:end_pos]
        
        # 如果不够长，填充
        if len(segment) < samples:
            segment = np.pad(segment, (0, samples - len(segment)))
        
        return segment
    
    def update(self, chunk_size: int = 1024):
        """更新播放（调用此方法推进播放）"""
        if self.is_playing and AUDIO_AVAILABLE:
            end_pos = min(self.current_position + chunk_size, len(self.audio_data))
            
            if self.current_position >= len(self.audio_data):
                self.stop()
                return
            
            chunk = self.audio_data[self.current_position:end_pos].astype(np.float32)
            self.stream.write(chunk.tobytes())
            self.current_position = end_pos
    
    def close(self):
        """关闭播放器"""
        if AUDIO_AVAILABLE:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            self.p.terminate()


# ==================== GUI 主窗口 ====================

if GUI_AVAILABLE:
    class ChordPlayerWindow(QMainWindow):
        """和弦识别播放器主窗口"""
        
        def __init__(self):
            super().__init__()
            self.setWindowTitle("和弦识别播放器 - AutoChord Demo")
            self.setGeometry(100, 100, 1200, 800)
            
            self.player = None
            self.engine = None
            self.current_chord = "--"
            self.confidence = 0.0
            self.chord_history = []  # [(time, chord, confidence), ...]
            
            self.setup_ui()
            
            # 定时器（用于更新播放和识别）
            self.timer = QTimer()
            self.timer.timeout.connect(self.update)
            self.timer.start(50)  # 50ms 更新一次
            
            self.recognition_timer = QTimer()
            self.recognition_timer.timeout.connect(self.recognize_chord)
            self.recognition_timer.start(500)  # 500ms 识别一次
        
        def setup_ui(self):
            """设置 UI"""
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            layout = QVBoxLayout(central_widget)
            
            # 1. 文件加载区域
            file_group = QGroupBox("文件加载")
            file_layout = QHBoxLayout()
            
            self.file_label = QLabel("未加载文件")
            self.file_label.setStyleSheet("QLabel { color: gray; }")
            file_layout.addWidget(self.file_label)
            
            btn_load_audio = QPushButton("加载音频")
            btn_load_audio.clicked.connect(self.load_audio)
            file_layout.addWidget(btn_load_audio)
            
            btn_load_model = QPushButton("加载模型")
            btn_load_model.clicked.connect(self.load_model)
            file_layout.addWidget(btn_load_model)
            
            file_group.setLayout(file_layout)
            layout.addWidget(file_group)
            
            # 2. 和弦显示区域
            chord_group = QGroupBox("当前和弦")
            chord_layout = QVBoxLayout()
            
            self.chord_label = QLabel("--")
            self.chord_label.setAlignment(Qt.AlignCenter)
            self.chord_label.setFont(QFont("Arial", 72, QFont.Bold))
            self.chord_label.setStyleSheet("QLabel { color: #00FF00; }")
            chord_layout.addWidget(self.chord_label)
            
            self.confidence_label = QLabel("置信度: --")
            self.confidence_label.setAlignment(Qt.AlignCenter)
            self.confidence_label.setFont(QFont("Arial", 18))
            chord_layout.addWidget(self.confidence_label)
            
            chord_group.setLayout(chord_layout)
            layout.addWidget(chord_group)
            
            # 3. 播放控制区域
            control_group = QGroupBox("播放控制")
            control_layout = QVBoxLayout()
            
            # 进度条
            self.progress_slider = QSlider(Qt.Horizontal)
            self.progress_slider.setMinimum(0)
            self.progress_slider.setMaximum(1000)
            self.progress_slider.sliderPressed.connect(self.on_slider_pressed)
            self.progress_slider.sliderReleased.connect(self.on_slider_released)
            control_layout.addWidget(self.progress_slider)
            
            self.time_label = QLabel("00:00 / 00:00")
            self.time_label.setAlignment(Qt.AlignCenter)
            control_layout.addWidget(self.time_label)
            
            # 按钮
            btn_layout = QHBoxLayout()
            
            self.btn_play = QPushButton("▶ 播放")
            self.btn_play.clicked.connect(self.toggle_play)
            self.btn_play.setEnabled(False)
            btn_layout.addWidget(self.btn_play)
            
            btn_stop = QPushButton("■ 停止")
            btn_stop.clicked.connect(self.stop)
            btn_layout.addWidget(btn_stop)
            
            control_layout.addLayout(btn_layout)
            control_group.setLayout(control_layout)
            layout.addWidget(control_group)
            
            # 4. Top-5 预测显示
            top5_group = QGroupBox("Top-5 预测")
            self.top5_layout = QVBoxLayout()
            self.top5_labels = []
            for i in range(5):
                label = QLabel(f"{i+1}. --")
                label.setFont(QFont("Arial", 12))
                self.top5_labels.append(label)
                self.top5_layout.addWidget(label)
            top5_group.setLayout(self.top5_layout)
            layout.addWidget(top5_group)
            
            self.slider_pressed = False
        
        def load_audio(self):
            """加载音频文件"""
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择音频文件", "", "Audio Files (*.wav *.mp3 *.flac)"
            )
            
            if file_path:
                try:
                    self.player = AudioPlayer(file_path)
                    self.file_label.setText(f"音频: {Path(file_path).name}")
                    self.file_label.setStyleSheet("QLabel { color: green; }")
                    self.btn_play.setEnabled(True)
                    self.chord_history = []
                    print(f"✓ 音频加载: {file_path}")
                except Exception as e:
                    self.file_label.setText(f"加载失败: {e}")
                    self.file_label.setStyleSheet("QLabel { color: red; }")
        
        def load_model(self):
            """加载模型"""
            model_path, _ = QFileDialog.getOpenFileName(
                self, "选择模型文件", "", "Model Files (*.pth)"
            )
            
            if not model_path:
                return
            
            # 自动查找对应的 label mapping
            model_dir = Path(model_path).parent
            mapping_files = list(model_dir.glob("label_mappings_*.json"))
            
            if not mapping_files:
                self.file_label.setText("未找到标签映射文件")
                return
            
            mapping_path = mapping_files[0]
            
            try:
                device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
                self.engine = ChordRecognitionEngine(
                    model_path=model_path,
                    label_mapping_path=str(mapping_path),
                    device=device
                )
                self.file_label.setText(f"模型: {Path(model_path).name}")
                print(f"✓ 模型加载: {model_path}")
            except Exception as e:
                self.file_label.setText(f"模型加载失败: {e}")
                print(f"✗ 错误: {e}")
        
        def toggle_play(self):
            """切换播放/暂停"""
            if self.player:
                if self.player.is_playing:
                    self.player.pause()
                    self.btn_play.setText("▶ 播放")
                else:
                    self.player.play()
                    self.btn_play.setText("⏸ 暂停")
        
        def stop(self):
            """停止播放"""
            if self.player:
                self.player.stop()
                self.btn_play.setText("▶ 播放")
        
        def on_slider_pressed(self):
            """滑块按下"""
            self.slider_pressed = True
        
        def on_slider_released(self):
            """滑块释放"""
            if self.player:
                position = self.progress_slider.value() / 1000.0 * self.player.duration
                self.player.seek(position)
            self.slider_pressed = False
        
        def update(self):
            """更新播放状态"""
            if self.player:
                # 更新播放
                self.player.update()
                
                # 更新进度条
                if not self.slider_pressed:
                    progress = self.player.current_position / len(self.player.audio_data)
                    self.progress_slider.setValue(int(progress * 1000))
                
                # 更新时间显示
                current_time = self.player.current_position / self.player.sr
                total_time = self.player.duration
                self.time_label.setText(
                    f"{self.format_time(current_time)} / {self.format_time(total_time)}"
                )
        
        def recognize_chord(self):
            """识别和弦"""
            if self.player and self.engine and self.player.is_playing:
                # 获取当前音频片段
                audio_segment = self.player.get_current_audio(duration=2.0)
                
                # 识别
                chord, confidence, top_5 = self.engine.predict(audio_segment, self.player.sr)
                
                # 更新显示
                self.current_chord = chord
                self.confidence = confidence
                self.chord_label.setText(chord)
                self.confidence_label.setText(f"置信度: {confidence*100:.1f}%")
                
                # 根据置信度改变颜色
                if confidence > 0.7:
                    color = "#00FF00"  # 绿色
                elif confidence > 0.4:
                    color = "#FFFF00"  # 黄色
                else:
                    color = "#FF8800"  # 橙色
                
                self.chord_label.setStyleSheet(f"QLabel {{ color: {color}; }}")
                
                # 更新 Top-5
                for i, (label, prob) in enumerate(top_5):
                    self.top5_labels[i].setText(f"{i+1}. {label}: {prob*100:.1f}%")
                
                # 记录历史
                current_time = self.player.current_position / self.player.sr
                self.chord_history.append((current_time, chord, confidence))
        
        @staticmethod
        def format_time(seconds: float) -> str:
            """格式化时间"""
            minutes = int(seconds // 60)
            seconds = int(seconds % 60)
            return f"{minutes:02d}:{seconds:02d}"
        
        def closeEvent(self, event):
            """关闭窗口"""
            if self.player:
                self.player.close()
            event.accept()


# ==================== 命令行模式 ====================

def run_cli_mode():
    """命令行模式"""
    print("\n" + "="*60)
    print("和弦识别播放器 - 命令行模式")
    print("="*60 + "\n")
    
    # 加载音频
    wav_path = input("请输入 WAV 文件路径: ").strip()
    if not os.path.exists(wav_path):
        print("文件不存在！")
        return
    
    # 加载模型
    model_path = input("请输入模型文件路径 (.pth): ").strip()
    if not os.path.exists(model_path):
        print("模型文件不存在！")
        return
    
    # 查找标签映射
    model_dir = Path(model_path).parent
    mapping_files = list(model_dir.glob("label_mappings_*.json"))
    if not mapping_files:
        print("未找到标签映射文件！")
        return
    
    mapping_path = mapping_files[0]
    
    # 初始化
    player = AudioPlayer(wav_path)
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    engine = ChordRecognitionEngine(model_path, str(mapping_path), device=device)
    
    print("\n开始播放和识别...\n")
    player.play()
    
    try:
        while player.current_position < len(player.audio_data):
            player.update()
            
            # 每 2 秒识别一次
            current_time = player.current_position / player.sr
            if int(current_time * 2) != int((current_time - 0.1) * 2):
                audio_segment = player.get_current_audio()
                chord, confidence, _ = engine.predict(audio_segment, player.sr)
                print(f"[{current_time:6.2f}s] {chord:15s} (置信度: {confidence*100:5.1f}%)")
    
    except KeyboardInterrupt:
        print("\n\n播放已停止")
    
    finally:
        player.close()


# ==================== 主函数 ====================

def main():
    if GUI_AVAILABLE:
        app = QApplication(sys.argv)
        window = ChordPlayerWindow()
        window.show()
        sys.exit(app.exec_())
    else:
        run_cli_mode()


if __name__ == '__main__':
    main()
