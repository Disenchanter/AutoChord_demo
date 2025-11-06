#!/usr/bin/env python3
"""
和弦识别深度学习训练脚本 - CQT 版本
使用 Constant-Q Transform 特征，更适合音乐和弦分析

⚠️ 重要：音乐识别应使用 CQT 或原始 STFT 频谱
- CQT: 对数频率，最适合音乐，保留完整音高信息 ✅✅✅
- 原始STFT: 1025 频率bins，线性尺度，保留完整音高信息 ✅
- Mel频谱: 压缩高频，丢失音高细节，适合语音而非音乐 ❌
"""

import os
import re
import argparse
from pathlib import Path
from typing import Tuple, List, Dict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==================== 导入 STFT 版本的共用类 ====================
from train_chord_stft import LabelExtractor, ChordCNN, ChordTrainer


# ==================== 数据集 (CQT 特征版本) ====================

class ChordDatasetCQT(Dataset):
    """
    和弦识别数据集 - 使用 CQT 频谱（适合音乐）
    
    与 ChordDataset (STFT版本) 的区别：
    - 使用 librosa.cqt() 替代 torchaudio.Spectrogram()
    - 输出特征形状: (1, 84, T) vs (1, 1025, T)
    - 对数频率分布 vs 线性频率分布
    """
    
    def __init__(
        self,
        wav_dir: str,
        label_mappings: Dict,
        target_sr: int = 22050,
        duration: float = 2.0,
        n_bins: int = 84,  # 7 octaves × 12 bins/octave
        bins_per_octave: int = 12,
        hop_length: int = 512,
        task: str = 'full'  # 'full', 'root', 'chord'
    ):
        """
        Args:
            wav_dir: WAV 文件目录
            label_mappings: 标签映射字典
            target_sr: 目标采样率
            duration: 音频时长（秒）
            n_bins: CQT 频率bin数（建议 84 = 7个八度）
            bins_per_octave: 每个八度的bin数（12=半音分辨率）
            hop_length: 跳跃长度
            task: 任务类型 - 'full'(完整标签), 'root'(仅根音), 'chord'(仅和弦类型)
        """
        self.wav_dir = Path(wav_dir)
        self.label_mappings = label_mappings
        self.target_sr = target_sr
        self.duration = duration
        self.task = task
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.hop_length = hop_length
        
        # 收集所有 WAV 文件
        self.wav_files = sorted(list(self.wav_dir.glob('*.wav')))
        
        if len(self.wav_files) == 0:
            raise ValueError(f"No WAV files found in {wav_dir}")
        
        print(f"Found {len(self.wav_files)} WAV files")
        print(f"使用 CQT 频谱: {n_bins} 频率bins (对数尺度，{bins_per_octave} bins/octave)")
    
    def __len__(self):
        return len(self.wav_files)
    
    def __getitem__(self, idx):
        wav_path = self.wav_files[idx]
        
        # 加载音频（librosa 自动转单声道）
        y, sr = librosa.load(wav_path, sr=self.target_sr, mono=True)
        
        # 裁剪或填充到固定长度
        target_length = int(self.target_sr * self.duration)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        else:
            y = y[:target_length]
        
        # 提取 CQT (Constant-Q Transform)
        C = librosa.cqt(
            y,
            sr=self.target_sr,
            hop_length=self.hop_length,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave
        )
        
        # 转换为 dB
        C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)
        
        # 归一化到 [0, 1]
        C_norm = (C_db - C_db.min()) / (C_db.max() - C_db.min() + 1e-8)
        
        # 转换为 PyTorch tensor 并添加 channel 维度
        cqt_tensor = torch.FloatTensor(C_norm).unsqueeze(0)
        
        # 提取标签
        label_info = LabelExtractor.parse_filename(wav_path.name)
        
        # 根据任务类型选择标签
        if self.task == 'full':
            label = self.label_mappings['full_label_to_idx'][label_info['full_label']]
        elif self.task == 'root':
            label = self.label_mappings['root_to_idx'][label_info['root']]
        elif self.task == 'chord':
            label = self.label_mappings['chord_to_idx'][label_info['chord_type']]
        else:
            raise ValueError(f"Unknown task: {self.task}")
        
        return cqt_tensor, label


# ==================== 主函数 ====================
# 注意：ChordCNN 和 ChordTrainer 已从 train_chord_recognition.py 导入

def main():
    parser = argparse.ArgumentParser(description='和弦识别深度学习训练（使用CQT频谱）')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='WAV 文件目录')
    parser.add_argument('--task', type=str, default='full',
                        choices=['full', 'root', 'chord'],
                        help='任务类型: full(完整), root(根音), chord(和弦类型)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='验证集比例')
    parser.add_argument('--n_bins', type=int, default=84,
                        help='CQT bins数量（推荐: 84 = 7个八度）')
    parser.add_argument('--bins_per_octave', type=int, default=12,
                        help='每个八度的bins (12=半音分辨率)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备: cuda 或 cpu')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 检测设备
    device = args.device if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    print(f"特征类型: CQT (Constant-Q Transform - 对数频率，更适合音乐)")
    print(f"CQT 配置: {args.n_bins} bins, {args.bins_per_octave} bins/octave")
    
    # 1. 构建标签映射
    print("\n" + "="*60)
    print("步骤 1: 构建标签映射")
    print("="*60)
    
    wav_files = list(Path(args.data_dir).glob('*.wav'))
    label_mappings = LabelExtractor.build_label_mappings(
        [f.name for f in wav_files]
    )
    
    if args.task == 'full':
        num_classes = len(label_mappings['full_label_to_idx'])
        print(f"完整标签数: {num_classes}")
    elif args.task == 'root':
        num_classes = len(label_mappings['root_to_idx'])
        print(f"根音类别数: {num_classes}")
    else:
        num_classes = len(label_mappings['chord_to_idx'])
        print(f"和弦类型数: {num_classes}")
    
    # 2. 创建数据集
    print("\n" + "="*60)
    print("步骤 2: 创建数据集")
    print("="*60)
    
    dataset = ChordDatasetCQT(
        wav_dir=args.data_dir,
        label_mappings=label_mappings,
        n_bins=args.n_bins,
        bins_per_octave=args.bins_per_octave,
        task=args.task
    )
    
    # 划分训练集和验证集
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"训练集: {train_size} 样本")
    print(f"验证集: {val_size} 样本")
    
    # 创建 DataLoader
    # 注意：MPS 不支持 pin_memory，多进程可能导致 torchaudio 加载问题
    # 设置 num_workers=0 使用单进程加载
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # 使用单进程避免 torchcodec 错误
        pin_memory=False  # MPS 不支持 pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # 使用单进程避免 torchcodec 错误
        pin_memory=False  # MPS 不支持 pin_memory
    )
    
    # 3. 创建模型
    print("\n" + "="*60)
    print("步骤 3: 创建模型")
    print("="*60)
    
    model = ChordCNN(num_classes=num_classes)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 4. 训练
    print("\n" + "="*60)
    print("步骤 4: 开始训练")
    print("="*60)
    
    trainer = ChordTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        num_epochs=args.epochs
    )
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 训练
    model_path = output_dir / f'chord_model_{args.task}_{timestamp}.pth'
    best_acc = trainer.train(save_path=str(model_path))
    
    # 5. 保存训练历史
    history_path = output_dir / f'training_history_{args.task}_{timestamp}.png'
    trainer.plot_history(save_path=str(history_path), task=args.task)
    
    # 6. 保存标签映射
    import json
    mapping_path = output_dir / f'label_mappings_{args.task}_{timestamp}.json'
    with open(mapping_path, 'w') as f:
        json.dump({
            'task': args.task,
            'num_classes': num_classes,
            'feature_type': 'CQT',
            'n_bins': args.n_bins,
            'bins_per_octave': args.bins_per_octave,
            'mappings': {
                k: v for k, v in label_mappings.items()
                if not k.startswith('idx_to_')  # 只保存正向映射
            }
        }, f, indent=2)
    
    print(f"\n✓ 标签映射保存到: {mapping_path}")
    print(f"✓ 模型保存到: {model_path}")
    print(f"✓ 训练历史保存到: {history_path}")
    print(f"\n最佳验证准确率: {best_acc:.2f}%")


if __name__ == '__main__':
    main()
