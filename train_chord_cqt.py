#!/usr/bin/env python3
"""
和弦识别深度学习训练脚本 - CQT 版本
使用 Constant-Q Transform 特征，更适合音乐和弦分析
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


# ==================== 标签提取 ====================

class LabelExtractor:
    """从文件名提取和弦标签"""
    
    CHORD_TYPES = {
        'maj': 'major',
        'min': 'minor',
        'dim': 'diminished',
        'aug': 'augmented',
        'sus': 'suspended',
        'dom': 'dominant7',
        'maj7': 'major7',
        'min7': 'minor7',
        'dim7': 'diminished7',
        'hdi': 'half_diminished7',
    }
    
    ROOT_NOTES = {
        'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11,
    }
    
    @classmethod
    def parse_filename(cls, filename: str) -> Dict[str, str]:
        """解析文件名提取标签"""
        name = Path(filename).stem
        parts = name.split('_')
        
        if len(parts) < 3:
            raise ValueError(f"Invalid filename format: {filename}")
        
        root = parts[0]
        chord_abbr = parts[1]
        voicing = parts[2] if len(parts) > 2 else 'unknown'
        index = parts[3] if len(parts) > 3 else '0000'
        
        chord_type = cls.CHORD_TYPES.get(chord_abbr, chord_abbr)
        
        return {
            'root': root,
            'root_idx': cls.ROOT_NOTES.get(root, 0),
            'chord_type': chord_type,
            'chord_abbr': chord_abbr,
            'voicing': voicing,
            'index': index,
            'full_label': f"{root}_{chord_type}"
        }
    
    @classmethod
    def build_label_mappings(cls, wav_files: List[str]) -> Tuple[Dict, Dict]:
        """构建标签到索引的映射"""
        roots = set()
        chords = set()
        full_labels = set()
        
        for wav_file in wav_files:
            try:
                info = cls.parse_filename(wav_file)
                roots.add(info['root'])
                chords.add(info['chord_type'])
                full_labels.add(info['full_label'])
            except Exception as e:
                print(f"Warning: Skipping {wav_file}: {e}")
        
        root_to_idx = {root: idx for idx, root in enumerate(sorted(roots))}
        chord_to_idx = {chord: idx for idx, chord in enumerate(sorted(chords))}
        full_label_to_idx = {label: idx for idx, label in enumerate(sorted(full_labels))}
        
        return {
            'root_to_idx': root_to_idx,
            'chord_to_idx': chord_to_idx,
            'full_label_to_idx': full_label_to_idx,
            'idx_to_root': {v: k for k, v in root_to_idx.items()},
            'idx_to_chord': {v: k for k, v in chord_to_idx.items()},
            'idx_to_full_label': {v: k for k, v in full_label_to_idx.items()},
        }


# ==================== 数据集 (CQT 版本) ====================

class ChordDatasetCQT(Dataset):
    """和弦识别数据集 - 使用 CQT 特征"""
    
    def __init__(
        self,
        wav_dir: str,
        label_mappings: Dict,
        target_sr: int = 22050,
        duration: float = 2.0,
        n_bins: int = 84,  # 7 octaves × 12 bins/octave
        bins_per_octave: int = 12,
        hop_length: int = 512,
        task: str = 'full'
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
            task: 任务类型
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
        print(f"CQT 参数: {n_bins} bins, {bins_per_octave} bins/octave")
    
    def __len__(self):
        return len(self.wav_files)
    
    def extract_cqt(self, y: np.ndarray) -> np.ndarray:
        """提取 CQT 特征"""
        # CQT
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
        
        return C_norm
    
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
        
        # 提取 CQT
        cqt_features = self.extract_cqt(y)
        
        # 转换为 PyTorch tensor 并添加 channel 维度
        cqt_tensor = torch.FloatTensor(cqt_features).unsqueeze(0)
        
        # 提取标签
        label_info = LabelExtractor.parse_filename(wav_path.name)
        
        if self.task == 'full':
            label = self.label_mappings['full_label_to_idx'][label_info['full_label']]
        elif self.task == 'root':
            label = self.label_mappings['root_to_idx'][label_info['root']]
        elif self.task == 'chord':
            label = self.label_mappings['chord_to_idx'][label_info['chord_type']]
        else:
            raise ValueError(f"Unknown task: {self.task}")
        
        return cqt_tensor, label


# ==================== 模型 (CQT 版本) ====================

class ChordCNN_CQT(nn.Module):
    """和弦识别 CNN 模型 - 针对 CQT 优化"""
    
    def __init__(self, num_classes: int, input_channels: int = 1, n_bins: int = 84):
        super(ChordCNN_CQT, self).__init__()
        
        # CQT 特征的频率维度是对数的，更适合音乐
        # 使用更小的 kernel 来捕捉音乐结构
        
        self.conv_layers = nn.Sequential(
            # Conv Block 1 - 捕捉单个音符
            nn.Conv2d(input_channels, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.2),
            
            # Conv Block 2 - 捕捉音程关系
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.2),
            
            # Conv Block 3 - 捕捉和弦结构
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.3),
            
            # Conv Block 4 - 高级特征
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# ==================== 训练器 ====================

class ChordTrainer:
    """训练器 - 与 Mel 版本相同"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        lr: float = 0.001,
        num_epochs: int = 50
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def train_epoch(self):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc='Validation'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, save_path: str = 'chord_model_cqt.pth'):
        """完整训练流程"""
        best_val_acc = 0
        
        print(f"\n{'='*60}")
        print(f"Starting Training (CQT Features)")
        print(f"{'='*60}\n")
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print('-' * 60)
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            self.scheduler.step(val_loss)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, save_path)
                print(f"✓ Best model saved! Val Acc: {val_acc:.2f}%")
        
        print(f"\n{'='*60}")
        print(f"Training Completed! Best Val Acc: {best_val_acc:.2f}%")
        print(f"{'='*60}\n")
        
        return best_val_acc
    
    def plot_history(self, save_path: str = 'training_history_cqt.png'):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss (CQT)')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(self.train_accs, label='Train Acc')
        ax2.plot(self.val_accs, label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy (CQT)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description='和弦识别深度学习训练 (CQT)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='WAV 文件目录')
    parser.add_argument('--task', type=str, default='full',
                        choices=['full', 'root', 'chord'],
                        help='任务类型')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='验证集比例')
    parser.add_argument('--n_bins', type=int, default=84,
                        help='CQT bins 数量 (推荐: 84 = 7个八度)')
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
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    print(f"特征类型: CQT (Constant-Q Transform)")
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
    print("步骤 2: 创建数据集 (CQT 特征)")
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 3. 创建模型
    print("\n" + "="*60)
    print("步骤 3: 创建模型 (CQT-CNN)")
    print("="*60)
    
    model = ChordCNN_CQT(num_classes=num_classes, n_bins=args.n_bins)
    
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
    model_path = output_dir / f'chord_model_cqt_{args.task}_{timestamp}.pth'
    best_acc = trainer.train(save_path=str(model_path))
    
    # 5. 保存训练历史
    history_path = output_dir / f'training_history_cqt_{args.task}_{timestamp}.png'
    trainer.plot_history(save_path=str(history_path))
    
    # 6. 保存标签映射
    import json
    mapping_path = output_dir / f'label_mappings_cqt_{args.task}_{timestamp}.json'
    with open(mapping_path, 'w') as f:
        json.dump({
            'task': args.task,
            'num_classes': num_classes,
            'feature_type': 'CQT',
            'n_bins': args.n_bins,
            'bins_per_octave': args.bins_per_octave,
            'mappings': {
                k: v for k, v in label_mappings.items()
                if not k.startswith('idx_to_')
            }
        }, f, indent=2)
    
    print(f"\n✓ 标签映射保存到: {mapping_path}")
    print(f"✓ 模型保存到: {model_path}")
    print(f"✓ 训练历史保存到: {history_path}")
    print(f"\n最佳验证准确率: {best_acc:.2f}%")
    
    print("\n" + "="*60)
    print("CQT vs Mel-spectrogram 对比:")
    print("="*60)
    print("CQT 优势:")
    print("  ✅ 对数频率分辨率（符合音乐感知）")
    print("  ✅ 更好的低频分辨率（适合贝斯音）")
    print("  ✅ 直接对应音乐音高")
    print("  ✅ 对移调更鲁棒")
    print("\nMel 优势:")
    print("  ✅ 计算更快")
    print("  ✅ 预训练模型更多")
    print("  ✅ 内存占用更小")
    print("="*60)


if __name__ == '__main__':
    main()
