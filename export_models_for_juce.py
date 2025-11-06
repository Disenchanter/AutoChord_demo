#!/usr/bin/env python3
"""
导出PyTorch模型为TorchScript格式
用于JUCE插件加载
"""

import torch
import argparse
from pathlib import Path
import json
from train_chord_stft import ChordCNN

def export_model(model_path, output_path, num_classes, input_shape=(1, 1, 1025, 173)):
    """
    导出模型为TorchScript格式
    
    Args:
        model_path: 原始.pth模型路径
        output_path: 输出.pt路径
        num_classes: 类别数
        input_shape: 输入张量形状 [batch, channel, freq, time]
                    freq=1025 (FFT_SIZE/2 + 1，原始STFT频谱bins数)
                    time=173 (约2秒音频的帧数)
    """
    print(f"加载模型: {model_path}")
    
    # 加载模型
    model = ChordCNN(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"模型训练轮数: {checkpoint['epoch'] + 1}")
    print(f"验证准确率: {checkpoint['val_acc']:.2f}%")
    
    # 创建示例输入
    example_input = torch.randn(input_shape)
    
    # 导出为TorchScript
    print(f"导出TorchScript模型...")
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save(output_path)
    
    print(f"✓ 模型导出成功: {output_path}")
    print(f"  输入形状: {input_shape}")
    print(f"  类别数: {num_classes}")
    print()


def main():
    parser = argparse.ArgumentParser(description='导出PyTorch模型为TorchScript')
    parser.add_argument('--export_all', action='store_true',
                        help='导出所有模型（root, chord, full）')
    parser.add_argument('--output_dir', type=str, default='exported_models',
                        help='导出模型的输出目录（默认: exported_models）')
    
    args = parser.parse_args()
    
    if args.export_all:
        print("="*60)
        print("导出所有模型为TorchScript格式")
        print("="*60 + "\n")
        
        # 创建输出目录
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        print(f"输出目录: {output_dir.absolute()}\n")
        
        # 导出Root模型
        root_dir = Path('models_root_stft')
        if root_dir.exists():
            models = sorted(list(root_dir.glob('chord_model_root_*.pth')))
            if models:
                latest_model = models[-1]
                output_path = output_dir / 'root_model.pt'
                export_model(latest_model, output_path, num_classes=7)
            else:
                print(f"⚠ 未找到Root模型文件: {root_dir}/chord_model_root_*.pth\n")
        else:
            print(f"⚠ Root模型目录不存在: {root_dir}\n")
        
        # 导出Chord模型
        chord_dir = Path('models_chord_stft')
        if chord_dir.exists():
            models = sorted(list(chord_dir.glob('chord_model_chord_*.pth')))
            if models:
                latest_model = models[-1]
                output_path = output_dir / 'chord_model.pt'
                export_model(latest_model, output_path, num_classes=11)
            else:
                print(f"⚠ 未找到Chord模型文件: {chord_dir}/chord_model_chord_*.pth\n")
        else:
            print(f"⚠ Chord模型目录不存在: {chord_dir}\n")
        
        # 导出Full模型
        full_dir = Path('models_full_stft')
        if full_dir.exists():
            models = sorted(list(full_dir.glob('chord_model_full_*.pth')))
            if models:
                latest_model = models[-1]
                output_path = output_dir / 'full_model.pt'
                export_model(latest_model, output_path, num_classes=77)
            else:
                print(f"⚠ 未找到Full模型文件: {full_dir}/chord_model_full_*.pth\n")
        else:
            print(f"⚠ Full模型目录不存在: {full_dir}\n")
        
        print("="*60)
        print(f"所有模型已导出到: {output_dir.absolute()}")
        print("="*60)
    else:
        print("使用 --export_all 导出所有模型")


if __name__ == '__main__':
    main()
