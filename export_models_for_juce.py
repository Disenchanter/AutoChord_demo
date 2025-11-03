#!/usr/bin/env python3
"""
导出PyTorch模型为TorchScript格式
用于JUCE插件加载
"""

import torch
import argparse
from pathlib import Path
import json
from train_chord_recognition import ChordCNN

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
    
    args = parser.parse_args()
    
    if args.export_all:
        print("="*60)
        print("导出所有模型为TorchScript格式")
        print("="*60 + "\n")
        
        # 导出Root模型
        root_dir = Path('models_root_stft')
        if root_dir.exists():
            models = sorted(list(root_dir.glob('chord_model_root_*.pth')))
            if models:
                latest_model = models[-1]
                output_path = root_dir / 'root_model.pt'
                export_model(latest_model, output_path, num_classes=7)
        
        # 导出Chord模型
        chord_dir = Path('models_chord_stft')
        if chord_dir.exists():
            models = sorted(list(chord_dir.glob('chord_model_chord_*.pth')))
            if models:
                latest_model = models[-1]
                output_path = chord_dir / 'chord_model.pt'
                export_model(latest_model, output_path, num_classes=11)
        
        # 导出Full模型
        full_dir = Path('models_full_stft')
        if full_dir.exists():
            models = sorted(list(full_dir.glob('chord_model_full_*.pth')))
            if models:
                latest_model = models[-1]
                output_path = full_dir / 'full_model.pt'
                export_model(latest_model, output_path, num_classes=77)
        
        print("="*60)
        print("所有模型导出完成！")
        print("="*60)
        print("\n现在可以编译JUCE插件:")
        print("  cd AutoChordPlugin/build")
        print("  cmake ..")
        print("  cmake --build . --config Release")
    else:
        print("使用 --export_all 导出所有模型")


if __name__ == '__main__':
    main()
