#!/usr/bin/env python3
"""
和弦识别推理脚本 - CQT 版本
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import librosa
# 复用 STFT 版本的加载与预测逻辑，只替换特征提取为 CQT
from predict_chord_stft import load_model, predict
from train_chord_stft import LabelExtractor



def preprocess_audio_cqt(
    wav_path: str,
    target_sr: int = 22050,
    duration: float = 2.0,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    hop_length: int = 512
):
    """预处理音频文件 - CQT"""
    # 加载音频
    y, sr = librosa.load(wav_path, sr=target_sr, mono=True)
    
    # 裁剪或填充
    target_length = int(target_sr * duration)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]
    
    # CQT
    C = librosa.cqt(
        y,
        sr=target_sr,
        hop_length=hop_length,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave
    )
    
    # 转换为 dB
    C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)
    
    # 归一化
    C_norm = (C_db - C_db.min()) / (C_db.max() - C_db.min() + 1e-8)
    
    # 转换为 tensor
    cqt_tensor = torch.FloatTensor(C_norm).unsqueeze(0).unsqueeze(0)
    
    return cqt_tensor


# 使用从 predict_chord 导入的 `load_model` 和 `predict`


def main():
    parser = argparse.ArgumentParser(description='和弦识别推理 (CQT)')
    parser.add_argument('--wav_file', type=str, required=True,
                        help='WAV 文件路径')
    parser.add_argument('--model', type=str, required=True,
                        help='模型文件路径 (.pth)')
    parser.add_argument('--mappings', type=str, required=True,
                        help='标签映射文件 (.json)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备: cuda 或 cpu')
    
    args = parser.parse_args()
    
    # 检测设备
    device = args.device if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # 加载标签映射
    with open(args.mappings, 'r') as f:
        mapping_data = json.load(f)
    
    task = mapping_data['task']
    num_classes = mapping_data['num_classes']
    n_bins = mapping_data.get('n_bins', 84)
    bins_per_octave = mapping_data.get('bins_per_octave', 12)
    mappings = mapping_data['mappings']
    
    # 构建反向映射
    if task == 'full':
        idx_to_label = {v: k for k, v in mappings['full_label_to_idx'].items()}
    elif task == 'root':
        idx_to_label = {v: k for k, v in mappings['root_to_idx'].items()}
    else:
        idx_to_label = {v: k for k, v in mappings['chord_to_idx'].items()}
    
    # 加载模型（复用 STFT 版本的 load_model）
    print(f"加载模型: {args.model}")
    print(f"特征类型: CQT ({n_bins} bins, {bins_per_octave} bins/octave)")
    model = load_model(args.model, num_classes, device)
    
    # 预处理音频
    print(f"处理音频: {args.wav_file}")
    input_tensor = preprocess_audio_cqt(
        args.wav_file,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave
    )
    
    # 预测
    print("预测中...")
    predicted_idx, confidence = predict(model, input_tensor, device)
    predicted_label = idx_to_label[predicted_idx]
    
    # 提取真实标签
    try:
        true_label_info = LabelExtractor.parse_filename(args.wav_file)
        if task == 'full':
            true_label = true_label_info['full_label']
        elif task == 'root':
            true_label = true_label_info['root']
        else:
            true_label = true_label_info['chord_type']
        
        is_correct = (predicted_label == true_label)
    except:
        true_label = 'Unknown'
        is_correct = None
    
    # 打印结果
    print("\n" + "="*60)
    print("预测结果 (CQT Features)")
    print("="*60)
    print(f"文件: {Path(args.wav_file).name}")
    print(f"真实标签: {true_label}")
    print(f"预测标签: {predicted_label}")
    print(f"置信度: {confidence*100:.2f}%")
    
    if is_correct is not None:
        status = "✓ 正确" if is_correct else "✗ 错误"
        print(f"结果: {status}")
    
    print("="*60)


if __name__ == '__main__':
    main()
