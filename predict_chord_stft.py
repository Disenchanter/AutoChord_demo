#!/usr/bin/env python3
"""
和弦识别推理脚本（使用原始STFT频谱）
使用训练好的模型对新的 WAV 文件进行预测
"""

import argparse
import json
from pathlib import Path
import torch
import torchaudio
import torchaudio.transforms as T
from train_chord_stft import ChordCNN, LabelExtractor


def load_model(model_path: str, num_classes: int, device: str = 'cuda'):
    """加载训练好的模型"""
    model = ChordCNN(num_classes=num_classes)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def preprocess_audio(
    wav_path: str,
    target_sr: int = 22050,
    duration: float = 2.0,
    n_fft: int = 2048,
    hop_length: int = 512
):
    """预处理音频文件 - 使用原始STFT频谱"""
    # 加载音频
    waveform, sr = torchaudio.load(wav_path)
    
    # 转换为单声道
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # 重采样
    if sr != target_sr:
        resampler = T.Resample(sr, target_sr)
        waveform = resampler(waveform)
    
    # 裁剪或填充
    target_length = int(target_sr * duration)
    if waveform.shape[1] < target_length:
        padding = target_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    else:
        waveform = waveform[:, :target_length]
    
    # 原始 STFT 频谱
    spectrogram_transform = T.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
    )
    amplitude_to_db = T.AmplitudeToDB()
    
    spec = spectrogram_transform(waveform)
    spec_db = amplitude_to_db(spec)
    
    # 归一化
    spec_db = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min() + 1e-8)
    
    return spec_db.unsqueeze(0)  # 添加 batch 维度


def predict(model, input_tensor, device: str = 'cuda'):
    """预测"""
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_idx = probabilities.max(1)
    
    return predicted_idx.item(), confidence.item()


def main():
    parser = argparse.ArgumentParser(description='和弦识别推理')
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
    mappings = mapping_data['mappings']
    
    # 构建反向映射
    if task == 'full':
        idx_to_label = {v: k for k, v in mappings['full_label_to_idx'].items()}
    elif task == 'root':
        idx_to_label = {v: k for k, v in mappings['root_to_idx'].items()}
    else:  # chord
        idx_to_label = {v: k for k, v in mappings['chord_to_idx'].items()}
    
    # 加载模型
    print(f"加载模型: {args.model}")
    model = load_model(args.model, num_classes, device)
    
    # 预处理音频
    print(f"处理音频: {args.wav_file}")
    input_tensor = preprocess_audio(args.wav_file)
    
    # 预测
    print("预测中...")
    predicted_idx, confidence = predict(model, input_tensor, device)
    predicted_label = idx_to_label[predicted_idx]
    
    # 提取真实标签（如果文件名符合格式）
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
    print("预测结果")
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
