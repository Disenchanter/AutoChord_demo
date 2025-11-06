#!/usr/bin/env python3
"""
批量测试脚本
在测试集上评估模型性能，生成详细报告
"""

import argparse
import json
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from train_chord_stft import ChordCNN, ChordDataset, LabelExtractor
from train_chord_cqt import ChordDatasetCQT


def evaluate_model(model, test_loader, device, idx_to_label):
    """评估模型"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_confidences = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidences, predictions = probabilities.max(1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_confidences)


def plot_confusion_matrix(y_true, y_pred, labels, save_path, title='Confusion Matrix'):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    # 计算百分比
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    n = len(labels)
    # 每个类别分配0.7英寸，最小12，最大40
    size = min(max(n * 0.7, 12), 40)
    plt.figure(figsize=(size, size * 0.8))
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt='.1f',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Percentage (%)'}
    )
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵保存到: {save_path}")


def plot_confidence_distribution(confidences, predictions, labels, correct_mask, save_path):
    """绘制置信度分布"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 正确 vs 错误预测的置信度
    correct_conf = confidences[correct_mask]
    wrong_conf = confidences[~correct_mask]
    
    ax1.hist(correct_conf, bins=50, alpha=0.7, label='Correct', color='green')
    ax1.hist(wrong_conf, bins=50, alpha=0.7, label='Wrong', color='red')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Count')
    ax1.set_title('Confidence Distribution: Correct vs Wrong')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 每个类别的平均置信度
    unique_labels = np.unique(predictions)
    avg_confidences = [confidences[predictions == label].mean() for label in unique_labels]
    label_names = [labels[label] for label in unique_labels]
    
    ax2.bar(range(len(unique_labels)), avg_confidences)
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Average Confidence')
    ax2.set_title('Average Confidence by Class')
    ax2.set_xticks(range(len(unique_labels)))
    ax2.set_xticklabels(label_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"置信度分布图保存到: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='批量测试和弦识别模型（支持 STFT 和 CQT）')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='测试数据目录')
    parser.add_argument('--model', type=str, required=True,
                        help='模型文件路径')
    parser.add_argument('--mappings', type=str, required=True,
                        help='标签映射文件')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备: cuda, mps 或 cpu')
    parser.add_argument('--output_dir', type=str, default='test_results'+'_undefined',
                        help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 检测设备
    if args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"使用设备: {device}\n")
    
    # 加载标签映射
    with open(args.mappings, 'r') as f:
        mapping_data = json.load(f)
    
    task = mapping_data['task']
    num_classes = mapping_data['num_classes']
    label_mappings = mapping_data['mappings']
    
    # 检测模型类型（STFT 或 CQT）
    model_type = 'cqt' if 'n_bins' in mapping_data else 'stft'
    n_bins = mapping_data.get('n_bins', 84)
    bins_per_octave = mapping_data.get('bins_per_octave', 12)
    
    print(f"模型类型: {model_type.upper()}")
    if model_type == 'cqt':
        print(f"CQT 参数: n_bins={n_bins}, bins_per_octave={bins_per_octave}")
    print()
    
    # 构建完整的映射（包括反向映射，直接用字符串标签，不做任何映射）
    full_mappings = label_mappings.copy()
    if task == 'full':
        idx_to_label = {v: k for k, v in label_mappings['full_label_to_idx'].items()}
    elif task == 'root':
        idx_to_label = {v: k for k, v in label_mappings['root_to_idx'].items()}
    else:
        idx_to_label = {v: k for k, v in label_mappings['chord_to_idx'].items()}
    
    print(f"任务类型: {task}")
    print(f"类别数: {num_classes}\n")
    
    # 创建数据集（根据模型类型选择）
    print("加载测试数据...")
    if model_type == 'cqt':
        dataset = ChordDatasetCQT(
            wav_dir=args.data_dir,
            label_mappings=full_mappings,
            task=task,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave
        )
    else:
        dataset = ChordDataset(
            wav_dir=args.data_dir,
            label_mappings=full_mappings,
            task=task
        )
    
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    print(f"测试样本数: {len(dataset)}\n")
    
    # 加载模型
    print("加载模型...")
    model = ChordCNN(num_classes=num_classes)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"模型训练轮数: {checkpoint['epoch'] + 1}")
    print(f"验证准确率: {checkpoint['val_acc']:.2f}%\n")
    
    # 评估
    print("="*60)
    print("开始评估")
    print("="*60 + "\n")
    
    predictions, true_labels, confidences = evaluate_model(
        model, test_loader, device, idx_to_label
    )
    
    # 计算准确率
    accuracy = (predictions == true_labels).mean() * 100
    correct_mask = predictions == true_labels
    
    print(f"\n总体准确率: {accuracy:.2f}%")
    print(f"平均置信度: {confidences.mean()*100:.2f}%")
    print(f"正确预测平均置信度: {confidences[correct_mask].mean()*100:.2f}%")
    print(f"错误预测平均置信度: {confidences[~correct_mask].mean()*100:.2f}%\n")
    
    # 分类报告
    label_names = [idx_to_label[i] for i in range(num_classes)]
    report = classification_report(
        true_labels,
        predictions,
        target_names=label_names,
        digits=4
    )
    
    print("="*60)
    print("详细分类报告")
    print("="*60)
    print(report)
    
    # 保存报告
    report_path = output_dir / f'classification_report_{task}.txt'
    with open(report_path, 'w') as f:
        f.write(f"Task: {task}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Test Accuracy: {accuracy:.2f}%\n\n")
        f.write(report)
    
    print(f"\n报告保存到: {report_path}")
    
    # 绘制混淆矩阵
    cm_path = output_dir / f'confusion_matrix_{task}.png'
    plot_confusion_matrix(
        true_labels,
        predictions,
        label_names,
        cm_path,
        title=f'Confusion Matrix - {task.capitalize()} Task'
    )
    
    # 绘制置信度分布
    conf_path = output_dir / f'confidence_distribution_{task}.png'
    plot_confidence_distribution(
        confidences,
        predictions,
        label_names,
        correct_mask,
        conf_path
    )
    
    # 保存错误样本
    print("\n分析错误样本...")
    wav_files = sorted(list(Path(args.data_dir).glob('*.wav')))
    errors = []
    
    for i, (pred, true, conf) in enumerate(zip(predictions, true_labels, confidences)):
        if pred != true:
            errors.append({
                'file': wav_files[i].name,
                'true_label': idx_to_label[true],
                'predicted_label': idx_to_label[pred],
                'confidence': f"{conf*100:.2f}%"
            })
    
    if errors:
        error_path = output_dir / f'error_samples_{task}.json'
        with open(error_path, 'w') as f:
            json.dump(errors, f, indent=2)
        
        print(f"错误样本数: {len(errors)}")
        print(f"错误样本保存到: {error_path}")
        
        # 打印前 10 个错误
        print("\n前 10 个错误样本:")
        print("-" * 80)
        for error in errors[:10]:
            print(f"文件: {error['file']}")
            print(f"  真实: {error['true_label']} | 预测: {error['predicted_label']} | 置信度: {error['confidence']}")
    else:
        print("🎉 完美！没有错误样本！")
    
    print("\n" + "="*60)
    print(f"测试完成！结果保存在: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    # 默认调用最新模型和标签映射
    import glob
    import os
    import sys
    
    # 优先查找 CQT 模型，如果没有则使用 STFT 模型
    cqt_model_dir = 'models_full_cqt'
    stft_model_dir = 'models_full_stft'
    
    # 查找 CQT 模型
    cqt_model_files = sorted(glob.glob(os.path.join(cqt_model_dir, 'chord_model_full_*.pth'))) if os.path.exists(cqt_model_dir) else []
    cqt_mapping_files = sorted(glob.glob(os.path.join(cqt_model_dir, 'label_mappings_full_*.json'))) if os.path.exists(cqt_model_dir) else []
    
    # 查找 STFT 模型
    stft_model_files = sorted(glob.glob(os.path.join(stft_model_dir, 'chord_model_full_*.pth'))) if os.path.exists(stft_model_dir) else []
    stft_mapping_files = sorted(glob.glob(os.path.join(stft_model_dir, 'label_mappings_full_*.json'))) if os.path.exists(stft_model_dir) else []
    
    if cqt_model_files and cqt_mapping_files:
        # 使用 CQT 模型
        latest_model = cqt_model_files[-1]
        latest_mapping = cqt_mapping_files[-1]
        output_dir = 'test_results_cqt'
        print(f"找到 CQT 模型: {latest_model}")
    elif stft_model_files and stft_mapping_files:
        # 使用 STFT 模型
        latest_model = stft_model_files[-1]
        latest_mapping = stft_mapping_files[-1]
        output_dir = 'test_results_stft'
        print(f"找到 STFT 模型: {latest_model}")
    else:
        print("错误: 未找到可用的模型文件")
        sys.exit(1)
    
    if len(sys.argv) == 1:  # 如果没有命令行参数，使用默认值
        sys.argv += [
            '--data_dir', 'single_chords_output',
            '--model', latest_model,
            '--mappings', latest_mapping,
            '--output_dir', output_dir,
            '--device', 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
        ]
    main()
