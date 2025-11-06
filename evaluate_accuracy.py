#!/usr/bin/env python3
"""
和弦识别准确度评估脚本
支持多维度准确度计算,包括不同错误严重程度分类

错误分类:
1. 完全正确 - 根音和类型都正确
2. 类型混淆(可接受) - 根音正确,major↔maj7 或 minor↔min7
3. 高重叠根音错误(可接受) - 根音错误但音符重叠度≥67%
4. 中重叠根音错误 - 音符重叠度33-66%
5. 低重叠根音错误(严重) - 音符重叠度<33%
6. 类型混淆(其他) - 根音正确但其他类型混淆
"""

import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple
import json

# ==================== 和弦音符定义 ====================

CHORD_NOTES = {
    # C系列
    "C_major": ["C", "E", "G"],
    "C_min": ["C", "Eb", "G"],
    "C_maj7": ["C", "E", "G", "B"],
    "C_min7": ["C", "Eb", "G", "Bb"],
    "C_dom7": ["C", "E", "G", "Bb"],
    "C_dim7": ["C", "Eb", "Gb", "A"],
    "C_hdim7": ["C", "Eb", "Gb", "Bb"],
    "C_aug": ["C", "E", "G#"],
    "C_sus4": ["C", "F", "G"],
    "C_sus2": ["C", "D", "G"],
    "C_dim": ["C", "Eb", "Gb"],
    
    # D系列
    "D_major": ["D", "F#", "A"],
    "D_min": ["D", "F", "A"],
    "D_maj7": ["D", "F#", "A", "C#"],
    "D_min7": ["D", "F", "A", "C"],
    "D_dom7": ["D", "F#", "A", "C"],
    "D_dim7": ["D", "F", "Ab", "B"],
    "D_hdim7": ["D", "F", "Ab", "C"],
    "D_aug": ["D", "F#", "A#"],
    "D_sus4": ["D", "G", "A"],
    "D_sus2": ["D", "E", "A"],
    "D_dim": ["D", "F", "Ab"],
    
    # E系列
    "E_major": ["E", "G#", "B"],
    "E_min": ["E", "G", "B"],
    "E_maj7": ["E", "G#", "B", "D#"],
    "E_min7": ["E", "G", "B", "D"],
    "E_dom7": ["E", "G#", "B", "D"],
    "E_dim7": ["E", "G", "Bb", "C#"],
    "E_hdim7": ["E", "G", "Bb", "D"],
    "E_aug": ["E", "G#", "B#"],
    "E_sus4": ["E", "A", "B"],
    "E_sus2": ["E", "F#", "B"],
    "E_dim": ["E", "G", "Bb"],
    
    # F系列
    "F_major": ["F", "A", "C"],
    "F_min": ["F", "Ab", "C"],
    "F_maj7": ["F", "A", "C", "E"],
    "F_min7": ["F", "Ab", "C", "Eb"],
    "F_dom7": ["F", "A", "C", "Eb"],
    "F_dim7": ["F", "Ab", "B", "D"],
    "F_hdim7": ["F", "Ab", "B", "Eb"],
    "F_aug": ["F", "A", "C#"],
    "F_sus4": ["F", "Bb", "C"],
    "F_sus2": ["F", "G", "C"],
    "F_dim": ["F", "Ab", "B"],
    
    # G系列
    "G_major": ["G", "B", "D"],
    "G_min": ["G", "Bb", "D"],
    "G_maj7": ["G", "B", "D", "F#"],
    "G_min7": ["G", "Bb", "D", "F"],
    "G_dom7": ["G", "B", "D", "F"],
    "G_dim7": ["G", "Bb", "Db", "E"],
    "G_hdim7": ["G", "Bb", "Db", "F"],
    "G_aug": ["G", "B", "D#"],
    "G_sus4": ["G", "C", "D"],
    "G_sus2": ["G", "A", "D"],
    "G_dim": ["G", "Bb", "Db"],
    
    # A系列
    "A_major": ["A", "C#", "E"],
    "A_min": ["A", "C", "E"],
    "A_maj7": ["A", "C#", "E", "G#"],
    "A_min7": ["A", "C", "E", "G"],
    "A_dom7": ["A", "C#", "E", "G"],
    "A_dim7": ["A", "C", "Eb", "Gb"],
    "A_hdim7": ["A", "C", "Eb", "G"],
    "A_aug": ["A", "C#", "E#"],
    "A_sus4": ["A", "D", "E"],
    "A_sus2": ["A", "B", "E"],
    "A_dim": ["A", "C", "Eb"],
    
    # B系列
    "B_major": ["B", "D#", "F#"],
    "B_min": ["B", "D", "F#"],
    "B_maj7": ["B", "D#", "F#", "A#"],
    "B_min7": ["B", "D", "F#", "A"],
    "B_dom7": ["B", "D#", "F#", "A"],
    "B_dim7": ["B", "D", "F", "Ab"],
    "B_hdim7": ["B", "D", "F", "A"],
    "B_aug": ["B", "D#", "F##"],
    "B_sus4": ["B", "E", "F#"],
    "B_sus2": ["B", "C#", "F#"],
    "B_dim": ["B", "D", "F"],
}

# ==================== 辅助函数 ====================

def calculate_overlap(exp_chord: str, pred_chord: str) -> Tuple[int, int, float]:
    """计算音符重叠度"""
    exp_notes = set(CHORD_NOTES.get(exp_chord, []))
    pred_notes = set(CHORD_NOTES.get(pred_chord, []))
    
    if not exp_notes or not pred_notes:
        return 0, 0, 0.0
    
    overlap = exp_notes & pred_notes
    overlap_ratio = len(overlap) / len(exp_notes)
    
    return len(overlap), len(exp_notes), overlap_ratio


def classify_error(expected: str, predicted: str) -> Tuple[str, float]:
    """
    分类错误类型
    返回: (错误类型, 得分)
    """
    if expected == predicted:
        return "完全正确", 1.0
    
    # 规范化命名
    expected = expected.replace("_minor", "_min")
    predicted = predicted.replace("_minor", "_min")
    
    exp_root = expected.split('_')[0]
    pred_root = predicted.split('_')[0]
    exp_type = expected.split('_', 1)[1] if '_' in expected else ""
    pred_type = predicted.split('_', 1)[1] if '_' in predicted else ""
    
    # 类型1: 根音正确,类型混淆
    if exp_root == pred_root:
        # major ↔ maj7 或 minor ↔ min7 (可接受的类型混淆)
        if (exp_type in ["major", "maj7"] and pred_type in ["major", "maj7"]) or \
           (exp_type in ["min", "min7"] and pred_type in ["min", "min7"]):
            return "类型混淆(可接受)", 0.9
        else:
            # 其他类型混淆
            return "类型混淆(其他)", 0.6
    
    # 计算音符重叠度
    overlap_count, exp_count, overlap_ratio = calculate_overlap(expected, predicted)
    
    # 类型2: 根音错误,按重叠度分类
    if overlap_ratio >= 0.67:  # 至少2/3音符相同
        return "高重叠根音错误(可接受)", 0.7
    elif overlap_ratio >= 0.33:
        return "中重叠根音错误", 0.3
    else:
        return "低重叠根音错误(严重)", 0.0


def parse_ground_truth_from_filename(filename: str) -> List[str]:
    """从文件名解析真实和弦序列"""
    # prog_0001_satb_C_major-G_major-A_minor-F_major.wav
    match = re.search(r'_([A-G]_[a-z0-9]+(?:-[A-G]_[a-z0-9]+)*)\.wav$', filename)
    if match:
        chord_str = match.group(1)
        chords = chord_str.split('-')
        # 规范化命名
        return [c.replace('_minor', '_min').replace('_diminished7', '_dim7').replace('_diminished', '_dim') 
                for c in chords]
    return []


def parse_predictions_from_txt(txt_file: Path) -> List[str]:
    """从txt文件解析预测的和弦序列"""
    predictions = []
    
    with open(txt_file, 'r', encoding='utf-8') as f:
        in_progression_section = False
        for line in f:
            # 找到"和弦进行序列"部分
            if "和弦进行序列" in line:
                in_progression_section = True
                continue
            
            if in_progression_section:
                # 结束标记
                if line.strip().startswith("="):
                    if predictions:  # 如果已经有预测结果了,说明这部分结束
                        break
                    continue
                
                # 解析和弦行
                # 格式: "  0.00-  2.00s |     2.00s |        C_maj7       "
                match = re.search(r'\|\s+[\d.]+s\s+\|\s+(\S+)\s*$', line)
                if match:
                    chord = match.group(1).strip()
                    predictions.append(chord.replace('_minor', '_min'))
    
    return predictions


# ==================== 评估函数 ====================

class AccuracyEvaluator:
    """准确度评估器"""
    
    def __init__(self):
        self.results = {
            "完全正确": 0,
            "类型混淆(可接受)": 0,
            "高重叠根音错误(可接受)": 0,
            "类型混淆(其他)": 0,
            "中重叠根音错误": 0,
            "低重叠根音错误(严重)": 0,
        }
        self.total_chords = 0
        self.detailed_results = []  # 保存详细结果
    
    def evaluate_pair(self, expected: str, predicted: str, file_info: str = ""):
        """评估单个和弦对"""
        error_type, score = classify_error(expected, predicted)
        self.results[error_type] += 1
        self.total_chords += 1
        
        self.detailed_results.append({
            "file": file_info,
            "expected": expected,
            "predicted": predicted,
            "error_type": error_type,
            "score": score
        })
        
        return error_type, score
    
    def evaluate_file(self, wav_file: Path, txt_file: Path):
        """评估单个文件"""
        # 从文件名解析真实标签
        ground_truth = parse_ground_truth_from_filename(wav_file.name)
        
        # 从txt文件解析预测结果
        predictions = parse_predictions_from_txt(txt_file)
        
        if not ground_truth:
            print(f"  ⚠️  无法从文件名解析真实标签: {wav_file.name}")
            return False
        
        if not predictions:
            print(f"  ⚠️  无法从txt解析预测结果: {txt_file.name}")
            return False
        
        # 对齐长度
        min_len = min(len(ground_truth), len(predictions))
        
        # 逐个评估
        for exp, pred in zip(ground_truth[:min_len], predictions[:min_len]):
            self.evaluate_pair(exp, pred, wav_file.name)
        
        return True
    
    def calculate_metrics(self) -> Dict[str, float]:
        """计算各种准确度指标"""
        if self.total_chords == 0:
            return {
                "严格准确度": 0.0,
                "宽松准确度": 0.0,
                "根音准确度": 0.0,
                "严重错误率": 0.0,
                "加权准确度": 0.0,
            }
        
        metrics = {}
        
        # 1. 严格准确度 (只有完全正确算对)
        metrics["严格准确度"] = self.results["完全正确"] / self.total_chords
        
        # 2. 宽松准确度 (包括可接受的错误)
        acceptable = (self.results["完全正确"] + 
                     self.results["类型混淆(可接受)"] + 
                     self.results["高重叠根音错误(可接受)"])
        metrics["宽松准确度"] = acceptable / self.total_chords
        
        # 3. 根音准确度 (根音正确即可)
        root_correct = (self.results["完全正确"] + 
                       self.results["类型混淆(可接受)"] + 
                       self.results["类型混淆(其他)"])
        metrics["根音准确度"] = root_correct / self.total_chords
        
        # 4. 严重错误率
        serious_errors = self.results["低重叠根音错误(严重)"]
        metrics["严重错误率"] = serious_errors / self.total_chords
        
        # 5. 加权准确度 (根据错误严重程度加权)
        weighted_score = sum(
            self.results[error_type] * self._get_weight(error_type)
            for error_type in self.results
        )
        metrics["加权准确度"] = weighted_score / self.total_chords
        
        return metrics
    
    def _get_weight(self, error_type: str) -> float:
        """获取错误类型的权重"""
        weights = {
            "完全正确": 1.0,
            "类型混淆(可接受)": 0.9,
            "高重叠根音错误(可接受)": 0.7,
            "类型混淆(其他)": 0.6,
            "中重叠根音错误": 0.3,
            "低重叠根音错误(严重)": 0.0,
        }
        return weights.get(error_type, 0.0)
    
    def print_report(self):
        """打印评估报告"""
        print("\n" + "="*80)
        print("和弦识别准确度评估报告")
        print("="*80 + "\n")
        
        print(f"总和弦数: {self.total_chords}\n")
        
        # 详细统计
        print("="*80)
        print("错误类型分布:")
        print("="*80)
        print(f"{'错误类型':<30s} {'数量':>10s} {'占比':>10s} {'权重':>8s}")
        print("-"*80)
        
        for error_type, count in self.results.items():
            percentage = (count / self.total_chords * 100) if self.total_chords > 0 else 0
            weight = self._get_weight(error_type)
            symbol = "✓" if weight >= 0.7 else ("~" if weight >= 0.3 else "✗")
            print(f"{symbol} {error_type:<27s} {count:>10d} {percentage:>9.1f}% {weight:>7.1f}")
        
        # 准确度指标
        metrics = self.calculate_metrics()
        
        print("\n" + "="*80)
        print("准确度指标:")
        print("="*80)
        print(f"{'指标':<30s} {'数值':>15s} {'说明'}")
        print("-"*80)
        
        print(f"严格准确度                    {metrics['严格准确度']*100:>14.2f}%  完全正确")
        print(f"宽松准确度                    {metrics['宽松准确度']*100:>14.2f}%  包括可接受错误")
        print(f"根音准确度                    {metrics['根音准确度']*100:>14.2f}%  根音正确")
        print(f"加权准确度                    {metrics['加权准确度']*100:>14.2f}%  考虑错误严重程度")
        print(f"严重错误率                    {metrics['严重错误率']*100:>14.2f}%  需要重点解决")
        
        # 可用性评估
        print("\n" + "="*80)
        print("可用性评估:")
        print("="*80 + "\n")
        
        if metrics["宽松准确度"] >= 0.85:
            usability = "✓ 优秀 - 可用于实际应用"
        elif metrics["宽松准确度"] >= 0.70:
            usability = "~ 良好 - 需要小幅改进"
        elif metrics["宽松准确度"] >= 0.50:
            usability = "! 一般 - 需要显著改进"
        else:
            usability = "✗ 较差 - 需要重新训练"
        
        print(f"  {usability}")
        print(f"  宽松准确度: {metrics['宽松准确度']*100:.1f}%")
        print(f"  严重错误率: {metrics['严重错误率']*100:.1f}%")
        
        # 改进建议
        print("\n" + "="*80)
        print("改进建议:")
        print("="*80 + "\n")
        
        if metrics["严重错误率"] > 0.15:
            print("  ⚠️  严重错误率过高(>15%),建议:")
            print("     1. 使用CQT特征提高低频分辨率")
            print("     2. 增强Bass声部权重")
            print("     3. 实现两阶段模型(根音→类型)")
        
        if metrics["根音准确度"] < 0.70:
            print("  ⚠️  根音识别不足(<70%),建议:")
            print("     1. 重点训练低音特征")
            print("     2. 添加根音分类损失函数")
        
        type_confusion = (self.results["类型混淆(可接受)"] + 
                         self.results["类型混淆(其他)"])
        if type_confusion / self.total_chords > 0.30:
            print("  ℹ️  类型混淆较多(>30%),可选优化:")
            print("     1. 增加七和弦训练样本")
            print("     2. 引入和弦类型特征")
        
        if metrics["宽松准确度"] >= 0.85:
            print("  ✓ 模型表现优秀,继续保持!")
    
    def save_detailed_report(self, output_file: Path):
        """保存详细报告到文件"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("和弦识别准确度评估详细报告\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"总和弦数: {self.total_chords}\n\n")
            
            # 错误分布
            f.write("="*80 + "\n")
            f.write("错误类型分布:\n")
            f.write("="*80 + "\n")
            f.write(f"{'错误类型':<30s} {'数量':>10s} {'占���':>10s}\n")
            f.write("-"*80 + "\n")
            
            for error_type, count in self.results.items():
                percentage = (count / self.total_chords * 100) if self.total_chords > 0 else 0
                f.write(f"{error_type:<30s} {count:>10d} {percentage:>9.1f}%\n")
            
            # 准确度指标
            metrics = self.calculate_metrics()
            
            f.write("\n" + "="*80 + "\n")
            f.write("准确度指标:\n")
            f.write("="*80 + "\n")
            
            for metric_name, value in metrics.items():
                f.write(f"{metric_name:<30s} {value*100:>14.2f}%\n")
            
            # 详细错误列表
            f.write("\n" + "="*80 + "\n")
            f.write("详细错误列表 (非完全正确的情况):\n")
            f.write("="*80 + "\n")
            f.write(f"{'文件':<50s} {'期望':<15s} {'预测':<15s} {'错误类型'}\n")
            f.write("-"*80 + "\n")
            
            for result in self.detailed_results:
                if result["error_type"] != "完全正确":
                    f.write(f"{result['file']:<50s} {result['expected']:<15s} "
                           f"{result['predicted']:<15s} {result['error_type']}\n")


# ==================== 主函数 ====================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='评估和弦识别准确度')
    parser.add_argument('--wav-dir', type=str, default='progressions_chords_output',
                       help='WAV文件目录')
    parser.add_argument('--txt-dir', type=str, default='analysis_test',
                       help='分析结果txt文件目录')
    parser.add_argument('--output', type=str, default='analysis_test/accuracy_report.txt',
                       help='详细报告输出文件')
    parser.add_argument('--limit', type=int, default=0,
                       help='限制评估文件数(0=全部)')
    
    args = parser.parse_args()
    
    wav_dir = Path(args.wav_dir)
    txt_dir = Path(args.txt_dir)
    
    if not wav_dir.exists():
        print(f"错误: WAV目录不存在 - {wav_dir}")
        return
    
    if not txt_dir.exists():
        print(f"错误: TXT目录不存在 - {txt_dir}")
        return
    
    print("\n" + "="*80)
    print("和弦识别准确度评估")
    print("="*80 + "\n")
    
    # 收集文件对
    wav_files = sorted(wav_dir.glob("prog_*_satb_*.wav"))
    
    if args.limit > 0:
        wav_files = wav_files[:args.limit]
    
    print(f"找到 {len(wav_files)} 个WAV文件\n")
    
    # 创建评估器
    evaluator = AccuracyEvaluator()
    
    # 逐个评估
    success_count = 0
    for i, wav_file in enumerate(wav_files, 1):
        # 找到对应的txt文件
        txt_file = txt_dir / (wav_file.stem + "_chord_analysis.txt")
        
        if not txt_file.exists():
            print(f"[{i}/{len(wav_files)}] ✗ {wav_file.name} - 缺少txt文件")
            continue
        
        if evaluator.evaluate_file(wav_file, txt_file):
            success_count += 1
            if i % 50 == 0:
                print(f"[{i}/{len(wav_files)}] 已处理 {success_count} 个文件...")
    
    print(f"\n✓ 成功评估 {success_count}/{len(wav_files)} 个文件")
    
    # 打印报告
    evaluator.print_report()
    
    # 保存详细报告
    output_file = Path(args.output)
    output_file.parent.mkdir(exist_ok=True)
    evaluator.save_detailed_report(output_file)
    
    print(f"\n✓ 详细报告已保存到: {output_file}")


if __name__ == '__main__':
    main()
