#!/usr/bin/env python3
"""
评估长进行音频文件的调性检测准确率
"""

import os
import re
import subprocess
from pathlib import Path
from collections import defaultdict

def extract_key_from_filename(filename):
    """从文件名提取真实调性"""
    # 文件名格式: long_prog_XXX_piano_KEY_...
    match = re.search(r'long_prog_\d+_piano_([A-G]#?_(?:major|minor))', filename)
    if match:
        return match.group(1)
    return None

def run_key_detection(wav_file):
    """运行调性检测脚本并提取结果"""
    try:
        result = subprocess.run(
            ['conda', 'run', '-n', 'librosa', 'python', 'realtime_chord_analyzer.py', 
             wav_file, '--no-realtime'],
            capture_output=True,
            text=True,
            timeout=180
        )
        
        output = result.stdout
        
        # 提取最终调性
        match = re.search(r'✓ 最终调性: ([A-G]#?_(?:major|minor)) \(置信度: ([\d.]+)%\)', output)
        if match:
            predicted_key = match.group(1)
            confidence = float(match.group(2))
            
            # 提取调性变化次数
            changes_count = len(re.findall(r'\d+\.\s+[A-G]#?_(?:major|minor)', output))
            
            return predicted_key, confidence, changes_count
        
    except Exception as e:
        print(f"  ✗ 错误: {e}")
    
    return None, None, None

def main():
    output_dir = Path('long_progressions_piano_midi_output')
    
    if not output_dir.exists():
        print(f"✗ 找不到目录: {output_dir}")
        return
    
    # 获取所有WAV文件
    wav_files = sorted(output_dir.glob('*.wav'))
    
    if not wav_files:
        print(f"✗ 目录中没有WAV文件: {output_dir}")
        return
    
    print("="*80)
    print(f"评估调性检测准确率 - 共 {len(wav_files)} 个文件")
    print("="*80)
    print()
    
    results = []
    correct_count = 0
    
    for i, wav_file in enumerate(wav_files, 1):
        filename = wav_file.name
        true_key = extract_key_from_filename(filename)
        
        if not true_key:
            print(f"[{i:2d}/{len(wav_files)}] ⚠️  无法从文件名提取调性: {filename}")
            continue
        
        print(f"[{i:2d}/{len(wav_files)}] {filename[:60]}...")
        print(f"  真实调性: {true_key}")
        
        predicted_key, confidence, changes = run_key_detection(str(wav_file))
        
        if predicted_key:
            is_correct = (predicted_key == true_key)
            correct_count += is_correct
            
            symbol = "✓" if is_correct else "✗"
            print(f"  预测调性: {predicted_key} (置信度: {confidence:.1f}%, 变化: {changes}次) {symbol}")
            
            results.append({
                'filename': filename,
                'true_key': true_key,
                'predicted_key': predicted_key,
                'confidence': confidence,
                'changes': changes,
                'correct': is_correct
            })
        else:
            print(f"  预测失败")
            results.append({
                'filename': filename,
                'true_key': true_key,
                'predicted_key': None,
                'confidence': None,
                'changes': None,
                'correct': False
            })
        
        print()
    
    # 统计结果
    print("="*80)
    print("评估结果")
    print("="*80)
    print()
    
    total = len(results)
    accuracy = (correct_count / total * 100) if total > 0 else 0
    
    print(f"总文件数: {total}")
    print(f"正确识别: {correct_count}")
    print(f"识别错误: {total - correct_count}")
    print(f"准确率: {accuracy:.1f}%")
    print()
    
    # 按调性统计
    key_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    for r in results:
        if r['true_key']:
            key_stats[r['true_key']]['total'] += 1
            if r['correct']:
                key_stats[r['true_key']]['correct'] += 1
    
    print("按调性统计:")
    for key in sorted(key_stats.keys()):
        stats = key_stats[key]
        acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"  {key:15} : {stats['correct']}/{stats['total']} ({acc:.0f}%)")
    print()
    
    # 错误分析
    errors = [r for r in results if not r['correct']]
    if errors:
        print("错误详情:")
        for r in errors:
            print(f"  {r['true_key']:15} → {r['predicted_key']:15} | {r['filename'][:50]}")
    
    print()
    print("="*80)

if __name__ == '__main__':
    main()
