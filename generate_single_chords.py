#!/usr/bin/env python3
"""
生成单和弦 MIDI 文件
每个和弦一个文件夹，包含各声部的单轨 MIDI
支持音符省略、加倍、声部对调等丰富变化
"""

import os
import random
import argparse
from pathlib import Path
from typing import List, Dict
import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage

# C大调音阶
C_MAJOR_SCALE = {
    'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11,
}

# 和弦定义
CHORD_TYPES = {
    'major': [0, 4, 7],
    'minor': [0, 3, 7],
    'dim': [0, 3, 6],
    'aug': [0, 4, 8],
    'sus2': [0, 2, 7],
    'sus4': [0, 5, 7],
    'maj7': [0, 4, 7, 11],
    'min7': [0, 3, 7, 10],
    'dom7': [0, 4, 7, 10],
    'dim7': [0, 3, 6, 9],
    'hdim7': [0, 3, 6, 10],
    }

# 声部配置
VOICE_CONFIGS = {
    'Soprano': {'range': (60, 79), 'channel': 0},
    'Alto': {'range': (55, 74), 'channel': 0},
    'Tenor': {'range': (48, 67), 'channel': 0},
    'Bass': {'range': (40, 60), 'channel': 0},
    'Piano_RH': {'range': (60, 84), 'channel': 0},
    'Piano_LH': {'range': (36, 60), 'channel': 0},
    'Strings': {'range': (36, 84), 'channel': 0},
    'Pad': {'range': (36, 72), 'channel': 0},
}

# 声部组合
VOICING_SCHEMES = {
    'satb': ['Soprano', 'Alto', 'Tenor', 'Bass'],
    'sat': ['Soprano', 'Alto', 'Tenor'],
    'atb': ['Alto', 'Tenor', 'Bass'],
    'sa': ['Soprano', 'Alto'],
    'piano': ['Piano_RH', 'Piano_LH'],
    'piano_bass': ['Piano_RH', 'Piano_LH', 'Bass'],
    'strings': ['Strings', 'Bass'],
    'full': ['Soprano', 'Alto', 'Tenor', 'Bass', 'Strings'],
}


def generate_chord_notes(root, chord_type, voices):
    """为各声部分配和弦音符（智能省略、加倍、八度变化）"""
    intervals = CHORD_TYPES[chord_type]
    num_notes = len(intervals)
    num_voices = len(voices)
    
    # 音符重要性排序
    if num_notes == 3:  # 三和弦
        priority = [0, 2, 1]  # 根、五、三（五音可省略）
    elif num_notes == 4:  # 七和弦
        priority = [0, 1, 3, 2]  # 根、三、七、五（五音最容易省略）
    else:  # 九和弦等
        priority = [0, 1, 3, 4, 2]  # 根、三、七、九、五
    
    # 根据声部数量选择音符
    if num_voices < num_notes:
        # 省略不重要的音
        selected_indices = priority[:num_voices]
    else:
        selected_indices = list(range(num_notes))
    
    voice_notes = {}
    
    # 为每个声部分配音符
    for voice_idx, voice in enumerate(voices):
        config = VOICE_CONFIGS[voice]
        min_note, max_note = config['range']
        
        notes = []
        
        # 选择要分配的和弦音
        if voice_idx < len(selected_indices):
            interval_idx = selected_indices[voice_idx]
            interval = intervals[interval_idx]
            
            # 找合适的八度
            target_note = root + interval
            
            # 调整到声部音域内
            while target_note < min_note:
                target_note += 12
            while target_note > max_note:
                target_note -= 12
            
            # 确保在范围内
            if min_note <= target_note <= max_note:
                notes.append(target_note)
                
                # 可能加倍（根音或五音）
                if interval_idx in [0, 2]:  # 根音或五音
                    if random.random() > 0.7:  # 30% 概率加倍
                        doubled = target_note + 12
                        if doubled <= max_note:
                            notes.append(doubled)
                        elif target_note - 12 >= min_note:
                            notes.insert(0, target_note - 12)
                
                # 有时添加额外的和弦音（增加丰富度）
                if len(notes) < 2 and num_notes > 3 and random.random() > 0.6:
                    # 添加另一个和弦音
                    extra_idx = random.choice([i for i in range(num_notes) if i != interval_idx])
                    extra_note = root + intervals[extra_idx]
                    while extra_note < min_note:
                        extra_note += 12
                    while extra_note > max_note:
                        extra_note -= 12
                    if min_note <= extra_note <= max_note:
                        notes.append(extra_note)
        
        # 如果没有分配到音符，从所有和弦音中随机选一个
        if not notes:
            available = []
            for interval in intervals:
                for octave in range(-2, 4):
                    candidate = root + interval + octave * 12
                    if min_note <= candidate <= max_note:
                        available.append(candidate)
            if available:
                notes.append(random.choice(available))
        
        voice_notes[voice] = notes
    
    return voice_notes


def apply_voice_swapping(voice_notes, voices, repetition_idx):
    """
    根据重复次数随机交换声部音符分配（配器变化）
    
    Args:
        voice_notes: 原始声部音符字典
        voices: 声部名称列表
        repetition_idx: 当前重复次数索引
    
    Returns:
        交换后的声部音符字典, 交换描述
    """
    if repetition_idx == 0:
        # 第一次重复保持原样
        return voice_notes, None
    
    # 使用重复索引作为随机种子（保证可重现性）
    rng = random.Random(repetition_idx)
    
    # 定义可以互换的声部组
    swap_groups = {
        'satb': [
            (['Soprano', 'Alto'], 0.3, 'S-A对调'),
            (['Tenor', 'Bass'], 0.3, 'T-B对调'),
            (['Alto', 'Tenor'], 0.2, 'A-T对调'),
        ],
        'piano': [
            (['Piano_RH', 'Piano_LH'], 0.4, '左右手对调'),
        ],
    }
    
    swapped_notes = voice_notes.copy()
    swap_desc = []
    
    # SATB 系列的对调
    if any(v in voices for v in ['Soprano', 'Alto', 'Tenor', 'Bass']):
        for pair, prob, desc in swap_groups['satb']:
            if all(v in voices for v in pair) and rng.random() < prob:
                # 交换这对声部的音符
                swapped_notes[pair[0]], swapped_notes[pair[1]] = \
                    swapped_notes[pair[1]], swapped_notes[pair[0]]
                swap_desc.append(desc)
                
    # Piano 的对调
    if all(v in voices for v in ['Piano_RH', 'Piano_LH']):
        for pair, prob, desc in swap_groups['piano']:
            if rng.random() < prob:
                swapped_notes[pair[0]], swapped_notes[pair[1]] = \
                    swapped_notes[pair[1]], swapped_notes[pair[0]]
                swap_desc.append(desc)
    
    # 全编制的随机重排（小概率）
    if len(voices) >= 4 and rng.random() < 0.15:  # 15% 概率完全重排
        voice_list = list(voices)
        notes_list = [swapped_notes[v] for v in voice_list]
        rng.shuffle(notes_list)
        swapped_notes = {voice_list[i]: notes_list[i] for i in range(len(voice_list))}
        swap_desc.append('完全重排')
    
    return swapped_notes, ('+'.join(swap_desc) if swap_desc else None)


def create_single_track_midi(notes, voice_name, tempo=120, duration=4.0):
    """创建单轨 Type 0 MIDI 文件，支持多个音符"""
    mid = MidiFile(type=0, ticks_per_beat=480)
    track = MidiTrack()
    mid.tracks.append(track)
    
    # 设置 tempo
    track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo), time=0))
    track.append(MetaMessage('time_signature', numerator=4, denominator=4))
    track.append(MetaMessage('track_name', name=voice_name))
    
    # 添加音符（随机速度增加表现力）
    ticks_duration = int(duration * mid.ticks_per_beat)
    
    # Note on 事件
    for i, note in enumerate(notes):
        velocity = random.randint(75, 100)  # 随机力度
        time = 10 * i if i > 0 else 0  # 稍微错开时间（10 ticks）
        track.append(Message('note_on', note=note, velocity=velocity, time=time, channel=0))
    
    # Note off 事件
    for i, note in enumerate(notes):
        # 主音符持续完整时长，其他音符可能略短
        if i == 0:
            time = ticks_duration - 10 * len(notes)
        else:
            time = ticks_duration - random.randint(50, 150)  # 略微变化
        track.append(Message('note_off', note=note, velocity=0, time=max(0, time), channel=0))
    
    track.append(MetaMessage('end_of_track', time=0))
    
    return mid


def generate_split_chords(output_base, repetitions=1):
    """生成拆分的和弦 MIDI 文件"""
    
    output_path = Path(output_base)
    output_path.mkdir(parents=True, exist_ok=True)
    
    total_count = 0
    
    print(f"生成单和弦 MIDI...")
    print(f"输出目录: {output_path}")
    print()
    
    for rep in range(repetitions):
        for root_name, root_midi in C_MAJOR_SCALE.items():
            for chord_type in CHORD_TYPES.keys():
                for scheme_name, voices in VOICING_SCHEMES.items():
                    
                    # 创建文件夹名称：根音_和弦类型_声部配置_序号
                    # 例如: C_major_satb_0001
                    chord_abbr = chord_type
                    folder_name = f"{root_name}_{chord_abbr}_{scheme_name}_{rep+1:02d}"
                    folder_path = output_path / folder_name
                    folder_path.mkdir(exist_ok=True)
                    
                    # 生成各声部的音符
                    voice_notes = generate_chord_notes(root_midi, chord_type, voices)
                    
                    # 应用声部对调（配器变化）
                    voice_notes, swap_info = apply_voice_swapping(voice_notes, voices, rep)
                    
                    # 为每个声部创建单独的 MIDI 文件
                    for voice_name in voices:
                        notes = voice_notes.get(voice_name, [])
                        
                        if notes:
                            # 创建单轨 MIDI
                            midi = create_single_track_midi(notes, voice_name)
                            
                            # 保存文件：文件名就是轨道名
                            midi_filename = f"{voice_name}.mid"
                            midi_path = folder_path / midi_filename
                            midi.save(str(midi_path))
                    
                    total_count += 1
                    
                    if total_count % 50 == 0:
                        print(f"已生成 {total_count} 个文件夹...")
    
    print()
    print(f"完成! 总共生成 {total_count} 个文件夹")
    print(f"位置: {output_path}")
    print()
    print(f"每个文件夹包含:")
    for scheme_name, voices in VOICING_SCHEMES.items():
        print(f"  {scheme_name}: {', '.join([v + '.mid' for v in voices])}")


def main():
    parser = argparse.ArgumentParser(description='生成单和弦 MIDI 文件（支持音符变化和声部对调）')
    parser.add_argument('-r', '--repetitions', type=int, default=1,
                        help='每个和弦的重复次数 (默认: 1)')
    parser.add_argument('-o', '--output', type=str, default='single_chords',
                        help='输出目录 (默认: single_chords)')

    args = parser.parse_args()
    
    generate_split_chords(args.output, args.repetitions)


if __name__ == '__main__':
    main()
