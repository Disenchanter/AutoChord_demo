#!/usr/bin/env python3
"""
生成长和弦进行 MIDI 文件 (60秒, 24个和弦)
每个进行都在同一个调式内,用于调性检测测试
"""

import argparse
import random
from pathlib import Path
from mido import MidiFile, MidiTrack, Message, MetaMessage

# 定义7个根音的MIDI音高
ROOTS = {
    'C': 60, 'D': 62, 'E': 64, 'F': 65, 'G': 67, 'A': 69, 'B': 71
}

# 定义和弦类型（11种）
CHORD_TYPES = {
    'major':  [0, 4, 7],
    'minor':  [0, 3, 7],
    'dim':    [0, 3, 6],
    'aug':    [0, 4, 8],
    'sus2':   [0, 2, 7],
    'sus4':   [0, 5, 7],
    'maj7':   [0, 4, 7, 11],
    'min7':   [0, 3, 7, 10],
    'dom7':   [0, 4, 7, 10],
    'dim7':   [0, 3, 6, 9],
    'hdim7':  [0, 3, 6, 10],
}

# 声部音域
VOICE_RANGES = {
    'Soprano': (60, 79),
    'Alto': (55, 74),
    'Tenor': (48, 67),
    'Bass': (40, 60),
}

# 定义24个调式的和弦库
KEYS = {
    # 大调
    'C_major': {
        'I': ['C_major', 'C_maj7'],
        'ii': ['D_minor', 'D_min7'],
        'iii': ['E_minor', 'E_min7'],
        'IV': ['F_major', 'F_maj7'],
        'V': ['G_major', 'G_dom7'],
        'vi': ['A_minor', 'A_min7'],
        'vii': ['B_dim', 'B_hdim7'],
    },
    'D_major': {
        'I': ['D_major', 'D_maj7'],
        'ii': ['E_minor', 'E_min7'],
        'iii': ['F_major', 'F_maj7'],  # 注: 实际应该是F#minor,但这里简化
        'IV': ['G_major', 'G_maj7'],
        'V': ['A_major', 'A_dom7'],
        'vi': ['B_minor', 'B_min7'],
        'vii': ['C_dim', 'C_hdim7'],
    },
    'E_major': {
        'I': ['E_major', 'E_maj7'],
        'ii': ['F_minor', 'F_min7'],
        'iii': ['G_minor', 'G_min7'],
        'IV': ['A_major', 'A_maj7'],
        'V': ['B_major', 'B_dom7'],
        'vi': ['C_minor', 'C_min7'],
        'vii': ['D_dim', 'D_hdim7'],
    },
    'F_major': {
        'I': ['F_major', 'F_maj7'],
        'ii': ['G_minor', 'G_min7'],
        'iii': ['A_minor', 'A_min7'],
        'IV': ['A_major', 'A_maj7'],  # 注: 这里为了简化用A而不是Bb
        'V': ['C_major', 'C_dom7'],
        'vi': ['D_minor', 'D_min7'],
        'vii': ['E_dim', 'E_hdim7'],
    },
    'G_major': {
        'I': ['G_major', 'G_maj7'],
        'ii': ['A_minor', 'A_min7'],
        'iii': ['B_minor', 'B_min7'],
        'IV': ['C_major', 'C_maj7'],
        'V': ['D_major', 'D_dom7'],
        'vi': ['E_minor', 'E_min7'],
        'vii': ['F_dim', 'F_hdim7'],
    },
    'A_major': {
        'I': ['A_major', 'A_maj7'],
        'ii': ['B_minor', 'B_min7'],
        'iii': ['C_minor', 'C_min7'],
        'IV': ['D_major', 'D_maj7'],
        'V': ['E_major', 'E_dom7'],
        'vi': ['F_minor', 'F_min7'],
        'vii': ['G_dim', 'G_hdim7'],
    },
    'B_major': {
        'I': ['B_major', 'B_maj7'],
        'ii': ['C_minor', 'C_min7'],
        'iii': ['D_minor', 'D_min7'],
        'IV': ['E_major', 'E_maj7'],
        'V': ['F_major', 'F_dom7'],
        'vi': ['G_minor', 'G_min7'],
        'vii': ['A_dim', 'A_hdim7'],
    },
    # 小调
    'A_minor': {
        'i': ['A_minor', 'A_min7'],
        'ii': ['B_dim', 'B_hdim7'],
        'III': ['C_major', 'C_maj7'],
        'iv': ['D_minor', 'D_min7'],
        'v': ['E_minor', 'E_min7'],
        'VI': ['F_major', 'F_maj7'],
        'VII': ['G_major', 'G_dom7'],
    },
    'B_minor': {
        'i': ['B_minor', 'B_min7'],
        'ii': ['C_dim', 'C_hdim7'],
        'III': ['D_major', 'D_maj7'],
        'iv': ['E_minor', 'E_min7'],
        'v': ['F_minor', 'F_min7'],
        'VI': ['G_major', 'G_maj7'],
        'VII': ['A_major', 'A_dom7'],
    },
    'C_minor': {
        'i': ['C_minor', 'C_min7'],
        'ii': ['D_dim', 'D_hdim7'],
        'III': ['E_major', 'E_maj7'],
        'iv': ['F_minor', 'F_min7'],
        'v': ['G_minor', 'G_min7'],
        'VI': ['A_major', 'A_maj7'],
        'VII': ['B_major', 'B_dom7'],
    },
    'D_minor': {
        'i': ['D_minor', 'D_min7'],
        'ii': ['E_dim', 'E_hdim7'],
        'III': ['F_major', 'F_maj7'],
        'iv': ['G_minor', 'G_min7'],
        'v': ['A_minor', 'A_min7'],
        'VI': ['B_major', 'B_maj7'],
        'VII': ['C_major', 'C_dom7'],
    },
    'E_minor': {
        'i': ['E_minor', 'E_min7'],
        'ii': ['F_dim', 'F_hdim7'],
        'III': ['G_major', 'G_maj7'],
        'iv': ['A_minor', 'A_min7'],
        'v': ['B_minor', 'B_min7'],
        'VI': ['C_major', 'C_maj7'],
        'VII': ['D_major', 'D_dom7'],
    },
    'F_minor': {
        'i': ['F_minor', 'F_min7'],
        'ii': ['G_dim', 'G_hdim7'],
        'III': ['A_major', 'A_maj7'],
        'iv': ['B_minor', 'B_min7'],
        'v': ['C_minor', 'C_min7'],
        'VI': ['D_major', 'D_maj7'],
        'VII': ['E_major', 'E_dom7'],
    },
    'G_minor': {
        'i': ['G_minor', 'G_min7'],
        'ii': ['A_dim', 'A_hdim7'],
        'III': ['B_major', 'B_maj7'],
        'iv': ['C_minor', 'C_min7'],
        'v': ['D_minor', 'D_min7'],
        'VI': ['E_major', 'E_maj7'],
        'VII': ['F_major', 'F_dom7'],
    },
}

# 常见的功能和声进行模板
PROGRESSION_TEMPLATES = [
    # 大调进行
    ['I', 'IV', 'V', 'I'],           # 经典终止
    ['I', 'V', 'vi', 'IV'],          # 流行进行
    ['I', 'vi', 'IV', 'V'],          # 50年代进行
    ['I', 'iii', 'IV', 'V'],         # 上行进行
    ['I', 'IV', 'vi', 'V'],          # 变体
    ['ii', 'V', 'I', 'vi'],          # 爵士进行
    ['I', 'V', 'vi', 'iii', 'IV', 'I', 'IV', 'V'],  # 扩展进行
    # 小调进行
    ['i', 'iv', 'v', 'i'],           # 小调终止
    ['i', 'VI', 'III', 'VII'],       # 小调流行
    ['i', 'VII', 'VI', 'VII'],       # 小调上行
    ['i', 'III', 'VII', 'iv'],       # 变体
]


def generate_chord_notes(root_name, chord_type, voices):
    """生成和弦的SATB声部"""
    root = ROOTS[root_name]
    intervals = CHORD_TYPES[chord_type]
    voice_notes = {}
    
    for voice in voices:
        min_note, max_note = VOICE_RANGES[voice]
        notes = []
        
        # 根据声部分配音符
        if voice == 'Bass':
            # 低音声部优先根音
            bass_note = root
            while bass_note < min_note:
                bass_note += 12
            while bass_note > max_note:
                bass_note -= 12
            notes = [bass_note]
        else:
            # 其他声部分配和弦音
            for interval in intervals:
                note = root + interval
                while note < min_note:
                    note += 12
                while note > max_note:
                    note -= 12
                if min_note <= note <= max_note:
                    notes.append(note)
                    break
        
        voice_notes[voice] = notes
    
    return voice_notes


def create_chord_progression(key_name, num_chords=24, chord_duration=2.5):
    """
    创建一个在指定调式内的和弦进行
    
    Args:
        key_name: 调式名称 (如 'C_major', 'A_minor')
        num_chords: 和弦数量 (默认24个 = 60秒)
        chord_duration: 每个和弦时长(秒, 默认2.5秒)
    
    Returns:
        和弦列表 ['C_major', 'F_major', ...]
    """
    key_chords = KEYS[key_name]
    progression = []
    
    # 首先选择一些功能和声模板
    templates = random.sample(PROGRESSION_TEMPLATES, k=min(3, len(PROGRESSION_TEMPLATES)))
    
    for template in templates:
        for function in template:
            if function in key_chords:
                chord = random.choice(key_chords[function])
                progression.append(chord)
                
                if len(progression) >= num_chords:
                    break
        
        if len(progression) >= num_chords:
            break
    
    # 如果还不够,随机添加调内和弦
    while len(progression) < num_chords:
        function = random.choice(list(key_chords.keys()))
        chord = random.choice(key_chords[function])
        progression.append(chord)
    
    # 确保以主和弦结束
    if len(progression) >= num_chords:
        if key_name.endswith('major'):
            progression[-1] = random.choice(key_chords['I'])
        else:  # minor
            progression[-1] = random.choice(key_chords['i'])
    
    return progression[:num_chords]


def create_piano_midi_file(chord_progression, output_filename, tempo=120, chord_duration=2.5):
    """
    创建钢琴版本的MIDI文件(所有音符在一个文件中)
    
    Args:
        chord_progression: 和弦列表
        output_filename: 输出文件名(Path对象)
        tempo: BPM
        chord_duration: 每个和弦时长(秒)
    """
    # 创建MIDI文件
    mid = MidiFile(type=0)
    track = MidiTrack()
    mid.tracks.append(track)
    
    # 添加tempo
    track.append(MetaMessage('set_tempo', tempo=int(60000000 / tempo)))
    track.append(MetaMessage('track_name', name='Piano_LH'))
    
    # 设置为钢琴音色 (Program 0 = Acoustic Grand Piano)
    track.append(Message('program_change', program=0, time=0))
    
    time_offset = 0
    
    for chord_name in chord_progression:
        # 解析和弦
        root_name, chord_type = chord_name.split('_')
        
        # 生成所有声部的音符
        voice_notes = generate_chord_notes(root_name, chord_type, ['Soprano', 'Alto', 'Tenor', 'Bass'])
        
        # 合并所有音符
        all_notes = []
        for voice in ['Soprano', 'Alto', 'Tenor', 'Bass']:
            all_notes.extend(voice_notes[voice])
        
        # 计算时长 (ticks)
        ticks_per_beat = mid.ticks_per_beat
        beats_per_chord = (chord_duration * tempo) / 60
        duration_ticks = int(beats_per_chord * ticks_per_beat)
        
        # Note on
        for i, note in enumerate(all_notes):
            track.append(Message('note_on', note=note, velocity=80, time=time_offset if i == 0 else 0))
        
        # Note off
        for i, note in enumerate(all_notes):
            track.append(Message('note_off', note=note, velocity=64, 
                               time=duration_ticks if i == 0 else 0))
        
        time_offset = 0
    
    # End of track
    track.append(MetaMessage('end_of_track'))
    
    # 保存文件
    mid.save(output_filename)


def main():
    parser = argparse.ArgumentParser(
        description='生成60秒长的和弦进行MIDI文件,每个进行在同一调式内',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--num-progressions', type=int, default=20,
                        help='生成的进行数量 (默认: 20)')
    parser.add_argument('--num-chords', type=int, default=24,
                        help='每个进行的和弦数量 (默认: 24个 = 60秒)')
    parser.add_argument('--chord-duration', type=float, default=2.5,
                        help='每个和弦的时长(秒, 默认: 2.5秒)')
    parser.add_argument('--output-dir', type=str, default='long_progressions_piano_midi',
                        help='输出目录 (默认: long_progressions_piano_midi)')
    parser.add_argument('--tempo', type=int, default=120,
                        help='BPM (默认: 120)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("生成长和弦进行 MIDI 文件")
    print("="*70)
    print(f"进行数量: {args.num_progressions}")
    print(f"每个进行: {args.num_chords}个和弦 × {args.chord_duration}秒 = {args.num_chords * args.chord_duration:.1f}秒")
    print(f"输出目录: {output_dir}")
    print()
    
    # 选择要使用的调式
    available_keys = list(KEYS.keys())
    
    # 生成进行
    for i in range(args.num_progressions):
        # 随机选择一个调式
        key_name = random.choice(available_keys)
        
        # 生成和弦进行
        progression = create_chord_progression(key_name, args.num_chords, args.chord_duration)
        
        # 创建文件夹和文件名
        progression_str = '-'.join(progression[:6])  # 只显示前6个和弦
        if len(progression) > 6:
            progression_str += f'-etc{len(progression)-6}more'
        
        folder_name = f"long_prog_{i+1:03d}_piano_{key_name}_{progression_str}"
        folder_path = output_dir / folder_name
        
        # 创建文件夹
        folder_path.mkdir(parents=True, exist_ok=True)
        
        # 创建钢琴MIDI文件 (文件名必须与Reaper轨道名一致)
        midi_filename = folder_path / "Piano_LH.mid"
        create_piano_midi_file(progression, midi_filename, args.tempo, args.chord_duration)
        
        print(f"[{i+1:2d}/{args.num_progressions}] ✓ {key_name:15} | {len(progression)}和弦 | {folder_name}/")
    
    print()
    print("="*70)
    print(f"✓ 完成! 共生成 {args.num_progressions} 个钢琴和弦进行文件夹")
    print(f"  输出目录: {output_dir}")
    print(f"  每个文件夹: 1个钢琴MIDI文件 (Piano_LH.mid)")
    print(f"  文件时长: ~{args.num_chords * args.chord_duration:.0f}秒")
    print()
    print("下一步:")
    print("  1. 用Reaper批量渲染: 打开batch_render_simple.lua脚本")
    print("     - 设置input_dir为: " + str(output_dir.absolute()))
    print("     - 运行脚本自动渲染所有文件夹")
    print("  2. 运行和弦识别: python chord_analyzer_simple.py <wav_file>")
    print("  3. 测试调性检测: python realtime_chord_analyzer.py <wav_file>")
    print("="*70)


if __name__ == '__main__':
    main()
