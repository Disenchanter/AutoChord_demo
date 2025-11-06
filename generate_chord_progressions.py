#!/usr/bin/env python3
"""
生成和弦进行 MIDI 文件
基于 generate_single_chords.py 的架构，生成包含多个和弦变化的 MIDI 文件
每个和弦可以自定义时长（默认2秒）
"""

import argparse
import random
from pathlib import Path
from mido import MidiFile, MidiTrack, Message, MetaMessage
import mido

# 定义 C 大调音阶的根音
C_MAJOR_SCALE = {
    'C': 60,   # C4
    'D': 62,   # D4
    'E': 64,   # E4
    'F': 65,   # F4
    'G': 67,   # G4
    'A': 69,   # A4
    'B': 71,   # B4
}

# 定义和弦类型（11种）
CHORD_TYPES = {
    'major':  [0, 4, 7],           # 大三和弦
    'minor':  [0, 3, 7],           # 小三和弦
    'dim':    [0, 3, 6],           # 减三和弦
    'aug':    [0, 4, 8],           # 增三和弦
    'sus2':   [0, 2, 7],           # 挂二和弦
    'sus4':   [0, 5, 7],           # 挂四和弦
    'maj7':   [0, 4, 7, 11],       # 大七和弦
    'min7':   [0, 3, 7, 10],       # 小七和弦
    'dom7':   [0, 4, 7, 10],       # 属七和弦
    'dim7':   [0, 3, 6, 9],        # 减七和弦
    'hdim7':  [0, 3, 6, 10],       # 半减七和弦
}

# 定义各声部的音域
VOICE_RANGES = {
    'Soprano': (60, 79),      # C4-G5
    'Alto': (55, 74),         # G3-D5
    'Tenor': (48, 67),        # C3-G4
    'Bass': (40, 60),         # E2-C4
    'Piano_RH': (60, 84),     # C4-C6 (右手)
    'Piano_LH': (36, 60),     # C2-C4 (左手)
    'Strings': (48, 84),      # C3-C6
    'Pad': (36, 84),          # C2-C6
}

# 定义声部配置方案
VOICING_SCHEMES = {
    'satb': ['Soprano', 'Alto', 'Tenor', 'Bass'],
    'sat': ['Soprano', 'Alto', 'Tenor'],
    'atb': ['Alto', 'Tenor', 'Bass'],
    'sa': ['Soprano', 'Alto'],
    'piano': ['Piano_RH', 'Piano_LH'],
    'piano_bass': ['Piano_RH', 'Piano_LH', 'Bass'],
    'strings': ['Strings'],
    'full': ['Soprano', 'Alto', 'Tenor', 'Bass', 'Piano_RH', 'Piano_LH', 'Strings', 'Pad'],
}

# 常见和弦进行模板
COMMON_PROGRESSIONS = {
    'I-IV-V-I': ['C_major', 'F_major', 'G_major', 'C_major'],
    'I-V-vi-IV': ['C_major', 'G_major', 'A_minor', 'F_major'],
    'ii-V-I': ['D_minor', 'G_major', 'C_major'],
    'I-vi-IV-V': ['C_major', 'A_minor', 'F_major', 'G_major'],
    'I-iii-IV-V': ['C_major', 'E_minor', 'F_major', 'G_major'],
    'vi-IV-I-V': ['A_minor', 'F_major', 'C_major', 'G_major'],
    'I-IV-vi-V': ['C_major', 'F_major', 'A_minor', 'G_major'],
    'I-V-vi-iii-IV-I-IV-V': ['C_major', 'G_major', 'A_minor', 'E_minor', 'F_major', 'C_major', 'F_major', 'G_major'],
}


def generate_chord_notes(root, chord_type, voices):
    """
    为指定的和弦和声部生成音符
    
    Args:
        root: 根音的 MIDI 音高
        chord_type: 和弦类型 (major, minor, etc.)
        voices: 要使用的声部列表
    
    Returns:
        字典 {声部名: [音符列表]}
    """
    intervals = CHORD_TYPES[chord_type]
    num_notes = len(intervals)
    voice_notes = {}
    
    # 为每个声部分配音符
    for voice in voices:
        min_note, max_note = VOICE_RANGES[voice]
        notes = []
        
        # 根据声部的音域和和弦音选择合适的音符
        if num_notes <= 3:
            # 三和弦: 优先保证根音、五音、三音
            priority = [0, 2, 1]  # 根音 > 五音 > 三音
            for interval_idx in priority[:min(len(priority), num_notes)]:
                target_note = root + intervals[interval_idx]
                
                # 调整到合适的八度
                while target_note < min_note:
                    target_note += 12
                while target_note > max_note:
                    target_note -= 12
                
                if min_note <= target_note <= max_note:
                    notes.append(target_note)
                    
                    # 根音或五音可能加倍
                    if interval_idx in [0, 2]:
                        if random.random() > 0.7:  # 30% 概率加倍
                            doubled = target_note + 12
                            if doubled <= max_note:
                                notes.append(doubled)
                            elif target_note - 12 >= min_note:
                                notes.insert(0, target_note - 12)
        else:
            # 七和弦: 优先根音、三音、七音、五音
            priority = [0, 1, 3, 2]  # 根音 > 三音 > 七音 > 五音
            for interval_idx in priority[:min(len(priority), num_notes)]:
                target_note = root + intervals[interval_idx]
                
                while target_note < min_note:
                    target_note += 12
                while target_note > max_note:
                    target_note -= 12
                
                if min_note <= target_note <= max_note:
                    notes.append(target_note)
                    
                    if interval_idx in [0, 2]:
                        if random.random() > 0.7:
                            doubled = target_note + 12
                            if doubled <= max_note:
                                notes.append(doubled)
                            elif target_note - 12 >= min_note:
                                notes.insert(0, target_note - 12)
                    
                    if len(notes) < 2 and num_notes > 3 and random.random() > 0.6:
                        extra_idx = random.choice([i for i in range(num_notes) if i != interval_idx])
                        extra_note = root + intervals[extra_idx]
                        while extra_note < min_note:
                            extra_note += 12
                        while extra_note > max_note:
                            extra_note -= 12
                        if min_note <= extra_note <= max_note:
                            notes.append(extra_note)
        
        # 如果没有分配到音符,从所有和弦音中随机选一个
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


def create_progression_midi_for_voice(chord_sequence, voice_notes_sequence, voice_name, tempo=120, chord_duration=2.0):
    """
    为单个声部创建和弦进行的 MIDI 文件
    
    Args:
        chord_sequence: 和弦序列 [(root_name, chord_type), ...]
        voice_notes_sequence: 每个和弦的音符列表 [[notes_chord1], [notes_chord2], ...]
        voice_name: 声部名称
        tempo: 速度 (BPM)
        chord_duration: 每个和弦的持续时间 (秒)
    
    Returns:
        MidiFile 对象
    """
    mid = MidiFile(type=0, ticks_per_beat=480)
    track = MidiTrack()
    mid.tracks.append(track)
    
    # 设置 tempo
    track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo), time=0))
    track.append(MetaMessage('time_signature', numerator=4, denominator=4))
    track.append(MetaMessage('track_name', name=voice_name))
    
    ticks_per_chord = int(chord_duration * mid.ticks_per_beat)
    
    # 为每个和弦生成音符
    for chord_idx, notes in enumerate(voice_notes_sequence):
        if not notes:
            # 如果没有音符,添加静音时长
            if chord_idx == 0:
                time = ticks_per_chord
            else:
                time = ticks_per_chord
            continue
        
        # Note on 事件 (稍微错开时间以增加表现力)
        for i, note in enumerate(notes):
            velocity = random.randint(75, 100)
            time = 10 * i if i > 0 else 0
            track.append(Message('note_on', note=note, velocity=velocity, time=time, channel=0))
        
        # Note off 事件
        for i, note in enumerate(notes):
            if i == 0:
                time = ticks_per_chord - 10 * len(notes)
            else:
                time = 0  # 后续 note_off 时间为0 (同时关闭)
            track.append(Message('note_off', note=note, velocity=0, time=max(0, time), channel=0))
    
    track.append(MetaMessage('end_of_track', time=0))
    
    return mid


def parse_chord_string(chord_str):
    """
    解析和弦字符串,如 'C_major' 或 'D_minor'
    
    Returns:
        (root_name, chord_type)
    """
    parts = chord_str.split('_')
    if len(parts) != 2:
        raise ValueError(f"Invalid chord string: {chord_str}. Expected format: 'ROOT_TYPE' (e.g., 'C_major')")
    
    root_name, chord_type = parts
    
    if root_name not in C_MAJOR_SCALE:
        raise ValueError(f"Invalid root: {root_name}. Must be one of {list(C_MAJOR_SCALE.keys())}")
    
    if chord_type not in CHORD_TYPES:
        raise ValueError(f"Invalid chord type: {chord_type}. Must be one of {list(CHORD_TYPES.keys())}")
    
    return root_name, chord_type


def generate_random_progression(num_chords=4):
    """生成随机和弦进行"""
    roots = list(C_MAJOR_SCALE.keys())
    chord_types = list(CHORD_TYPES.keys())
    
    progression = []
    for _ in range(num_chords):
        root = random.choice(roots)
        chord_type = random.choice(chord_types)
        progression.append((root, chord_type))
    
    return progression


def generate_progressions(output_base, num_files=100, chords_per_file=4, 
                          chord_duration=2.0, tempo=120, 
                          progression_template=None, scheme='satb'):
    """
    生成和弦进行 MIDI 文件 (每个声部独立的 MIDI 文件)
    
    Args:
        output_base: 输出目录
        num_files: 要生成的文件数量
        chords_per_file: 每个文件中的和弦数量
        chord_duration: 每个和弦的持续时间 (秒)
        tempo: 速度 (BPM)
        progression_template: 使用的和弦进行模板名称 (可选)
        scheme: 声部配置方案
    """
    output_path = Path(output_base)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if scheme not in VOICING_SCHEMES:
        raise ValueError(f"Invalid scheme: {scheme}. Must be one of {list(VOICING_SCHEMES.keys())}")
    
    voices = VOICING_SCHEMES[scheme]
    
    print(f"生成和弦进行 MIDI...")
    print(f"输出目录: {output_path}")
    print(f"配置:")
    print(f"  - 文件数量: {num_files}")
    print(f"  - 每个文件和弦数: {chords_per_file}")
    print(f"  - 每个和弦时长: {chord_duration}秒")
    print(f"  - 速度: {tempo} BPM")
    print(f"  - 声部方案: {scheme} ({', '.join(voices)})")
    if progression_template:
        print(f"  - 进行模板: {progression_template}")
    print()
    
    for i in range(num_files):
        # 生成和弦序列
        if progression_template and progression_template in COMMON_PROGRESSIONS:
            # 使用模板进行
            template = COMMON_PROGRESSIONS[progression_template]
            chord_sequence = [parse_chord_string(chord_str) for chord_str in template]
            
            # 如果模板长度与要求不符,调整
            if len(chord_sequence) < chords_per_file:
                # 循环重复模板
                chord_sequence = chord_sequence * ((chords_per_file // len(chord_sequence)) + 1)
                chord_sequence = chord_sequence[:chords_per_file]
            elif len(chord_sequence) > chords_per_file:
                chord_sequence = chord_sequence[:chords_per_file]
        else:
            # 生成随机进行
            chord_sequence = generate_random_progression(chords_per_file)
        
        # 创建文件夹名称
        chord_names = '-'.join([f"{root}_{chord_type}" for root, chord_type in chord_sequence])
        folder_name = f"prog_{i+1:04d}_{scheme}_{chord_names[:50]}"
        folder_path = output_path / folder_name
        folder_path.mkdir(exist_ok=True)
        
        # 为每个和弦生成所有声部的音符
        all_voice_notes = []  # [(voice_name, [notes_for_chord1, notes_for_chord2, ...]), ...]
        
        for voice in voices:
            voice_sequence = []
            for root_name, chord_type in chord_sequence:
                root_midi = C_MAJOR_SCALE[root_name]
                voice_notes = generate_chord_notes(root_midi, chord_type, [voice])
                voice_sequence.append(voice_notes.get(voice, []))
            all_voice_notes.append((voice, voice_sequence))
        
        # 为每个声部创建单独的 MIDI 文件
        for voice_name, voice_sequence in all_voice_notes:
            midi = create_progression_midi_for_voice(
                chord_sequence, voice_sequence, voice_name, tempo, chord_duration
            )
            
            # 保存文件：文件名就是声部名
            midi_filename = f"{voice_name}.mid"
            midi_path = folder_path / midi_filename
            midi.save(str(midi_path))
        
        if (i + 1) % 10 == 0:
            print(f"已生成 {i+1}/{num_files} 个文件夹...")
    
    print()
    print(f"完成! 总共生成 {num_files} 个文件夹")
    print(f"位置: {output_path}")
    print()
    print(f"每个文件夹包含: {', '.join([v + '.mid' for v in voices])}")


def main():
    parser = argparse.ArgumentParser(
        description='生成和弦进行 MIDI 文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成100个随机和弦进行 (每个4个和弦, 每个和弦2秒)
  python generate_chord_progressions.py -n 100 -c 4 -d 2.0
  
  # 生成50个使用 I-V-vi-IV 进行的文件
  python generate_chord_progressions.py -n 50 -t I-V-vi-IV
  
  # 生成8个和弦的长进行, 每个和弦3秒
  python generate_chord_progressions.py -n 20 -c 8 -d 3.0
  
  # 使用 piano 声部方案
  python generate_chord_progressions.py -n 100 -s piano
  
可用的和弦进行模板:
  I-IV-V-I, I-V-vi-IV, ii-V-I, I-vi-IV-V, I-iii-IV-V, 
  vi-IV-I-V, I-IV-vi-V, I-V-vi-iii-IV-I-IV-V
  
可用的声部方案:
  satb, sat, atb, sa, piano, piano_bass, strings, full
        """
    )
    
    parser.add_argument('-n', '--num-files', type=int, default=100,
                        help='要生成的 MIDI 文件数量 (默认: 100)')
    parser.add_argument('-c', '--chords', type=int, default=4,
                        help='每个文件中的和弦数量 (默认: 4)')
    parser.add_argument('-d', '--duration', type=float, default=2.0,
                        help='每个和弦的持续时间 (秒) (默认: 2.0)')
    parser.add_argument('--tempo', type=int, default=120,
                        help='速度 (BPM) (默认: 120)')
    parser.add_argument('-t', '--template', type=str, default=None,
                        help='使用的和弦进行模板 (可选)')
    parser.add_argument('-s', '--scheme', type=str, default='satb',
                        help='声部配置方案 (默认: satb)')
    parser.add_argument('-o', '--output', type=str, default='chord_progressions',
                        help='输出目录 (默认: chord_progressions)')
    
    args = parser.parse_args()
    
    generate_progressions(
        output_base=args.output,
        num_files=args.num_files,
        chords_per_file=args.chords,
        chord_duration=args.duration,
        tempo=args.tempo,
        progression_template=args.template,
        scheme=args.scheme
    )


if __name__ == '__main__':
    main()
