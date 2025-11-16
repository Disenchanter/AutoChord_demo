#!/usr/bin/env python3
"""
生成有明显调性特征的和弦进行 MIDI 文件
用于调性分析测试，每个进行在单一调性内，使用常见功能和声进行
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

# 定义12个大调的音阶和和弦（使用标准记号）
MAJOR_KEYS = {
    'C_major': {
        'tonic': 'C',
        'scale_degrees': {
            'I': 'C_major', 'ii': 'D_minor', 'iii': 'E_minor',
            'IV': 'F_major', 'V': 'G_major', 'vi': 'A_minor', 'vii°': 'B_dim'
        },
        'seventh_chords': {
            'Imaj7': 'C_maj7', 'ii7': 'D_min7', 'iii7': 'E_min7',
            'IVmaj7': 'F_maj7', 'V7': 'G_dom7', 'vi7': 'A_min7', 'viiø7': 'B_hdim7'
        }
    },
    'D_major': {
        'tonic': 'D',
        'scale_degrees': {
            'I': 'D_major', 'ii': 'E_minor', 'iii': 'F_minor',
            'IV': 'G_major', 'V': 'A_major', 'vi': 'B_minor', 'vii°': 'C_dim'
        },
        'seventh_chords': {
            'Imaj7': 'D_maj7', 'ii7': 'E_min7', 'iii7': 'F_min7',
            'IVmaj7': 'G_maj7', 'V7': 'A_dom7', 'vi7': 'B_min7', 'viiø7': 'C_hdim7'
        }
    },
    'E_major': {
        'tonic': 'E',
        'scale_degrees': {
            'I': 'E_major', 'ii': 'F_minor', 'iii': 'G_minor',
            'IV': 'A_major', 'V': 'B_major', 'vi': 'C_minor', 'vii°': 'D_dim'
        },
        'seventh_chords': {
            'Imaj7': 'E_maj7', 'ii7': 'F_min7', 'iii7': 'G_min7',
            'IVmaj7': 'A_maj7', 'V7': 'B_dom7', 'vi7': 'C_min7', 'viiø7': 'D_hdim7'
        }
    },
    'F_major': {
        'tonic': 'F',
        'scale_degrees': {
            'I': 'F_major', 'ii': 'G_minor', 'iii': 'A_minor',
            'IV': 'A_major', 'V': 'C_major', 'vi': 'D_minor', 'vii°': 'E_dim'
        },
        'seventh_chords': {
            'Imaj7': 'F_maj7', 'ii7': 'G_min7', 'iii7': 'A_min7',
            'IVmaj7': 'A_maj7', 'V7': 'C_dom7', 'vi7': 'D_min7', 'viiø7': 'E_hdim7'
        }
    },
    'G_major': {
        'tonic': 'G',
        'scale_degrees': {
            'I': 'G_major', 'ii': 'A_minor', 'iii': 'B_minor',
            'IV': 'C_major', 'V': 'D_major', 'vi': 'E_minor', 'vii°': 'F_dim'
        },
        'seventh_chords': {
            'Imaj7': 'G_maj7', 'ii7': 'A_min7', 'iii7': 'B_min7',
            'IVmaj7': 'C_maj7', 'V7': 'D_dom7', 'vi7': 'E_min7', 'viiø7': 'F_hdim7'
        }
    },
    'A_major': {
        'tonic': 'A',
        'scale_degrees': {
            'I': 'A_major', 'ii': 'B_minor', 'iii': 'C_minor',
            'IV': 'D_major', 'V': 'E_major', 'vi': 'F_minor', 'vii°': 'G_dim'
        },
        'seventh_chords': {
            'Imaj7': 'A_maj7', 'ii7': 'B_min7', 'iii7': 'C_min7',
            'IVmaj7': 'D_maj7', 'V7': 'E_dom7', 'vi7': 'F_min7', 'viiø7': 'G_hdim7'
        }
    },
    'B_major': {
        'tonic': 'B',
        'scale_degrees': {
            'I': 'B_major', 'ii': 'C_minor', 'iii': 'D_minor',
            'IV': 'E_major', 'V': 'F_major', 'vi': 'G_minor', 'vii°': 'A_dim'
        },
        'seventh_chords': {
            'Imaj7': 'B_maj7', 'ii7': 'C_min7', 'iii7': 'D_min7',
            'IVmaj7': 'E_maj7', 'V7': 'F_dom7', 'vi7': 'G_min7', 'viiø7': 'A_hdim7'
        }
    },
}

# 定义12个小调的音阶和和弦（自然小调）
MINOR_KEYS = {
    'A_minor': {
        'tonic': 'A',
        'scale_degrees': {
            'i': 'A_minor', 'ii°': 'B_dim', 'III': 'C_major',
            'iv': 'D_minor', 'v': 'E_minor', 'VI': 'F_major', 'VII': 'G_major'
        },
        'seventh_chords': {
            'i7': 'A_min7', 'iiø7': 'B_hdim7', 'IIImaj7': 'C_maj7',
            'iv7': 'D_min7', 'v7': 'E_min7', 'VImaj7': 'F_maj7', 'VII7': 'G_dom7'
        }
    },
    'B_minor': {
        'tonic': 'B',
        'scale_degrees': {
            'i': 'B_minor', 'ii°': 'C_dim', 'III': 'D_major',
            'iv': 'E_minor', 'v': 'F_minor', 'VI': 'G_major', 'VII': 'A_major'
        },
        'seventh_chords': {
            'i7': 'B_min7', 'iiø7': 'C_hdim7', 'IIImaj7': 'D_maj7',
            'iv7': 'E_min7', 'v7': 'F_min7', 'VImaj7': 'G_maj7', 'VII7': 'A_dom7'
        }
    },
    'C_minor': {
        'tonic': 'C',
        'scale_degrees': {
            'i': 'C_minor', 'ii°': 'D_dim', 'III': 'E_major',
            'iv': 'F_minor', 'v': 'G_minor', 'VI': 'A_major', 'VII': 'B_major'
        },
        'seventh_chords': {
            'i7': 'C_min7', 'iiø7': 'D_hdim7', 'IIImaj7': 'E_maj7',
            'iv7': 'F_min7', 'v7': 'G_min7', 'VImaj7': 'A_maj7', 'VII7': 'B_dom7'
        }
    },
    'D_minor': {
        'tonic': 'D',
        'scale_degrees': {
            'i': 'D_minor', 'ii°': 'E_dim', 'III': 'F_major',
            'iv': 'G_minor', 'v': 'A_minor', 'VI': 'B_major', 'VII': 'C_major'
        },
        'seventh_chords': {
            'i7': 'D_min7', 'iiø7': 'E_hdim7', 'IIImaj7': 'F_maj7',
            'iv7': 'G_min7', 'v7': 'A_min7', 'VImaj7': 'B_maj7', 'VII7': 'C_dom7'
        }
    },
    'E_minor': {
        'tonic': 'E',
        'scale_degrees': {
            'i': 'E_minor', 'ii°': 'F_dim', 'III': 'G_major',
            'iv': 'A_minor', 'v': 'B_minor', 'VI': 'C_major', 'VII': 'D_major'
        },
        'seventh_chords': {
            'i7': 'E_min7', 'iiø7': 'F_hdim7', 'IIImaj7': 'G_maj7',
            'iv7': 'A_min7', 'v7': 'B_min7', 'VImaj7': 'C_maj7', 'VII7': 'D_dom7'
        }
    },
    'F_minor': {
        'tonic': 'F',
        'scale_degrees': {
            'i': 'F_minor', 'ii°': 'G_dim', 'III': 'A_major',
            'iv': 'B_minor', 'v': 'C_minor', 'VI': 'D_major', 'VII': 'E_major'
        },
        'seventh_chords': {
            'i7': 'F_min7', 'iiø7': 'G_hdim7', 'IIImaj7': 'A_maj7',
            'iv7': 'B_min7', 'v7': 'C_min7', 'VImaj7': 'D_maj7', 'VII7': 'E_dom7'
        }
    },
    'G_minor': {
        'tonic': 'G',
        'scale_degrees': {
            'i': 'G_minor', 'ii°': 'A_dim', 'III': 'B_major',
            'iv': 'C_minor', 'v': 'D_minor', 'VI': 'E_major', 'VII': 'F_major'
        },
        'seventh_chords': {
            'i7': 'G_min7', 'iiø7': 'A_hdim7', 'IIImaj7': 'B_maj7',
            'iv7': 'C_min7', 'v7': 'D_min7', 'VImaj7': 'E_maj7', 'VII7': 'F_dom7'
        }
    },
}

# 常见的功能和声进行模板（有明确调性特征）
MAJOR_PROGRESSIONS = {
    # 基础进行
    'authentic_cadence': ['I', 'V', 'I'],                    # 正格终止
    'plagal_cadence': ['I', 'IV', 'I'],                      # 变格终止
    'half_cadence': ['I', 'IV', 'V'],                        # 半终止
    
    # 流行进行
    'pop_progression_1': ['I', 'V', 'vi', 'IV'],             # 最常见的流行进行
    'pop_progression_2': ['I', 'vi', 'IV', 'V'],             # 50年代进行
    'pop_progression_3': ['vi', 'IV', 'I', 'V'],             # 变体1
    'pop_progression_4': ['I', 'IV', 'vi', 'V'],             # 变体2
    
    # 圆圈进行
    'circle_of_fifths': ['I', 'IV', 'vii°', 'iii', 'vi', 'ii', 'V', 'I'],
    'diatonic_circle': ['vi', 'ii', 'V', 'I'],               # 六二五一
    
    # 爵士进行
    'jazz_251': ['ii', 'V', 'I'],                            # 二五一
    'jazz_extended': ['iii', 'vi', 'ii', 'V', 'I'],          # 扩展爵士
    
    # 其他常见进行
    'ascending': ['I', 'ii', 'iii', 'IV', 'V', 'vi'],        # 上行
    'descending': ['I', 'vii°', 'vi', 'V', 'IV', 'iii', 'ii', 'I'],  # 下行
    'pachelbel': ['I', 'V', 'vi', 'iii', 'IV', 'I', 'IV', 'V'],  # 卡农和弦
}

MINOR_PROGRESSIONS = {
    # 基础进行
    'minor_cadence': ['i', 'v', 'i'],                        # 小调终止
    'minor_plagal': ['i', 'iv', 'i'],                        # 小调变格终止
    
    # 流行小调进行
    'minor_pop_1': ['i', 'VI', 'III', 'VII'],                # 小调流行1
    'minor_pop_2': ['i', 'VII', 'VI', 'VII'],                # 小调流行2
    'minor_pop_3': ['i', 'iv', 'VII', 'III'],                # 小调流行3
    'minor_pop_4': ['vi', 'VII', 'i', 'III'],                # Andalusian变体
    
    # 自然小调进行
    'natural_minor_1': ['i', 'VII', 'VI', 'v'],              # 自然小调1
    'natural_minor_2': ['i', 'III', 'VII', 'iv'],            # 自然小调2
    
    # 小调圆圈进行
    'minor_circle': ['i', 'iv', 'VII', 'III', 'VI', 'ii°', 'v', 'i'],
    'minor_251': ['ii°', 'v', 'i'],                          # 小调二五一
}


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


def create_tonal_progression(key_name, num_chords=16, use_seventh=0.3):
    """
    创建一个在指定调式内的、有明显调性特征的和弦进行
    
    Args:
        key_name: 调式名称 (如 'C_major', 'A_minor')
        num_chords: 目标和弦数量
        use_seventh: 使用七和弦的概率 (0-1)
    
    Returns:
        和弦列表 ['C_major', 'F_major', ...]
    """
    is_major = key_name.endswith('_major')
    
    if is_major:
        key_info = MAJOR_KEYS[key_name]
        progression_templates = MAJOR_PROGRESSIONS
    else:
        key_info = MINOR_KEYS[key_name]
        progression_templates = MINOR_PROGRESSIONS
    
    scale_degrees = key_info['scale_degrees']
    seventh_chords = key_info['seventh_chords']
    
    progression = []
    
    # 从主和弦开始
    if random.random() < use_seventh:
        progression.append(seventh_chords['Imaj7' if is_major else 'i7'])
    else:
        progression.append(scale_degrees['I' if is_major else 'i'])
    
    # 选择多个功能和声模板
    templates = random.sample(
        list(progression_templates.values()), 
        k=min(3, len(progression_templates))
    )
    
    # 应用模板
    for template in templates:
        for degree in template:
            # 决定是否使用七和弦
            if random.random() < use_seventh:
                # 映射到七和弦标记
                seventh_map = {
                    'I': 'Imaj7', 'ii': 'ii7', 'iii': 'iii7', 'IV': 'IVmaj7',
                    'V': 'V7', 'vi': 'vi7', 'vii°': 'viiø7',
                    'i': 'i7', 'ii°': 'iiø7', 'III': 'IIImaj7', 'iv': 'iv7',
                    'v': 'v7', 'VI': 'VImaj7', 'VII': 'VII7'
                }
                seventh_degree = seventh_map.get(degree, degree)
                if seventh_degree in seventh_chords:
                    chord = seventh_chords[seventh_degree]
                elif degree in scale_degrees:
                    chord = scale_degrees[degree]
                else:
                    continue
            else:
                if degree in scale_degrees:
                    chord = scale_degrees[degree]
                else:
                    continue
            
            progression.append(chord)
            
            if len(progression) >= num_chords:
                break
        
        if len(progression) >= num_chords:
            break
    
    # 如果还不够，随机添加调内和弦（偏好主、属、下属功能）
    functional_degrees = ['I', 'IV', 'V'] if is_major else ['i', 'iv', 'v']
    while len(progression) < num_chords:
        if random.random() < 0.7:  # 70%使用主要功能和弦
            degree = random.choice(functional_degrees)
        else:
            degree = random.choice(list(scale_degrees.keys()))
        
        if random.random() < use_seventh and degree in ['I', 'ii', 'iii', 'IV', 'V', 'vi', 'i', 'iv', 'v', 'VI', 'VII']:
            seventh_map = {
                'I': 'Imaj7', 'ii': 'ii7', 'iii': 'iii7', 'IV': 'IVmaj7',
                'V': 'V7', 'vi': 'vi7',
                'i': 'i7', 'iv': 'iv7', 'v': 'v7', 'VI': 'VImaj7', 'VII': 'VII7'
            }
            seventh_degree = seventh_map.get(degree, degree)
            if seventh_degree in seventh_chords:
                chord = seventh_chords[seventh_degree]
            else:
                chord = scale_degrees[degree]
        else:
            chord = scale_degrees[degree]
        
        progression.append(chord)
    
    # 确保以主和弦结束（强化调性）
    if len(progression) >= num_chords:
        if random.random() < use_seventh:
            progression[-1] = seventh_chords['Imaj7' if is_major else 'i7']
        else:
            progression[-1] = scale_degrees['I' if is_major else 'i']
    
    return progression[:num_chords]


def create_midi_files_per_voice(chord_progression, output_dir, tempo=120, chord_duration=2.5):
    """
    为每个声部创建单独的MIDI文件
    
    Args:
        chord_progression: 和弦列表
        output_dir: 输出目录(Path对象)
        tempo: BPM
        chord_duration: 每个和弦时长(秒)
    """
    voices = ['Soprano', 'Alto', 'Tenor', 'Bass']
    
    for voice in voices:
        # 创建MIDI文件
        mid = MidiFile(type=0)
        track = MidiTrack()
        mid.tracks.append(track)
        
        # 添加tempo
        track.append(MetaMessage('set_tempo', tempo=int(60000000 / tempo)))
        track.append(MetaMessage('track_name', name=voice))
        
        time_offset = 0
        
        for chord_name in chord_progression:
            # 解析和弦
            root_name, chord_type = chord_name.split('_')
            
            # 生成音符
            voice_notes = generate_chord_notes(root_name, chord_type, [voice])
            notes = voice_notes[voice]
            
            # 计算时长 (ticks)
            ticks_per_beat = mid.ticks_per_beat
            beats_per_chord = (chord_duration * tempo) / 60
            duration_ticks = int(beats_per_chord * ticks_per_beat)
            
            # Note on
            for i, note in enumerate(notes):
                track.append(Message('note_on', note=note, velocity=80, time=time_offset if i == 0 else 0))
            
            # Note off
            for i, note in enumerate(notes):
                track.append(Message('note_off', note=note, velocity=64, 
                                   time=duration_ticks if i == 0 else 0))
            
            time_offset = 0
        
        # End of track
        track.append(MetaMessage('end_of_track'))
        
        # 保存文件
        voice_filename = f"{voice}.mid"
        mid.save(output_dir / voice_filename)


def main():
    parser = argparse.ArgumentParser(
        description='生成有明显调性特征的和弦进行MIDI文件，用于调性分析测试',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 生成20个进行，每个16个和弦（40秒）
  python generate_tonal_progressions.py --num-progressions 20 --num-chords 16
  
  # 生成50个进行，每个24个和弦（60秒），更多七和弦
  python generate_tonal_progressions.py -n 50 -c 24 --seventh-prob 0.5
  
  # 只生成大调进行
  python generate_tonal_progressions.py -n 30 --major-only
  
  # 只生成小调进行
  python generate_tonal_progressions.py -n 30 --minor-only
        """
    )
    
    parser.add_argument('-n', '--num-progressions', type=int, default=30,
                        help='生成的进行数量 (默认: 30)')
    parser.add_argument('-c', '--num-chords', type=int, default=16,
                        help='每个进行的和弦数量 (默认: 16个 = 40秒)')
    parser.add_argument('-d', '--chord-duration', type=float, default=2.5,
                        help='每个和弦的时长(秒, 默认: 2.5秒)')
    parser.add_argument('-o', '--output-dir', type=str, default='tonal_progressions_midi',
                        help='输出目录 (默认: tonal_progressions_midi)')
    parser.add_argument('-t', '--tempo', type=int, default=120,
                        help='BPM (默认: 120)')
    parser.add_argument('--seventh-prob', type=float, default=0.3,
                        help='使用七和弦的概率 0-1 (默认: 0.3)')
    parser.add_argument('--major-only', action='store_true',
                        help='只生成大调进行')
    parser.add_argument('--minor-only', action='store_true',
                        help='只生成小调进行')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 选择要使用的调式
    if args.major_only:
        available_keys = list(MAJOR_KEYS.keys())
        mode_label = "仅大调"
    elif args.minor_only:
        available_keys = list(MINOR_KEYS.keys())
        mode_label = "仅小调"
    else:
        available_keys = list(MAJOR_KEYS.keys()) + list(MINOR_KEYS.keys())
        mode_label = "大调+小调"
    
    print("="*80)
    print("生成有明显调性特征的和弦进行 MIDI 文件")
    print("="*80)
    print(f"进行数量: {args.num_progressions}")
    print(f"每个进行: {args.num_chords}个和弦 × {args.chord_duration}秒 = {args.num_chords * args.chord_duration:.1f}秒")
    print(f"调式范围: {mode_label} ({len(available_keys)}个调)")
    print(f"七和弦概率: {args.seventh_prob*100:.0f}%")
    print(f"输出目录: {output_dir}")
    print()
    
    # 生成进行
    for i in range(args.num_progressions):
        # 随机选择一个调式
        key_name = random.choice(available_keys)
        
        # 生成调性明确的和弦进行
        progression = create_tonal_progression(
            key_name, 
            args.num_chords, 
            args.seventh_prob
        )
        
        # 创建文件夹名称（显示前5个和弦）
        progression_str = '-'.join([c.replace('_', '') for c in progression[:5]])
        if len(progression) > 5:
            progression_str += f'-etc{len(progression)-5}'
        
        folder_name = f"tonal_{i+1:03d}_satb_{key_name}_{progression_str}"
        folder_path = output_dir / folder_name
        
        # 创建文件夹
        folder_path.mkdir(parents=True, exist_ok=True)
        
        # 为每个声部创建MIDI文件
        create_midi_files_per_voice(
            progression, 
            folder_path, 
            args.tempo, 
            args.chord_duration
        )
        
        # 显示进度
        mode = "大调" if key_name.endswith('_major') else "小调"
        print(f"[{i+1:3d}/{args.num_progressions}] ✓ {key_name:15} ({mode}) | "
              f"{len(progression):2d}和弦 | {folder_name}/")
    
    print()
    print("="*80)
    print(f"✓ 完成! 共生成 {args.num_progressions} 个调性明确的和弦进行")
    print(f"  输出目录: {output_dir.absolute()}")
    print(f"  每个文件夹: 4个声部MIDI文件 (Soprano, Alto, Tenor, Bass)")
    print(f"  文件时长: ~{args.num_chords * args.chord_duration:.0f}秒")
    print()
    print("📝 和弦进行特点:")
    print("  - 使用常见功能和声进行（I-IV-V, I-V-vi-IV等）")
    print("  - 每个进行在单一调性内，调性中心明确")
    print("  - 适合调性检测和分析算法测试")
    print()
    print("🎵 下一步:")
    print("  1. 用Reaper批量渲染: 打开 midi_render.lua 脚本")
    print("     - 设置 input_dir 为: " + str(output_dir.absolute()))
    print("     - 运行脚本自动渲染所有文件夹为WAV")
    print("  2. 测试调性检测:")
    print("     - python analyze_tonality.py <wav_file>")
    print("  3. 批量测试:")
    print("     - python batch_tonality_analysis.py --input-dir <wav_dir>")
    print("="*80)


if __name__ == '__main__':
    main()
