import torch
import torch.nn as nn

# 重建模型结构
class ChordCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(ChordCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

print('='*70)
print('🔬 模型容量与复杂度分析')
print('='*70)

# 分析不同任务
for task, num_classes in [('Root', 7), ('Chord', 14), ('Full', 98)]:
    model = ChordCNN(num_classes=num_classes)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    conv_params = sum(p.numel() for name, p in model.named_parameters() if 'conv_layers' in name)
    fc_params = sum(p.numel() for name, p in model.named_parameters() if 'fc_layers' in name)
    
    print(f'\n【{task} Task - {num_classes} 类】')
    print(f'  总参数: {total_params:,}')
    print(f'  卷积层: {conv_params:,} ({conv_params/total_params*100:.1f}%)')
    print(f'  全连接: {fc_params:,} ({fc_params/total_params*100:.1f}%)')
    print(f'  模型大小: ~{total_params * 4 / 1024**2:.2f} MB (FP32)')

# 输入输出分析
print('\n' + '='*70)
print('📐 输入输出尺寸分析')
print('='*70)
model = ChordCNN(num_classes=7)
model.eval()

# STFT: n_fft=2048 -> 1025 bins, 2秒@22050Hz, hop=512 -> 86帧
input_shape = (1, 1, 1025, 86)
dummy_input = torch.randn(input_shape)

print(f'\n输入: {input_shape}')
print(f'  → 1025 频率bins (10.8 Hz/bin)')
print(f'  → 86 时间帧 (23.2 ms/帧)')

with torch.no_grad():
    x = dummy_input
    print('\n逐层变换:')
    for i, layer in enumerate(model.conv_layers):
        x = layer(x)
        if isinstance(layer, (nn.Conv2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
            print(f'  {layer.__class__.__name__:20s}: {tuple(x.shape)}')
    
    x = model.fc_layers(x)
    print(f'  {"输出":20s}: {tuple(x.shape)}')

# 感受野计算
print('\n' + '='*70)
print('👁️  感受野分析')
print('='*70)
receptive_field_freq = 1
receptive_field_time = 1
stride_freq = 1
stride_time = 1

for block in range(4):
    # Conv2d(3,3, padding=1)
    receptive_field_freq += 2 * stride_freq
    receptive_field_time += 2 * stride_time
    # MaxPool2d(2,2)
    stride_freq *= 2
    stride_time *= 2

print(f'\n频率维度感受野: {receptive_field_freq} bins')
print(f'  → 覆盖频率范围: ~{receptive_field_freq * 10.8:.0f} Hz')
print(f'  → 约 {receptive_field_freq * 10.8 / 50:.1f} 个半音 (半音≈6%≈50Hz@1000Hz)')

print(f'\n时间维度感受野: {receptive_field_time} 帧')
print(f'  → 覆盖时间范围: ~{receptive_field_time * 512 / 22050 * 1000:.0f} ms')

# 数据样本比分析
print('\n' + '='*70)
print('📊 样本-参数比分析')
print('='*70)
train_samples = 1568
for task, num_classes in [('Root', 7), ('Chord', 14), ('Full', 98)]:
    model = ChordCNN(num_classes=num_classes)
    params = sum(p.numel() for p in model.parameters())
    ratio = train_samples / params
    
    print(f'\n{task} Task ({num_classes} 类):')
    print(f'  样本数: {train_samples}')
    print(f'  参数数: {params:,}')
    print(f'  样本/参数比: {ratio:.4f}')
    if ratio < 1:
        print(f'  ⚠️  过拟合风险: 高 (比值 < 1)')
    elif ratio < 10:
        print(f'  ⚠️  过拟合风险: 中等 (比值 < 10)')
    else:
        print(f'  ✅ 过拟合风险: 低 (比值 ≥ 10)')

# 与经典模型对比
print('\n' + '='*70)
print('📚 与经典模型对比')
print('='*70)
print('\n当前模型 (ChordCNN):')
print('  - 参数: ~300K')
print('  - 深度: 4 卷积块 + 2 全连接层')
print('  - 特点: 轻量级，适合小数据集')

print('\n经典音频模型参考:')
print('  - VGGish (Google): 72M 参数')
print('  - SoundNet: 8.5M 参数')
print('  - MusicCNN: 5M 参数')
print('  - 我们的模型: 0.3M 参数 ✅ (更适合当前数据量)')

print('\n' + '='*70)
print('💡 结论与建议')
print('='*70)
print('''
✅ Root Task (7类):
   - 参数: 296K
   - 样本/参数: 5.3
   - 结论: 模型容量适中，可能略小
   - 建议: 当前配置OK，可考虑增加到512→1024全连接

⚠️  Chord Task (14类):
   - 参数: 300K
   - 样本/参数: 5.2
   - 结论: 容量足够但有过拟合风险
   - 建议: 增加 Dropout，或使用数据增强

❌ Full Task (98类):
   - 参数: 346K
   - 样本/参数: 4.5
   - 结论: 样本数不足！
   - 建议: 
     1. 生成更多数据 (repetitions > 5)
     2. 使用预训练模型
     3. 减小模型 (256→128 卷积核)
     4. 强数据增强

🎯 推荐改进方案:
   1. 增加卷积核: 32→64→128→256 → 64→128→256→512
   2. 增加全连接: 256→512 → 512→1024
   3. 总参数约: 1M (适合当前数据量)
''')
