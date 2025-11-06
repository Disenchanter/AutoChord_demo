#pragma once

#include <juce_core/juce_core.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>
#include <mutex>

/**
 * 和弦识别器类
 * 负责加载LibTorch模型并进行实时推理
 */
class ChordRecognizer
{
public:
    enum class ModelType
    {
        ROOT,   // 7类根音识别
        CHORD,  // 11类和弦类型识别
        FULL    // 77类完整和弦识别
    };

    ChordRecognizer();
    ~ChordRecognizer();

    // 加载模型
    bool loadModel(ModelType type, const juce::File& modelFile);
    
    // 设置当前使用的模型类型
    void setModelType(ModelType type);
    
    // 处理音频缓冲区
    void processAudioBuffer(const juce::AudioBuffer<float>& buffer, double sampleRate);
    
    // 获取当前识别结果
    juce::String getCurrentChord() const;
    float getCurrentConfidence() const;
    
    // 获取所有类别的概率分布
    std::vector<std::pair<juce::String, float>> getProbabilities() const;

private:
    // STFT参数（与训练代码保持一致 - 使用原始频谱）
    static constexpr int FFT_SIZE = 2048;
    static constexpr int HOP_LENGTH = 512;
    static constexpr int SAMPLE_RATE = 22050;  // 与训练时一致！
    static constexpr float BUFFER_DURATION = 2.0f; // 2秒音频缓冲
    static constexpr int N_FFT_BINS = FFT_SIZE / 2 + 1; // 1025个频率bins

    // 模型
    torch::jit::script::Module rootModel_;
    torch::jit::script::Module chordModel_;
    torch::jit::script::Module fullModel_;
    
    ModelType currentModelType_;
    bool modelsLoaded_[3] = {false, false, false};
    
    // 音频缓冲
    std::vector<float> audioBuffer_;
    size_t bufferSize_;
    size_t writePosition_;
    
    // 推理结果
    mutable std::mutex resultMutex_;
    juce::String currentChord_;
    float currentConfidence_;
    std::vector<std::pair<juce::String, float>> probabilities_;
    
    // 标签映射
    std::vector<juce::String> rootLabels_;
    std::vector<juce::String> chordLabels_;
    std::vector<juce::String> fullLabels_;
    
    // STFT处理（使用原始频谱，不转换为Mel）
    torch::Tensor computeSTFT(const std::vector<float>& audio);
    
    // 推理
    void runInference();
    
    // 初始化标签
    void initializeLabels();
    
    // 获取当前模型和标签
    torch::jit::script::Module& getCurrentModel();
    const std::vector<juce::String>& getCurrentLabels() const;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ChordRecognizer)
};
