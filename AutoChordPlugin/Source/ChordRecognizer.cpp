#include "ChordRecognizer.h"
#include <cmath>

ChordRecognizer::ChordRecognizer()
    : currentModelType_(ModelType::FULL)
    , bufferSize_(static_cast<size_t>(SAMPLE_RATE * BUFFER_DURATION))
    , writePosition_(0)
    , currentConfidence_(0.0f)
{
    audioBuffer_.resize(bufferSize_, 0.0f);
    initializeLabels();
}

ChordRecognizer::~ChordRecognizer()
{
}

void ChordRecognizer::initializeLabels()
{
    // 根音标签 (7类)
    rootLabels_ = {"C", "D", "E", "F", "G", "A", "B"};
    
    // 和弦类型标签 (11类)
    chordLabels_ = {
        "major", "minor", "dim", "aug", "sus2", "sus4",
        "maj7", "min7", "dom7", "dim7", "hdim7"
    };
    
    // 完整和弦标签 (77类 = 7 roots × 11 chord types)
    fullLabels_.clear();
    for (const auto& root : rootLabels_)
    {
        for (const auto& chord : chordLabels_)
        {
            fullLabels_.push_back(root + "_" + chord);
        }
    }
}

bool ChordRecognizer::loadModel(ModelType type, const juce::File& modelFile)
{
    try
    {
        torch::jit::script::Module module = torch::jit::load(modelFile.getFullPathName().toStdString());
        module.eval();
        
        switch (type)
        {
            case ModelType::ROOT:
                rootModel_ = module;
                modelsLoaded_[0] = true;
                break;
            case ModelType::CHORD:
                chordModel_ = module;
                modelsLoaded_[1] = true;
                break;
            case ModelType::FULL:
                fullModel_ = module;
                modelsLoaded_[2] = true;
                break;
        }
        
        return true;
    }
    catch (const std::exception& e)
    {
        DBG("Failed to load model: " + juce::String(e.what()));
        return false;
    }
}

void ChordRecognizer::setModelType(ModelType type)
{
    currentModelType_ = type;
}

torch::jit::script::Module& ChordRecognizer::getCurrentModel()
{
    switch (currentModelType_)
    {
        case ModelType::ROOT: return rootModel_;
        case ModelType::CHORD: return chordModel_;
        case ModelType::FULL: return fullModel_;
    }
    return fullModel_;
}

const std::vector<juce::String>& ChordRecognizer::getCurrentLabels() const
{
    switch (currentModelType_)
    {
        case ModelType::ROOT: return rootLabels_;
        case ModelType::CHORD: return chordLabels_;
        case ModelType::FULL: return fullLabels_;
    }
    return fullLabels_;
}

void ChordRecognizer::processAudioBuffer(const juce::AudioBuffer<float>& buffer, double sampleRate)
{
    // 混合多声道为单声道
    auto numChannels = buffer.getNumChannels();
    auto numSamples = buffer.getNumSamples();
    
    for (int sample = 0; sample < numSamples; ++sample)
    {
        float mixedSample = 0.0f;
        for (int channel = 0; channel < numChannels; ++channel)
        {
            mixedSample += buffer.getSample(channel, sample);
        }
        mixedSample /= static_cast<float>(numChannels);
        
        // 写入循环缓冲区
        audioBuffer_[writePosition_] = mixedSample;
        writePosition_ = (writePosition_ + 1) % bufferSize_;
    }
    
    // 每隔一段时间进行推理（例如每512个样本）
    static int sampleCounter = 0;
    sampleCounter += numSamples;
    
    if (sampleCounter >= HOP_LENGTH)
    {
        sampleCounter = 0;
        runInference();
    }
}

torch::Tensor ChordRecognizer::computeSTFT(const std::vector<float>& audio)
{
    // 将音频转换为Tensor
    torch::Tensor audioTensor = torch::from_blob(
        const_cast<float*>(audio.data()),
        {static_cast<long>(audio.size())},
        torch::kFloat
    ).clone();
    
    // 应用汉宁窗
    torch::Tensor window = torch::hann_window(FFT_SIZE, torch::kFloat);
    
    // 计算STFT
    torch::Tensor stft = torch::stft(
        audioTensor,
        FFT_SIZE,
        HOP_LENGTH,
        FFT_SIZE,
        window,
        /*center=*/true,
        /*pad_mode=*/"reflect",
        /*normalized=*/false,
        /*onesided=*/true,
        /*return_complex=*/true
    );
    
    // 计算幅度谱
    torch::Tensor magnitude = torch::abs(stft);
    
    return magnitude;
}

void ChordRecognizer::runInference()
{
    // 检查模型是否加载
    int modelIndex = static_cast<int>(currentModelType_);
    if (!modelsLoaded_[modelIndex])
    {
        return;
    }
    
    try
    {
        // 准备输入数据
        std::vector<float> inputAudio(audioBuffer_.begin() + writePosition_, audioBuffer_.end());
        inputAudio.insert(inputAudio.end(), audioBuffer_.begin(), audioBuffer_.begin() + writePosition_);
        
        // 计算STFT（原始频谱，不转换为Mel）
        torch::Tensor stft = computeSTFT(inputAudio);
        
        // 转换为dB scale（与训练时一致）
        torch::Tensor stft_db = 20.0f * torch::log10(stft + 1e-9);
        
        // 调整维度 [1, 1, freq, time]
        // freq = 1025 (FFT_SIZE/2 + 1), time取决于音频长度
        stft_db = stft_db.unsqueeze(0).unsqueeze(0);
        
        // 推理
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(stft_db);
        
        auto& model = getCurrentModel();
        torch::Tensor output = model.forward(inputs).toTensor();
        
        // Softmax
        torch::Tensor probs = torch::softmax(output, 1);
        
        // 获取预测结果
        auto [maxProb, maxIdx] = torch::max(probs, 1);
        
        int predictedClass = maxIdx.item<int>();
        float confidence = maxProb.item<float>();
        
        // 更新结果
        const auto& labels = getCurrentLabels();
        
        std::lock_guard<std::mutex> lock(resultMutex_);
        
        if (predictedClass >= 0 && predictedClass < labels.size())
        {
            currentChord_ = labels[predictedClass];
            currentConfidence_ = confidence;
            
            // 更新概率分布
            probabilities_.clear();
            auto probsAccessor = probs.accessor<float, 2>();
            for (size_t i = 0; i < labels.size(); ++i)
            {
                probabilities_.push_back({labels[i], probsAccessor[0][i]});
            }
            
            // 按概率排序
            std::sort(probabilities_.begin(), probabilities_.end(),
                [](const auto& a, const auto& b) { return a.second > b.second; });
        }
    }
    catch (const std::exception& e)
    {
        DBG("Inference error: " + juce::String(e.what()));
    }
}

juce::String ChordRecognizer::getCurrentChord() const
{
    std::lock_guard<std::mutex> lock(resultMutex_);
    return currentChord_;
}

float ChordRecognizer::getCurrentConfidence() const
{
    std::lock_guard<std::mutex> lock(resultMutex_);
    return currentConfidence_;
}

std::vector<std::pair<juce::String, float>> ChordRecognizer::getProbabilities() const
{
    std::lock_guard<std::mutex> lock(resultMutex_);
    return probabilities_;
}
