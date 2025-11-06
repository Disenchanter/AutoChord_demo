#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include "ChordRecognizer.h"

//==============================================================================
/**
 * AutoChord实时和弦识别插件处理器
 */
class AutoChordAudioProcessor : public juce::AudioProcessor
{
public:
    //==============================================================================
    AutoChordAudioProcessor();
    ~AutoChordAudioProcessor() override;

    //==============================================================================
    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

#ifndef JucePlugin_PreferredChannelConfigurations
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
#endif

    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    //==============================================================================
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    //==============================================================================
    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    //==============================================================================
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram(int index) override;
    const juce::String getProgramName(int index) override;
    void changeProgramName(int index, const juce::String& newName) override;

    //==============================================================================
    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    //==============================================================================
    ChordRecognizer& getChordRecognizer() { return chordRecognizer_; }
    
    // 加载模型
    void loadModels();

private:
    ChordRecognizer chordRecognizer_;
    
    // 模型文件路径
    juce::File getModelDirectory();
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AutoChordAudioProcessor)
};
