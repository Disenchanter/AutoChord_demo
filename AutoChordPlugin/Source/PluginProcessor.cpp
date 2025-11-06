#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
AutoChordAudioProcessor::AutoChordAudioProcessor()
#ifndef JucePlugin_PreferredChannelConfigurations
     : AudioProcessor(BusesProperties()
                     #if ! JucePlugin_IsMidiEffect
                      #if ! JucePlugin_IsSynth
                       .withInput("Input", juce::AudioChannelSet::stereo(), true)
                      #endif
                       .withOutput("Output", juce::AudioChannelSet::stereo(), true)
                     #endif
                       )
#endif
{
}

AutoChordAudioProcessor::~AutoChordAudioProcessor()
{
}

//==============================================================================
const juce::String AutoChordAudioProcessor::getName() const
{
    return JucePlugin_Name;
}

bool AutoChordAudioProcessor::acceptsMidi() const
{
   #if JucePlugin_WantsMidiInput
    return true;
   #else
    return false;
   #endif
}

bool AutoChordAudioProcessor::producesMidi() const
{
   #if JucePlugin_ProducesMidiOutput
    return true;
   #else
    return false;
   #endif
}

bool AutoChordAudioProcessor::isMidiEffect() const
{
   #if JucePlugin_IsMidiEffect
    return true;
   #else
    return false;
   #endif
}

double AutoChordAudioProcessor::getTailLengthSeconds() const
{
    return 0.0;
}

int AutoChordAudioProcessor::getNumPrograms()
{
    return 1;
}

int AutoChordAudioProcessor::getCurrentProgram()
{
    return 0;
}

void AutoChordAudioProcessor::setCurrentProgram(int index)
{
}

const juce::String AutoChordAudioProcessor::getProgramName(int index)
{
    return {};
}

void AutoChordAudioProcessor::changeProgramName(int index, const juce::String& newName)
{
}

//==============================================================================
void AutoChordAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    // 自动加载模型
    loadModels();
}

void AutoChordAudioProcessor::releaseResources()
{
}

#ifndef JucePlugin_PreferredChannelConfigurations
bool AutoChordAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
  #if JucePlugin_IsMidiEffect
    juce::ignoreUnused(layouts);
    return true;
  #else
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
     && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

   #if ! JucePlugin_IsSynth
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;
   #endif

    return true;
  #endif
}
#endif

void AutoChordAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    
    // 将音频数据传递给和弦识别器
    chordRecognizer_.processAudioBuffer(buffer, getSampleRate());
    
    // 插件不修改音频，直接通过
}

//==============================================================================
bool AutoChordAudioProcessor::hasEditor() const
{
    return true;
}

juce::AudioProcessorEditor* AutoChordAudioProcessor::createEditor()
{
    return new AutoChordAudioProcessorEditor(*this);
}

//==============================================================================
void AutoChordAudioProcessor::getStateInformation(juce::MemoryBlock& destData)
{
}

void AutoChordAudioProcessor::setStateInformation(const void* data, int sizeInBytes)
{
}

juce::File AutoChordAudioProcessor::getModelDirectory()
{
    // 获取插件可执行文件所在目录（编译时模型已复制到这里）
    return juce::File::getSpecialLocation(juce::File::currentExecutableFile).getParentDirectory();
}

void AutoChordAudioProcessor::loadModels()
{
    auto modelDir = getModelDirectory();
    
    if (!modelDir.exists())
    {
        DBG("Model directory not found: " + modelDir.getFullPathName());
        return;
    }
    
    DBG("Loading models from: " + modelDir.getFullPathName());
    
    // 加载编译时复制的模型文件
    auto rootModel = modelDir.getChildFile("root_model.pt");
    auto chordModel = modelDir.getChildFile("chord_model.pt");
    auto fullModel = modelDir.getChildFile("full_model.pt");
    
    // 加载Root模型（7类根音识别）
    if (rootModel.existsAsFile())
    {
        bool success = chordRecognizer_.loadModel(ChordRecognizer::ModelType::ROOT, rootModel);
        DBG("Root model loaded: " + juce::String(success ? "✓ SUCCESS" : "✗ FAILED"));
    }
    else
    {
        DBG("Root model not found: " + rootModel.getFullPathName());
    }
    
    // 加载Chord模型（11类和弦类型识别）
    if (chordModel.existsAsFile())
    {
        bool success = chordRecognizer_.loadModel(ChordRecognizer::ModelType::CHORD, chordModel);
        DBG("Chord model loaded: " + juce::String(success ? "✓ SUCCESS" : "✗ FAILED"));
    }
    else
    {
        DBG("Chord model not found: " + chordModel.getFullPathName());
    }
    
    // 加载Full模型（77类完整和弦识别）
    if (fullModel.existsAsFile())
    {
        bool success = chordRecognizer_.loadModel(ChordRecognizer::ModelType::FULL, fullModel);
        DBG("Full model loaded: " + juce::String(success ? "✓ SUCCESS" : "✗ FAILED"));
        
        // 默认使用Full模型
        chordRecognizer_.setModelType(ChordRecognizer::ModelType::FULL);
    }
    else
    {
        DBG("Full model not found: " + fullModel.getFullPathName());
    }
}

//==============================================================================
// This creates new instances of the plugin..
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new AutoChordAudioProcessor();
}
