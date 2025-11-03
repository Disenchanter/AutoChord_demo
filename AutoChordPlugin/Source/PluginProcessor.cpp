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
    // 获取插件所在目录的父目录
    auto pluginDir = juce::File::getSpecialLocation(juce::File::currentExecutableFile).getParentDirectory();
    
    // 尝试多个可能的路径
    std::vector<juce::String> possiblePaths = {
        pluginDir.getFullPathName(),
        pluginDir.getParentDirectory().getFullPathName(),
        "D:\\Share_D\\Internship\\AutoChord_demo"  // 开发路径
    };
    
    for (const auto& path : possiblePaths)
    {
        juce::File dir(path);
        if (dir.exists())
            return dir;
    }
    
    return juce::File();
}

void AutoChordAudioProcessor::loadModels()
{
    auto baseDir = getModelDirectory();
    
    if (!baseDir.exists())
    {
        DBG("Model directory not found");
        return;
    }
    
    // 加载Root模型
    auto rootModelDir = baseDir.getChildFile("models_root_stft");
    if (rootModelDir.exists())
    {
        auto models = rootModelDir.findChildFiles(juce::File::findFiles, false, "*.pt");
        if (!models.isEmpty())
        {
            bool success = chordRecognizer_.loadModel(ChordRecognizer::ModelType::ROOT, models[0]);
            DBG("Root model loaded: " + juce::String(success ? "success" : "failed"));
        }
    }
    
    // 加载Chord模型
    auto chordModelDir = baseDir.getChildFile("models_chord_stft");
    if (chordModelDir.exists())
    {
        auto models = chordModelDir.findChildFiles(juce::File::findFiles, false, "*.pt");
        if (!models.isEmpty())
        {
            bool success = chordRecognizer_.loadModel(ChordRecognizer::ModelType::CHORD, models[0]);
            DBG("Chord model loaded: " + juce::String(success ? "success" : "failed"));
        }
    }
    
    // 加载Full模型
    auto fullModelDir = baseDir.getChildFile("models_full_stft");
    if (fullModelDir.exists())
    {
        auto models = fullModelDir.findChildFiles(juce::File::findFiles, false, "*.pt");
        if (!models.isEmpty())
        {
            bool success = chordRecognizer_.loadModel(ChordRecognizer::ModelType::FULL, models[0]);
            DBG("Full model loaded: " + juce::String(success ? "success" : "failed"));
        }
    }
}

//==============================================================================
// This creates new instances of the plugin..
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new AutoChordAudioProcessor();
}
