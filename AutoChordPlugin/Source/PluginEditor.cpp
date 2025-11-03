#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
AutoChordAudioProcessorEditor::AutoChordAudioProcessorEditor(AutoChordAudioProcessor& p)
    : AudioProcessorEditor(&p), audioProcessor(p)
{
    // 标题
    titleLabel_.setText("AutoChord - Real-time Chord Recognition", juce::dontSendNotification);
    titleLabel_.setFont(juce::Font(24.0f, juce::Font::bold));
    titleLabel_.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(titleLabel_);
    
    // 和弦显示
    chordLabel_.setText("--", juce::dontSendNotification);
    chordLabel_.setFont(juce::Font(72.0f, juce::Font::bold));
    chordLabel_.setJustificationType(juce::Justification::centred);
    chordLabel_.setColour(juce::Label::textColourId, juce::Colours::lightgreen);
    addAndMakeVisible(chordLabel_);
    
    // 置信度显示
    confidenceLabel_.setText("Confidence: --", juce::dontSendNotification);
    confidenceLabel_.setFont(juce::Font(18.0f));
    confidenceLabel_.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(confidenceLabel_);
    
    // 模型选择器
    modelSelector_.addItem("Root (7 classes)", 1);
    modelSelector_.addItem("Chord Type (11 classes)", 2);
    modelSelector_.addItem("Full Chord (77 classes)", 3);
    modelSelector_.setSelectedId(3);
    modelSelector_.onChange = [this]
    {
        int selectedId = modelSelector_.getSelectedId();
        ChordRecognizer::ModelType type;
        
        switch (selectedId)
        {
            case 1: type = ChordRecognizer::ModelType::ROOT; break;
            case 2: type = ChordRecognizer::ModelType::CHORD; break;
            case 3: type = ChordRecognizer::ModelType::FULL; break;
            default: type = ChordRecognizer::ModelType::FULL; break;
        }
        
        audioProcessor.getChordRecognizer().setModelType(type);
    };
    addAndMakeVisible(modelSelector_);
    
    // 加载模型按钮
    loadModelButton_.setButtonText("Load Models");
    loadModelButton_.onClick = [this]
    {
        audioProcessor.loadModels();
    };
    addAndMakeVisible(loadModelButton_);
    
    // 概率条
    addAndMakeVisible(probabilityBar_);
    
    // 启动定时器更新UI（30Hz）
    startTimerHz(30);
    
    setSize(600, 700);
}

AutoChordAudioProcessorEditor::~AutoChordAudioProcessorEditor()
{
    stopTimer();
}

//==============================================================================
void AutoChordAudioProcessorEditor::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colour(0xff1a1a2e));
    
    // 绘制装饰线
    g.setColour(juce::Colours::lightblue.withAlpha(0.3f));
    g.drawLine(10, 100, getWidth() - 10, 100, 2.0f);
}

void AutoChordAudioProcessorEditor::resized()
{
    auto bounds = getLocalBounds().reduced(10);
    
    // 标题
    titleLabel_.setBounds(bounds.removeFromTop(50));
    bounds.removeFromTop(10);
    
    // 模型选择和加载按钮
    auto controlArea = bounds.removeFromTop(40);
    modelSelector_.setBounds(controlArea.removeFromLeft(250).reduced(5));
    loadModelButton_.setBounds(controlArea.removeFromLeft(150).reduced(5));
    
    bounds.removeFromTop(20);
    
    // 和弦显示
    chordLabel_.setBounds(bounds.removeFromTop(120));
    
    // 置信度
    confidenceLabel_.setBounds(bounds.removeFromTop(40));
    
    bounds.removeFromTop(20);
    
    // 概率分布条
    probabilityBar_.setBounds(bounds);
}

void AutoChordAudioProcessorEditor::timerCallback()
{
    // 更新和弦显示
    auto chord = audioProcessor.getChordRecognizer().getCurrentChord();
    auto confidence = audioProcessor.getChordRecognizer().getCurrentConfidence();
    
    if (chord.isNotEmpty())
    {
        chordLabel_.setText(chord, juce::dontSendNotification);
        confidenceLabel_.setText("Confidence: " + juce::String(confidence * 100.0f, 1) + "%",
                                juce::dontSendNotification);
        
        // 根据置信度改变颜色
        if (confidence > 0.7f)
            chordLabel_.setColour(juce::Label::textColourId, juce::Colours::lightgreen);
        else if (confidence > 0.4f)
            chordLabel_.setColour(juce::Label::textColourId, juce::Colours::yellow);
        else
            chordLabel_.setColour(juce::Label::textColourId, juce::Colours::orange);
    }
    
    // 更新概率分布
    auto probs = audioProcessor.getChordRecognizer().getProbabilities();
    probabilityBar_.setProbabilities(probs);
}
