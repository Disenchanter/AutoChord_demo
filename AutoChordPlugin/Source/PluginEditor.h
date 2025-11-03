#pragma once

#include <JuceHeader.h>
#include "ChordRecognizer.h"

class AutoChordAudioProcessor;

//==============================================================================
class AutoChordAudioProcessorEditor : public juce::AudioProcessorEditor,
                                       private juce::Timer
{
public:
    AutoChordAudioProcessorEditor(AutoChordAudioProcessor&);
    ~AutoChordAudioProcessorEditor() override;

    //==============================================================================
    void paint(juce::Graphics&) override;
    void resized() override;

private:
    void timerCallback() override;
    
    AutoChordAudioProcessor& audioProcessor;
    
    // UI组件
    juce::Label titleLabel_;
    juce::Label chordLabel_;
    juce::Label confidenceLabel_;
    
    juce::ComboBox modelSelector_;
    juce::TextButton loadModelButton_;
    
    // 概率分布显示
    juce::TextEditor probabilitiesDisplay_;
    
    // 可视化组件
    class ProbabilityBar : public juce::Component
    {
    public:
        void setProbabilities(const std::vector<std::pair<juce::String, float>>& probs)
        {
            probabilities_ = probs;
            repaint();
        }
        
        void paint(juce::Graphics& g) override
        {
            g.fillAll(juce::Colours::darkgrey);
            
            auto bounds = getLocalBounds().reduced(5);
            int barHeight = 20;
            int spacing = 5;
            
            int displayCount = juce::jmin(10, static_cast<int>(probabilities_.size()));
            
            for (int i = 0; i < displayCount; ++i)
            {
                auto& [label, prob] = probabilities_[i];
                
                int y = i * (barHeight + spacing);
                auto barBounds = bounds.removeFromTop(barHeight);
                
                // 绘制标签
                auto labelBounds = barBounds.removeFromLeft(80);
                g.setColour(juce::Colours::white);
                g.drawText(label, labelBounds, juce::Justification::centredLeft);
                
                // 绘制进度条
                auto barArea = barBounds.reduced(2);
                g.setColour(juce::Colours::darkgrey.brighter());
                g.fillRect(barArea);
                
                float barWidth = barArea.getWidth() * prob;
                auto filledBar = barArea.withWidth(static_cast<int>(barWidth));
                
                // 颜色梯度
                juce::Colour barColour = i == 0 ? juce::Colours::green : juce::Colours::lightblue;
                g.setColour(barColour);
                g.fillRect(filledBar);
                
                // 概率文本
                g.setColour(juce::Colours::white);
                g.drawText(juce::String(prob * 100.0f, 1) + "%",
                          barBounds, juce::Justification::centredRight);
            }
        }
        
    private:
        std::vector<std::pair<juce::String, float>> probabilities_;
    };
    
    ProbabilityBar probabilityBar_;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AutoChordAudioProcessorEditor)
};
