cmake_minimum_required(VERSION 3.15)

project(AutoChordPlugin VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置JUCE路径
set(JUCE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../JUCE")
add_subdirectory(${JUCE_PATH} juce)

# 设置LibTorch路径
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# 插件目标
juce_add_plugin(AutoChordPlugin
    COMPANY_NAME "LiZhi_AutoChord"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD TRUE
    PLUGIN_MANUFACTURER_CODE Manu
    PLUGIN_CODE Acpg
    FORMATS VST3 Standalone
    PRODUCT_NAME "LiZhi_AutoChord_Demo")

# 源文件
target_sources(AutoChordPlugin
    PRIVATE
        Source/PluginProcessor.cpp
        Source/PluginEditor.cpp
        Source/ChordRecognizer.cpp)

# 编译定义
target_compile_definitions(AutoChordPlugin
    PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0)

# 链接库
target_link_libraries(AutoChordPlugin
    PRIVATE
        juce::juce_audio_utils
        juce::juce_dsp
        "${TORCH_LIBRARIES}"
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags)

# Windows平台特定配置
if(MSVC)
    target_compile_options(AutoChordPlugin PRIVATE /bigobj)
endif()

# 复制模型文件到构建目录
file(GLOB MODEL_FILES 
    "${CMAKE_CURRENT_SOURCE_DIR}/../models_root_stft/*.pt"
    "${CMAKE_CURRENT_SOURCE_DIR}/../models_chord_stft/*.pt"
    "${CMAKE_CURRENT_SOURCE_DIR}/../models_full_stft/*.pt")

foreach(MODEL_FILE ${MODEL_FILES})
    add_custom_command(TARGET AutoChordPlugin POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${MODEL_FILE}
        $<TARGET_FILE_DIR:AutoChordPlugin>)
endforeach()
