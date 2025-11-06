cmake_minimum_required(VERSION 3.15)

project(AutoChordPlugin VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置JUCE路径
set(JUCE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../JUCE")
add_subdirectory(${JUCE_PATH} juce)

# 设置LibTorch路径 (Windows使用libtorch_win)
if(WIN32)
    set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../libtorch_win")
else()
    set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../libtorch_mac")
endif()
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
    FORMATS AU VST3 Standalone
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

# 复制模型文件到各个插件格式目录
file(GLOB MODEL_FILES 
    "${CMAKE_CURRENT_SOURCE_DIR}/../exported_models/*.pt")

if(MODEL_FILES)
    message(STATUS "Found model files: ${MODEL_FILES}")
    
    # 复制到 VST3
    foreach(MODEL_FILE ${MODEL_FILES})
        add_custom_command(TARGET AutoChordPlugin POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${MODEL_FILE}
            "${CMAKE_CURRENT_BINARY_DIR}/AutoChordPlugin_artefacts/VST3/LiZhi_AutoChord_Demo.vst3/Contents/MacOS/"
            COMMENT "Copying ${MODEL_FILE} to VST3")
    endforeach()
    
    # 复制到 Standalone
    foreach(MODEL_FILE ${MODEL_FILES})
        add_custom_command(TARGET AutoChordPlugin POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${MODEL_FILE}
            "${CMAKE_CURRENT_BINARY_DIR}/AutoChordPlugin_artefacts/Standalone/LiZhi_AutoChord_Demo.app/Contents/MacOS/"
            COMMENT "Copying ${MODEL_FILE} to Standalone")
    endforeach()
    
    # 复制到 AU（如果启用）
    foreach(MODEL_FILE ${MODEL_FILES})
        add_custom_command(TARGET AutoChordPlugin POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${MODEL_FILE}
            "${CMAKE_CURRENT_BINARY_DIR}/AutoChordPlugin_artefacts/AU/LiZhi_AutoChord_Demo.component/Contents/MacOS/"
            COMMENT "Copying ${MODEL_FILE} to AU")
    endforeach()
else()
    message(WARNING "No model files found in ../exported_models/. Run 'python export_models_for_juce.py --export_all' first.")
endif()
