# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.13)  # The requirement comes from thirdparty/openvino_contrib/modules/custom_operations/CMakeLists.txt
project(GenAI)

list(APPEND CUSTOM_OPERATIONS tokenizer)
add_subdirectory(thirdparty)  # Build openvino_tokenizers
if(NOT MSVC)  # # Disable the warning from user_ov_extensions
    target_compile_options(sentencepiece-static PRIVATE -Wno-stringop-overflow)
endif()

add_subdirectory(llm/chatglm_cpp)
target_compile_definitions(chatglm PRIVATE OPENVINO_TOKENIZERS_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")
target_link_libraries(chatglm PRIVATE openvino_tokenizers)

add_subdirectory(llm/llama_cpp)
target_compile_definitions(llama PRIVATE OPENVINO_TOKENIZERS_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")
target_link_libraries(llama PRIVATE openvino_tokenizers)

add_subdirectory(llm/minicpm_cpp)
target_compile_definitions(minicpm PRIVATE OPENVINO_TOKENIZERS_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")
target_link_libraries(minicpm PRIVATE openvino_tokenizers)

add_subdirectory(llm/minicpm_longtest)
target_compile_definitions(minicpm_longtest PRIVATE OPENVINO_TOKENIZERS_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")
target_link_libraries(minicpm_longtest PRIVATE openvino_tokenizers)
