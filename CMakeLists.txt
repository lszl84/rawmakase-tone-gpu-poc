cmake_minimum_required(VERSION 3.24)
project(vkcontrast_poc CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Vulkan REQUIRED)
find_package(glfw3 3.4 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBRAW REQUIRED IMPORTED_TARGET libraw_r)
pkg_check_modules(SQLITE REQUIRED IMPORTED_TARGET sqlite3)
find_program(GLSLC glslc REQUIRED)

# RAWmakase's measured tone tables, read from its checkout at build time.
set(RAWMAKASE_DIR "$ENV{HOME}/Developer/rawmakase" CACHE PATH "RAWmakase checkout")
set(GEN_DIR ${CMAKE_BINARY_DIR}/gen)
set(TONE_DATA
  ${RAWMAKASE_DIR}/crates/rawmakase-engine/src/develop/basic_tone_data.rs
  ${RAWMAKASE_DIR}/crates/rawmakase-engine/src/develop/local_tone_data.rs)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_custom_command(
  OUTPUT ${GEN_DIR}/tables.h ${GEN_DIR}/tables.glsl
  COMMAND Python3::Interpreter ${CMAKE_SOURCE_DIR}/tools/gen_tables.py ${RAWMAKASE_DIR} ${GEN_DIR}
  DEPENDS ${CMAKE_SOURCE_DIR}/tools/gen_tables.py ${TONE_DATA}
  VERBATIM)

# Shaders are compiled to SPIR-V at build time and embedded in the binary.
set(SHADERS fullscreen.vert develop.frag resample.comp measure_tone.comp select_hist.comp
            select_scan.comp box_rows.comp box_cols.comp pivot_blocks.comp pivot_sort.comp build_lut.comp tone_display.comp)
set(SPV_DIR ${CMAKE_BINARY_DIR}/spv)
file(MAKE_DIRECTORY ${SPV_DIR})
foreach(shader ${SHADERS})
  set(out ${SPV_DIR}/${shader}.inc)
  add_custom_command(
    OUTPUT ${out}
    COMMAND ${GLSLC} -O --target-env=vulkan1.3 -mfmt=num -I ${CMAKE_SOURCE_DIR}/shaders -I ${GEN_DIR}
            ${CMAKE_SOURCE_DIR}/shaders/${shader} -o ${out}
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/${shader} ${CMAKE_SOURCE_DIR}/shaders/common.glsl ${GEN_DIR}/tables.glsl
    VERBATIM)
  list(APPEND SPV ${out})
endforeach()
add_custom_target(shaders DEPENDS ${SPV} ${GEN_DIR}/tables.h)

set(IMGUI ${CMAKE_SOURCE_DIR}/third_party/imgui)
add_executable(vkcontrast
  src/main.cpp
  ${IMGUI}/imgui.cpp
  ${IMGUI}/imgui_draw.cpp
  ${IMGUI}/imgui_tables.cpp
  ${IMGUI}/imgui_widgets.cpp
  ${IMGUI}/backends/imgui_impl_glfw.cpp
  ${IMGUI}/backends/imgui_impl_vulkan.cpp)
add_dependencies(vkcontrast shaders)
target_include_directories(vkcontrast PRIVATE ${IMGUI} ${IMGUI}/backends ${SPV_DIR} ${GEN_DIR})
target_link_libraries(vkcontrast PRIVATE Vulkan::Vulkan glfw PkgConfig::LIBRAW PkgConfig::SQLITE)
target_compile_options(vkcontrast PRIVATE -march=native)
