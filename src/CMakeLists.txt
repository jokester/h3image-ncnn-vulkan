cmake_policy(SET CMP0091 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0091 NEW)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

cmake_minimum_required(VERSION 3.18)
project(h3image-ncnn-vulkan LANGUAGES CXX)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release CACHE STRING "Choose the type of build" FORCE)
endif()

add_executable(h3image-ncnn-vulkan
    main.cpp
    models.cpp
    runtime.cpp
    allocation_layout.cpp
    memory_budget.cpp
    shape_memory.cpp
    memory_guard.cpp
    tokenizer.cpp
    artifact_hash.cpp
    exact_cache.cpp
    condition_tables.cpp
    modulation_builder.cpp
    image_io.cpp
    vision.cpp
    text_encoder.cpp
    vae.cpp
    vae_encoder.cpp
    transformer.cpp
    prefix_cache.cpp
    kv_cache.cpp
    scheduler.cpp
    pipeline.cpp
    edit_input.cpp
)

option(NCNN_DISABLE_RTTI "Build without C++ RTTI" ON)
option(NCNN_DISABLE_EXCEPTION "Build without C++ exceptions" ON)
include(${CMAKE_CURRENT_SOURCE_DIR}/deps_ncnn.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/deps_codec.cmake)
find_package(Threads REQUIRED)

if(MSVC)
    if(NCNN_DISABLE_RTTI)
        target_compile_options(h3image-ncnn-vulkan PRIVATE /GR-)
    endif()
    if(NCNN_DISABLE_EXCEPTION)
        target_compile_options(h3image-ncnn-vulkan PRIVATE /EHs-c-)
        target_compile_definitions(h3image-ncnn-vulkan PRIVATE _HAS_EXCEPTIONS=0)
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    if(NCNN_DISABLE_RTTI)
        target_compile_options(h3image-ncnn-vulkan PRIVATE -fno-rtti)
    endif()
    if(NCNN_DISABLE_EXCEPTION)
        target_compile_options(h3image-ncnn-vulkan PRIVATE -fno-exceptions)
    endif()
endif()

set(H3IMAGE_LINK_LIBRARIES ncnn ${DEPS_CODEC_LIBRARIES} Threads::Threads)
target_link_libraries(h3image-ncnn-vulkan PRIVATE ${H3IMAGE_LINK_LIBRARIES})

# Bind cache artifacts to the implementation, compiler options and actual ncnn kernels.
file(GLOB H3_OWN_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
file(GLOB_RECURSE H3_KERNEL_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ncnn/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ncnn/src/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/ncnn/src/*.comp")
set(h3_kernel_fingerprint "${CMAKE_CXX_COMPILER_ID}:${CMAKE_CXX_COMPILER_VERSION}:${CMAKE_CXX_FLAGS}:${CMAKE_CXX_FLAGS_RELEASE}:${CMAKE_BUILD_TYPE}")
get_cmake_property(h3_cache_variables CACHE_VARIABLES)
foreach(variable IN LISTS h3_cache_variables)
    get_property(kind CACHE ${variable} PROPERTY TYPE)
    if(variable MATCHES "^NCNN_" AND kind STREQUAL "BOOL")
        string(APPEND h3_kernel_fingerprint ":${variable}=${${variable}}")
    endif()
endforeach()
foreach(source IN LISTS H3_KERNEL_SOURCES)
    file(SHA256 "${source}" source_hash)
    string(APPEND h3_kernel_fingerprint "${source_hash}")
endforeach()
set(h3_condition_fingerprint "${h3_kernel_fingerprint}")
foreach(name IN ITEMS modulation_builder.cpp modulation_builder.h runtime.h condition_tables.cpp condition_tables.h artifact_hash.cpp artifact_hash.h scheduler.cpp scheduler.h h3_config.h)
    file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/${name}" source_hash)
    string(APPEND h3_condition_fingerprint "${source_hash}")
endforeach()
set(h3_cache_fingerprint "${h3_kernel_fingerprint}")
foreach(source IN LISTS H3_OWN_SOURCES)
    file(SHA256 "${source}" source_hash)
    string(APPEND h3_cache_fingerprint "${source_hash}")
endforeach()
string(SHA256 H3_CACHE_BUILD_ID "${h3_cache_fingerprint}")
string(SHA256 H3_CONDITION_BUILD_ID "${h3_condition_fingerprint}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${H3_OWN_SOURCES} ${H3_KERNEL_SOURCES})
if(WIN32)
    target_compile_definitions(h3image-ncnn-vulkan PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
target_compile_definitions(h3image-ncnn-vulkan PRIVATE
    H3_CACHE_BUILD_ID="${H3_CACHE_BUILD_ID}" H3_CONDITION_BUILD_ID="${H3_CONDITION_BUILD_ID}")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(pipeline.cpp scheduler.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
endif()
if(WIN32)
    target_link_libraries(h3image-ncnn-vulkan PRIVATE psapi)
endif()
if(MSVC)
    target_sources(h3image-ncnn-vulkan PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/app.manifest)
    set_property(TARGET h3image-ncnn-vulkan APPEND PROPERTY LINK_OPTIONS "/MANIFEST:EMBED")
endif()

install(TARGETS h3image-ncnn-vulkan RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../README.md ${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE
    DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/h3image-ncnn-vulkan)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../images
    DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/h3image-ncnn-vulkan)
