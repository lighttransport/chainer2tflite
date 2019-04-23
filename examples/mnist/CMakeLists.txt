cmake_minimum_required(VERSION 2.8)

set(TFLITE_INCLUDE_DIR "" CACHE PATH "include path to tflite")
set(TFLITE_LIBRARY_DIR "" CACHE PATH "library path to tflite")
set(FLATBUFFERS_INCLUDE_DIR "" CACHE PATH "include path to flatbuffers")

if (TFLITE_INCLUDE_DIR STREQUAL "")
  message(WARNING "TFLITE_INCLUDE_DIR is not set. Build will fail.")
endif()

if (TFLITE_LIBRARY_DIR STREQUAL "")
  message(WARNING "TFLITE_LIBRARY_DIR is not set. Build will fail.")
endif()

set(CMAKE_CXX_STANDARD 11)

set(EXAMPLE_MNIST_SOURCES
  mnist.cc)

find_package(Threads)


include_directories(${TFLITE_INCLUDE_DIR})
include_directories(${FLATBUFFERS_INCLUDE_DIR})
link_directories(${TFLITE_LIBRARY_DIR})

add_executable(mnist ${EXAMPLE_MNIST_SOURCES})

target_link_libraries(mnist tensorflow-lite ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})

# Increase warning level for clang.
IF (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(mnist PRIVATE -Weverything -Wno-system-headers -Werror -Wno-padded -Wno-c++98-compat-pedantic -Wno-documentation -Wno-documentation-unknown-command -Wno-reserved-id-macro)
ENDIF ()

