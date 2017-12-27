cmake_minimum_required(VERSION 3.9)
project(sbss)

set(CMAKE_CXX_STANDARD 11)

find_package( OpenCV REQUIRED )

include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB src_dir "src/*.cpp")

# set(SOURCE_FILES como/src/COMO.cpp)
add_executable(sbss ${src_dir})

target_link_libraries( sbss ${OpenCV_LIBS} )
