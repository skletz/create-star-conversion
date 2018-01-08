cmake_minimum_required(VERSION 3.9)
project(sbss)

set(CMAKE_CXX_STANDARD 11)

# opencv
find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})
#boost
find_package(Boost 1.58.0 COMPONENTS system filesystem REQUIRED program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

## chamfer
file(GLOB src_dir_chamfer "src/chamfer/*.cpp" "src/chamfer/*.hpp")

## FDCM_2
file(GLOB src_dir_Fdcm_2 "src/FDCM_2/Fdcm/*.cpp" "src/FDCM_2/Fdcm/*.h")
file(GLOB src_dir_Fdcm_2_Fitline "src/FDCM_2/Fitline/*.cpp" "src/FDCM_2/Fitline/*.h")
file(GLOB src_dir_Fdcm_2_Image "src/FDCM_2/Image/*.cpp" "src/FDCM_2/Image/*.h")
# add dir to includes
include_directories("src/FDCM_2/")


## Everything else
file(GLOB src_dir "src/*.cpp" "src/*.hpp" "src/*.h" "src/Utils/*.cpp" "src/Utils/*.hpp")

## source files
# set(SOURCE_FILES como/src/COMO.cpp)
add_executable(sbss ${src_dir} ${src_dir_chamfer} ${src_dir_Fdcm_2} ${src_dir_Fdcm_2_Fitline} ${src_dir_Fdcm_2_Image})

## LINKING
target_link_libraries( sbss ${OpenCV_LIBS} ${Boost_LIBRARIES})
