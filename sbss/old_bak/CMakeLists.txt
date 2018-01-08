cmake_minimum_required(VERSION 3.9)
project(sbss)

set(CMAKE_CXX_STANDARD 11)

# opencv
find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})
#boost
find_package(Boost 1.58.0 COMPONENTS system filesystem REQUIRED program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

## FDCM_1
#opengl, glut and glew
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS})
#glew
find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})
# lapack and blas
find_package(LAPACK REQUIRED)
include_directories( ${LAPACK_INCLUDE_DIRECTORIES})
#find_package(BLAS REQUIRED)
#include_directories( ${BLAS_INCLUDE_DIRECTORIES})

file(GLOB src_dir_Fdcm "src/FDCM_1/Fdcm/*.cpp" "src/FDCM_1/Fdcm/*.h")
file(GLOB src_dir_Fitline "src/FDCM_1/Fitline/*.cpp" "src/FDCM_1/Fitline/*.h")
file(GLOB src_dir_TooN "src/FDCM_1/TooN/*.cpp" "src/FDCM_1/TooN/*.h" "src/FDCM_1/TooN/*.hh")
file(GLOB src_dir_Image "src/FDCM_1/Image/*.cpp" "src/FDCM_1/Image/*.h")
file(GLOB src_dir_Utils "src/FDCM_1/Utils/*.cpp" "src/FDCM_1/Utils/*.h" "src/FDCM_1/Utils/*.hpp")
file(GLOB src_dir_Utils_tracking "src/FDCM_1/Utils/object_tracking_2D/*.cpp" "src/FDCM_1/Utils/object_tracking_2D/*.h")
file(GLOB src_dir_Utils_opengl "src/FDCM_1/Utils/OpenGL/include/gl/*.cpp" "src/FDCM_1/Utils/OpenGL/include/gl/*.h")

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
add_executable(sbss ${src_dir} ${src_dir_chamfer} ${src_dir_Utils_tracking} ${src_dir_Utils_opengl} ${src_dir_Image} ${src_dir_Fitline} ${src_dir_TooN} ${src_dir_Utils} ${src_dir_Fdcm} ${src_dir_Fdcm_2} ${src_dir_Fdcm_2_Fitline} ${src_dir_Fdcm_2_Image})

## LINKING
target_link_libraries( sbss ${OpenCV_LIBS} ${Boost_LIBRARIES})
# FDCM_1
target_link_libraries( sbss ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${GLEW_LIBRARIES} ${LAPACK_LIBRARIES})
