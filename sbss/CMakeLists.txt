cmake_minimum_required(VERSION 3.9)
project(sbss)

set(CMAKE_CXX_STANDARD 11)

# opencv
find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})
#boost
find_package(Boost 1.58.0 COMPONENTS system filesystem REQUIRED program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
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



file(GLOB src_dir_chamfer "src/chamfer/*.cpp" "src/chamfer/*.hpp")
file(GLOB src_dir_Fdcm "src/Fdcm/*.cpp" "src/Fdcm/*.h")
file(GLOB src_dir_Fitline "src/Fitline/*.cpp" "src/Fitline/*.h")
file(GLOB src_dir_TooN "src/TooN/*.cpp" "src/TooN/*.h" "src/TooN/*.hh")
file(GLOB src_dir_Image "src/Image/*.cpp" "src/Image/*.h")
file(GLOB src_dir_Utils "src/Utils/*.cpp" "src/Utils/*.h" "src/Utils/*.hpp")
file(GLOB src_dir_Utils_tracking "src/Utils/object_tracking_2D/*.cpp" "src/Utils/object_tracking_2D/*.h")
file(GLOB src_dir_Utils_opengl "src/Utils/OpenGL/include/gl/*.cpp" "src/Utils/OpenGL/include/gl/*.h")
file(GLOB src_dir "src/*.cpp" "src/*.hpp" "src/*.h")

# set(SOURCE_FILES como/src/COMO.cpp)
add_executable(sbss ${src_dir} ${src_dir_chamfer} ${src_dir_Utils_tracking} ${src_dir_Utils_opengl} ${src_dir_Image} ${src_dir_Fitline} ${src_dir_TooN} ${src_dir_Utils} ${src_dir_Fdcm})

target_link_libraries( sbss ${OpenCV_LIBS} )
target_link_libraries( sbss ${Boost_LIBRARIES})
target_link_libraries( sbss ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${GLEW_LIBRARIES} ${LAPACK_LIBRARIES})
