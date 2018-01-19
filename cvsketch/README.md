Color-Sketch-based video search
===
Color-Sketch-based video search is a way to sketch scenes in mind with rough color segments and search videos containing this color and shape composition.

The following directory is a test-bed and currently contains:
* Color reduction using k-means
* Superpixels Extracted via Energy-Driven Sampling (SEEDS)
* Refinement of superpixels using k-color's

![Alt text](snapshots/settings_color-segmentation_1.png?raw=true "Settings of color segmentation refinement")

![Alt text](snapshots/retrieval_weighted_OCCD.png?raw=true "Retrieval Results using OCCD")

## Related Work


### Datasets
Sun et al. in [Indexing billions of images for sketch-based retrieval](https://dl.acm.org/citation.cfm?id=2502281) uses the data set, described by Eitz et al.; Eitz et al. introduced it in [Sketch-based Image Retrieval: Benchmark and Bag-of-Features Descriptors](http://ieeexplore.ieee.org/document/5674030/). The data set can be downloaded from [Benchmark dataset](http://cybertron.cg.tu-berlin.de/eitz/tvcg_benchmark/index.html). It contains hand drawn sketches and for each sketch there are several corresponding images.

Samples to test shapes are available here: [OpenCV Shape Sample Data](https://github.com/opencv/opencv/tree/master/samples/data/shape_sample)

## Background

Searching images by selecting up to 5 colors: [TinEye - Multicolr: Search by color](http://labs.tineye.com/multicolr/)

Superpixel benchmark on [GitHub - Superpixels: An Evaluation of the State-of-the-Art](https://github.com/davidstutz/superpixel-benchmark)

Simple linear iterative clustering (SLIC), Reference: [SLIC Superpixels](http://www.kev-smith.com/papers/SLIC_Superpixels.pdf) is based on L,a,b and x,y coordinates and finally clusters this vector.


[Superpixel Algorithms: Overview and Comparison](http://davidstutz.de/superpixel-algorithms-overview-comparison/)


[ContentBased Image Retrieval System by Hierarchical Color Image Region Segmentation
](https://www.csie.ntu.edu.tw/~fuh/personal/ContentBasedImageRetrievalSystembyHierarchical.pdf)

[Comparing images using color blob approach](http://pe.org.pl/articles/2013/2a/11.pdf)

[Shape description with Hu-Moments](http://slideplayer.com/10871800/39/images/6/Poses+and+gestures+More+shape+properties.jpg)


# Setting up dev environment
The following describes how to setup this test-bed using Mac OS Sierra 10.12.3 and Xcode 8.3.3, or with Windows 10 and Visual Studio 2015:

## Mac OS and Xcode
Install Xcode.

### Link OpenCV
Install OpenCV.
Project Properties:
* Add to Header Search Paths: /usr/local/include
* Add to Library Search Path: /usr/local/lib
* Add to Other Linker Flags: -lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_dpm -lopencv_face -lopencv_photo -lopencv_fuzzy -lopencv_img_hash -lopencv_line_descriptor -lopencv_optflow -lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrapping -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_dnn -lopencv_plot -lopencv_xfeatures2d -lopencv_shape -lopencv_video -lopencv_ml -lopencv_ximgproc -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_flann -lopencv_xobjdetect -lopencv_imgcodecs -lopencv_objdetect -lopencv_xphoto -lopencv_imgproc -lopencv_core

### Boost
Install Boost.

### Errors
Possible errors:
* Not a Doxygen trailing comment: Go to Build Settings and search for Documentation Comments and set as No.

## Windows and Visual Studio
Install Visual Studio 2015.

### Link OpenCV

[General Tutorial](https://inside.mines.edu/~whoff/courses/EENG510/lectures/other/CompilingOpenCV.pdf) tested with the version 3.3.1; downloaded from GitHub.

Install OpenCV.
* Clone OpenCV Repository from [>>GitHub/OpenCV](https://github.com/opencv/opencv)
* Clone OpenCV Contrib Repository from [>>GitHub/OpenCV_Contrib](https://github.com/opencv/opencv_contrib)
* Save both to C:/libraries
    * C:/libraries/opencv
    * C:/libraries/opencv_contrib
* Install [>>Python 2.7](https://www.python.org/downloads/) to C:/libraries/python27
* Install [>>Python 3.6](https://www.python.org/downloads/) to C:/libraries/python36
    * Go to C:/libraries/python36 and install numby
* Install [>>CMake](https://cmake.org/download/)
* Open CMake-Gui with administrative rights
    * Source Code: C:/libraries/opencv
    * Build Binaries: C:/libraries/opencv-build
    * Press configure
    * Select Visual 14 2015 Win64
    * Add OPENCV_EXTRA_MODULES_PATH: C:/libraries/opencv_contrib/modules
    * Add PYTHON3_EXECUTABLE: C:/Python36
    * Check BUILD_EXAMPLES
    * Press generate
    * Press open project
* Build target "ALL_BUILD" in Debug und Release Mode (This will create all "lib" and "dll" files)
* Build target "INSTALL" for both: Debug and Release (This combines all the lib and dll files into a single "lib" and a single "bin" folder)
* Open C:\libraries\opencv-build\install
* Copy the content of the directory to C:\libraries\opencv331
* Set the system environment variables for development:
    * Create a new System variable "OpenCV_DIR"; value="C:\libraries\opencv331"
    * Add "%OpenCV_DIR%\x64\vc14\bin" to the variable "Path"
* Setting project properties in Visual Studio 2015:
    * Add "$(OpenCV_DIR)\include" to additional include directory (C/C++ -> General)
    * Add "$(OpenCV_DIR)\x64\vc14\lib" to additional library directories (Linker -> General)
    * Add "opencv_ts300d.lib;opencv_world300d.lib" to additional dependencies (Linker -> Input)
    * OR each individual lib: opencv_aruco331d.lib; opencv_bgsegm331d.lib; opencv_bioinspired331d.lib; opencv_calib3d331d.lib; opencv_ccalib331d.lib; opencv_core331d.lib; opencv_datasets331d.lib; opencv_dnn331d.lib; opencv_dpm331d.lib; opencv_face331d.lib; opencv_features2d331d.lib; opencv_flann331d.lib; opencv_fuzzy331d.lib; opencv_highgui331d.lib; opencv_img_hash331d.lib; opencv_imgcodecs331d.lib; opencv_imgproc331d.lib; opencv_line_descriptor331d.lib; opencv_ml331d.lib; opencv_objdetect331d.lib; opencv_optflow331d.lib; opencv_phase_unwrapping331d.lib; opencv_photo331d.lib; opencv_plot331d.lib; opencv_reg331d.lib; opencv_rgbd331d.lib; opencv_saliency331d.lib; opencv_shape331d.lib; opencv_stereo331d.lib; opencv_stitching331d.lib; opencv_structured_light331d.lib; opencv_superres331d.lib; opencv_surface_matching331d.lib; opencv_text331d.lib; opencv_tracking331d.lib; opencv_video331d.lib; opencv_videoio331d.lib; opencv_videostab331d.lib; opencv_xfeatures2d331d.lib; opencv_ximgproc331d.lib; opencv_xobjdetect331d.lib; opencv_xphoto331d.lib
* Disable further security warnings:
    * Preprocessor Definitions add "_CRT_SECURE_NO_WARNINGS" (C/C++ -> Preprocessor)


### Link Boost
Install Boost.
* Download Boost C++ library for Windows >> http://www.boost.org/users/download/; Choose the current version; tested with v1.65
* Extract its content to C:/libraries/boost165
* Open C:\Program Files (x86)\Microsoft Visual Studio 12.0\Common7\Tools\Shortcuts\Developer-Eingabeaufforderung für VS2015.exe
* Go to cd C:/libraries/boost165
* Type bootstrap  >> "Building Boost.Build engine"
* Type .\b2 --build-dir="C:\boost165" toolset=msvc-14.0 --build-type=complete --abbreviate-paths architecture=x86 address-model=64 install -j4 >> Build 64-bit Version for VS 2015 - Important is that the architecture have to be x86
* Set the system environment variables:
    * Create a new System variable "Boost_DIR"; value="C:\libraries\boost165"
    * Add "%Boost_DIR%\lib" to the variable "Path"
* Setting project properties in Visual Studio:
	* Add $(Boost_DIR) to additional include directory (C/C++ -> General)
	* Add $(Boost_DIR)\lib to additional library directories (Linker -> General)

### General Settings
* Add /FS to additional commandline options (C/C++ -> commandline input) (to all projects in the solution-explorer)
* Add _CRT_SECURE_NO_WARNINGS, DEBUG=true to preprocessor input
