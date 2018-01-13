/** CVSKETCH (Version 1.0) ******************************
 * ******************************************************
 *       _    _      ()_()
 *      | |  | |    |(o o)
 *   ___| | _| | ooO--`o'--Ooo
 *  / __| |/ / |/ _ \ __|_  /
 *  \__ \   <| |  __/ |_ / /
 *  |___/_|\_\_|\___|\__/___|
 *
 * ******************************************************
 * Purpose:
 * Input/Output:
 * @author skletz
 * @version 1.0 12/01/18
 *
 **/


#include "opencv2/opencv.hpp"
#include "boost/version.hpp"
//#include "opencv2/imgcodecs.hpp"
//#include "opencv2/highgui.hpp"
//#include "opencv2/imgproc.hpp"
//#include <opencv2/core/utility.hpp>

#include <iostream>
#include "src/cvsketch.hpp"
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

//std::string ENV_DIR="/Users/skletz/Dropbox/Programming/CUPCakes/vbssketch/cvsketch";
std::string ENV_DIR = "E:\\Dropbox\\Programming\\CUPCakes\\vbssketch\\cvsketch";

static void help()
{
    printf("\n"
           "This program demonstrates methods for shape comparisson based on shape context\n"
           "Run the program providing a number between 0 and 48 for selecting a sketch in the folder ../sketches/.\n"
           "Call\n"
           "./main [number between 0 and 48, 0 default]\n\n");
}

int main(int argc, const char * argv[]) {

	boost::filesystem::path p(ENV_DIR);
    std::string path = p.string() + "/data/sketch_sample";
    std::string path_sketches = path + "/sketches/";
    std::string path_images = path + "/images/";
    
    boost::program_options::variables_map args;
    
   
    vbs::cvSketch* sketch = new vbs::cvSketch();
    std::cout << sketch->getInfo() << std::endl;
    
    try
    {
        args = sketch->processProgramOptions(argc, argv);
    }
    catch (std::exception& e)
    {
        std::cerr << "ERROR: Programm options cannot be used!" << std::endl;
        std::cerr << e.what() << std::endl;
    }
    
    sketch->init(args);
    
    sketch->run();
    delete sketch;

    return 0;
}
