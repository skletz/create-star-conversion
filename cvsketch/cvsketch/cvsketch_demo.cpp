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
 * Purpose: The program aims to search videos using a rough sketch.
 * Input/Output: An image, as well as an input directory containing a dataset. As output, the discriptor of the image.
 * @author skletz
 * @version 1.0 12/01/18
 *
 **/


#include <iostream>
#include <boost/program_options.hpp>
#include "src/cvsketch.hpp"

int main(int argc, const char * argv[]) {

    boost::program_options::variables_map args;

    vbs::cvSketch* sketch = new vbs::cvSketch();
    std::cout << sketch->get_info() << std::endl;
    
    try
    {
        args = sketch->process_program_options(argc, argv);
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
