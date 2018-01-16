#ifndef _CVSKETCH_HPP_
#define  _CVSKETCH_HPP_

//
//  cvSketch
//  cvsketch.hpp
//  cvsketch
//
//  Created by Sabrina Kletz on 12.01.18.
//  Copyright © 2018 Sabrina Kletz. All rights reserved.
//

#include <string>
#include <iostream>
#include <sstream>
#include <boost/version.hpp>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include "segmentation.hpp"

#if defined(WIN32) || defined(_WIN32)
#define DIRECTORY_SEPARATOR "\\"
#else
#define DIRECTORY_SEPARATOR "/"
#endif

namespace vbs {
    
    class cvSketch
    {
        
    public:
		bool verbose = false;
		bool display = false;
        std::string input;
		std::string output;

        cvSketch();
        
        /**
         *
         *
         */
        std::string getInfo();
        
        /**
         *
         *
         */
        std::string help(const boost::program_options::options_description& desc);
        
        /**
         *
         *
         */
        boost::program_options::variables_map processProgramOptions(const int argc, const char *const argv[]);
        
        /**
         *
         *
         */
        bool init(boost::program_options::variables_map _args);
        
        /**
        *
        *
        */
        void run();
        
        //void on_trackbar_colorReduction_kMeans(const int kvalue, void* data);
        
        /**
         *
         *
         */
        void testColorSegmentation(cv::Mat& image);

        static void reduceColors(cv::Mat& image, int kvalue, cv::Mat& output);
        
        static void extractSuperpixels(cv::Mat& image, cv::Mat& output, cv::Mat& mask, int& num_output, int num_superpixels, int num_levels, int prior, int num_histogram_bins, bool double_step, int num_iterations);
        
        static void getColorchart(cv::Mat& image, std::map<cv::Vec3b, int, lessVec3b>& palette, cv::Mat& output, int chartwidth, int chartheight);
        
        static void quantizeColors(cv::Mat& image, cv::Mat& lables, int num_labels, cv::Mat& output, std::map<cv::Vec3b, int, lessVec3b> colorpalette);
        
		/**
		* @brief makeCanvas Makes composite image from the given images
		* @param vecMat Vector of Images.
		* @param windowHeight The height of the new composite image to be formed.
		* @param nRows Number of rows of images. (Number of columns will be calculated
		*              depending on the value of total number of images).
		* @return new composite image.
		*/
		cv::Mat makeCanvas(std::vector<cv::Mat>& vecMat, int windowHeight, int nRows);
		bool storeImage(std::string originalfile, std::string append, std::string extension, cv::Mat& image);
        ~cvSketch();

    };
}

#endif //_CVSKETCH_HPP_
