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
		std::string searchin;

        double execution_time = 0.0;

        int pad_top = 250;
        int pad_left = 50;
        int nr_input_images = -1;
        int top_kresults = 50;


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

        void searchimagein(std::string query_path);

        /**
         *
         *
         */
        void testColorSegmentation(cv::Mat& image, cv::Mat& colorSegments, cv::Mat& colorLabels, std::map<cv::Vec3b, int, lessVec3b>& palette);

        /**
         *
         *
         */
        void describeColorSegmentation(cv::Mat& image, cv::Mat& colorSegments, cv::Mat& colorLabels, std::map<cv::Vec3b, int, lessVec3b>& palette, cv::Mat& descriptors);


        static void reduceColors(const cv::Mat& image, int kvalue, cv::Mat& output);

        static void extractSuperpixels(cv::Mat& image, cv::Mat& output, cv::Mat& mask, int& num_output, int num_superpixels, int num_levels, int prior, int num_histogram_bins, bool double_step, int num_iterations);

        static void getColorchart(std::map<cv::Vec3b, int, lessVec3b>& palette, cv::Mat& output, int chartwidth, int chartheight, int area);

        static void getColorchart(std::vector<std::pair<cv::Vec3b, int>>& palette, cv::Mat& output, int chartwidth, int chartheight, int area);

        //static void convertDefault(std::map<cv::Vec3b, int, lessVec3b>& palette, cv::Mat& output, int chartwidth, int chartheight);

        static void getDefaultColorchart(std::map<cv::Vec3b, int, lessVec3b>& palette, cv::Mat& output, int chartwidth, int chartheight);

        static void quantizeColors(cv::Mat& image, cv::Mat& lables, int num_labels, cv::Mat& output, std::map<cv::Vec3b, int, lessVec3b> colorpalette);

        void show_image(const cv::Mat& image, std::string winname, int x, int y);

        void process_image(const cv::Mat& image, int width, int height, int colors, cv::Mat& image_withbarchart, std::vector<std::pair<cv::Vec3b, int>>& sorted_colorpalette);




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
