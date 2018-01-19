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

	struct SettingsKmeansCluster
	{
		std::string winname;
		int kvalue;
		int kvalue_max;
		SettingsKmeansCluster(){}
	};

	struct SettingsSuperpixelSEEDS
	{
		std::string winname;
		int num_superpixels;
		int num_superpixels_max;
		int prior;
		int prior_max;
		int num_levels;
		int num_levels_max;
		int num_iterations;
		int num_iterations_max;
		int double_step;
		int num_histogram_bins;
		int num_histogram_bins_max;
		SettingsSuperpixelSEEDS(){}
	};

	struct KmeansClusterSEEDSExchange
	{
		//dependent images
		std::string winnameQuantizedColors;
		std::string winnameColorchart;

		cv::Mat image;
		cv::Mat reduced_image;
		cv::Mat superpixel_image;
		cv::Mat mask;
		cv::Mat labels;
		int num_labels;

		std::map<cv::Vec3b, int, vbs::lessVec3b> colorpalette;
		std::vector<std::pair<cv::Vec3b, float>> colors;
	};

    class cvSketch
    {

    public:
		bool verbose;
		bool display;

		int max_width;
		int max_height;
        std::string in_query;
		std::string in_dataset;
		std::string output;


        double execution_time = 0.0;

        int pad_top = 250;
        int pad_left = 50;
        int nr_input_images = -1;
        int top_kresults = 50;

        vbs::Segmentation* segmentation;
		SettingsKmeansCluster* set_kmeans;
		SettingsSuperpixelSEEDS* set_seeds;
		KmeansClusterSEEDSExchange* set_exchange;

		std::vector<std::pair<cv::Vec3b, int>> out_colorpalette;

        cvSketch(bool _verbose = true, bool _display = false, int _max_width = 352, int _max_height = 240);

        /**
         * Returns the aim of the programm
         *
         */
        std::string get_info();

        /**
         * Returns the formatted help message, description of program options
         *
         */
        std::string help(const boost::program_options::options_description& _desc);

        /**
         * Process program options
         */
        boost::program_options::variables_map process_program_options(const int argc, const char *const argv[]);

        /**
         * Init variables using the program options
         *
         */
        bool init(boost::program_options::variables_map _args);

        /**
         * Run the program specified by its program options
         *
         */
        void run();

		/**
		 * Search a given image in an set of images
		 *
		 */
        void search_image(std::string query_path, std::string dataset_path);


        /**
         * Callback for setting up kmeans clustering
         *
         */
        static void on_trackbar_colorReduction_kMeans(int, void* _object);


        /**
         * Callback for setting up SEEDS superpixel
         *
         */
        static void on_trackbar_superpixel_SEEDS(int, void* _object);

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


        void reduce_colors(const cv::Mat& image, int kvalue, cv::Mat& output);

        void extract_superpixels(cv::Mat& image, cv::Mat& output, cv::Mat& mask, int& num_output, int num_superpixels, int num_levels, int prior, int num_histogram_bins, bool double_step, int num_iterations);

        static void getColorchart(std::map<cv::Vec3b, int, lessVec3b>& palette, cv::Mat& output, int chartwidth, int chartheight, int area);

        static void getColorchart(std::vector<std::pair<cv::Vec3b, int>>& palette, cv::Mat& output, int chartwidth, int chartheight, int area);

        void get_colorchart(std::vector<std::pair<cv::Vec3b, float>>& colors, cv::Mat& output, int chartwidth, int chartheight, int area);


        //static void convertDefault(std::map<cv::Vec3b, int, lessVec3b>& palette, cv::Mat& output, int chartwidth, int chartheight);

        static void getDefaultColorchart(std::map<cv::Vec3b, int, lessVec3b>& palette, cv::Mat& output, int chartwidth, int chartheight);

        void quantize_colors(const cv::Mat& image, cv::Mat& lables, int num_labels, cv::Mat& output, std::vector<std::pair<cv::Vec3b, float>>& colorpalette);

        void show_image(const cv::Mat& image, std::string winname, int x = -1, int y = -1);

        void process_image(const cv::Mat& image, int width, int height, int colors, cv::Mat& image_withbarchart, std::vector<std::pair<cv::Vec3b, int>>& sorted_colorpalette);



		/**
		* @brief makeCanvas Makes composite image from the given images
		* @param vecMat Vector of Images.
		* @param windowHeight The height of the new composite image to be formed.
		* @param nRows Number of rows of images. (Number of columns will be calculated
		*              depending on the value of total number of images).
		* @return new composite image.
		*/
		cv::Mat make_canvas(std::vector<cv::Mat>& vecMat, int windowHeight, int nRows);

		bool store_image(std::string originalfile, std::string append, std::string extension, cv::Mat& image);

    void set_label(cv::Mat& _im, const std::string _label, const cv::Point& _point, float _scale = 0.4f);

        ~cvSketch();

    };

}

#endif //_CVSKETCH_HPP_
