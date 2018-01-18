#ifndef _MATCHING_HPP_
#define  _MATCHING_HPP_

//
//  Matching
//  matching.cpp
//  cvsketch
//
//  Created by Sabrina Kletz on 12.01.18.
//  Copyright � 2018 Sabrina Kletz. All rights reserved.
//

#include <opencv2/opencv.hpp>
#include "../libs/CIEDE2000/CIEDE2000.h"

namespace vbs
{
	struct Match {
		std::string path;
		float dist;
		cv::Mat image;
	};

	class Matching {

	public:
		struct Match {
			std::string path;
			float dist;
			cv::Mat image;
		};




		bool compareMatchesByDist(const Match &a, const Match &b);
		static std::vector<cv::Point> getSimpleContours(const cv::Mat& currentQuery, int points = 1500);
		static void drawPoints(cv::Mat bg, std::vector<cv::Point> cont, cv::Mat& output);

    static double compareWithEuclid(const std::vector<std::pair<cv::Vec3b, float>>& c1, const std::vector<std::pair<cv::Vec3b, float>>& c2);
		static double compareWithCIEDE(const std::vector<std::pair<cv::Vec3b, float>>& c1, const std::vector<std::pair<cv::Vec3b, float>>& c2);
		static double compareWithOCCD(const std::vector<std::pair<cv::Vec3b, float>>& c1, const std::vector<std::pair<cv::Vec3b, float>>& c2, int area);

		static void show_image(const cv::Mat& image, const std::string winname, int x, int y);
		static void print_stack(const std::vector<std::pair<cv::Vec3b, float>>& colorpalette, cv::Mat& image);
		static void sortPaletteByArea(std::vector<std::pair<cv::Vec3b, int>> input, std::vector<std::pair<cv::Vec3b, int>>& output);


	};
}


#endif //_MATCHING_HPP_
