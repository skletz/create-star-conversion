#ifndef _MATCHING_HPP_
#define  _MATCHING_HPP_

//
//  Matching
//  matching.cpp
//  cvsketch
//
//  Created by Sabrina Kletz on 12.01.18.
//  Copyright © 2018 Sabrina Kletz. All rights reserved.
//

#include <opencv2/opencv.hpp>
namespace vbs
{
	struct Match {
		std::string path;
		float dist;
		cv::Mat image;
	};

	class Matching {

	public:

		bool compareMatchesByDist(const Match &a, const Match &b);
		static std::vector<cv::Point> getSimpleContours(const cv::Mat& currentQuery, int points = 1500);
		static void drawPoints(cv::Mat bg, std::vector<cv::Point> cont, cv::Mat& output);

	};
}


#endif //_MATCHING_HPP_