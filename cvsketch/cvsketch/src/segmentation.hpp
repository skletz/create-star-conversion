#ifndef _SEGMENTATION_HPP_
#define  _SEGMENTATION_HPP_

//
//  Segmentation
//  segmentation.cpp
//  cvsketch
//
//  Created by Sabrina Kletz on 12.01.18.
//  Copyright © 2018 Sabrina Kletz. All rights reserved.
//

#include <opencv2/opencv.hpp>
namespace vbs
{
	struct lessVec3b
	{
		bool operator()(const cv::Vec3b& lhs, const cv::Vec3b& rhs) const
		{
			return (lhs[0] != rhs[0]) ? (lhs[0] < rhs[0]) : ((lhs[1] != rhs[1]) ? (lhs[1] < rhs[1]) : (lhs[2] < rhs[2]));
		}
	};

	class Segmentation {

	public:
		//Color reduction
		static void reduceColor_Quantization(const cv::Mat3b& src, cv::Mat3b& dst);
		static void reduceColor_kmeans(const cv::Mat3b& src, cv::Mat3b& dst, int k = 3);
		static void reduceColor_spatialkmeans(const cv::Mat3b& src, cv::Mat3b& dst, int k = 8);
		static void reduceColor_Stylization(const cv::Mat3b& src, cv::Mat3b& dst);
		static void reduceColor_EdgePreserving(const cv::Mat3b& src, cv::Mat3b& dst);

		static std::map<cv::Vec3b, int, lessVec3b> getPalette(const cv::Mat3b& src);

		//Utils for superpixels
		cv::Mat meanImage(cv::Mat& labels, cv::Mat& image, int numberOfSuperpixels);
	};
}


#endif //_SEGMENTATION_HPP_