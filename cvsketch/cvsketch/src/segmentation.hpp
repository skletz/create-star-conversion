#ifndef _SEGMENTATION_HPP_
#define  _SEGMENTATION_HPP_

//
//  Segmentation
//  segmentation.cpp
//  cvsketch
//
//  Created by Sabrina Kletz on 12.01.18.
//  Copyright � 2018 Sabrina Kletz. All rights reserved.
//

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

namespace vbs
{
	struct lessVec3b
	{
		bool operator()(const cv::Vec3b& lhs, const cv::Vec3b& rhs) const
		{

			return (lhs[0] != rhs[0]) ? (lhs[0] < rhs[0]) : ((lhs[1] != rhs[1]) ? (lhs[1] < rhs[1]) : (lhs[2] < rhs[2]));
		}

	};

	static std::map<cv::Vec3b, int, lessVec3b> default_palette_rgb = {
		std::make_pair(cv::Vec3b(209,33,51),1), //red
		std::make_pair(cv::Vec3b(243,105,69),1), //orange
		std::make_pair(cv::Vec3b(249,233,8),1), //yellow
		std::make_pair(cv::Vec3b(12,129,67),1), //green
		std::make_pair(cv::Vec3b(47,99,175),1), //blue_l
		std::make_pair(cv::Vec3b(77,50,145),1), //blue_d
		std::make_pair(cv::Vec3b(249,204,226),1), //skin
		std::make_pair(cv::Vec3b(127,90,50),1), //brown
		std::make_pair(cv::Vec3b(129,129,131),1), //grey
        std::make_pair(cv::Vec3b(0,0,0),1), //black
		std::make_pair(cv::Vec3b(255,255,255),1), //white
		std::make_pair(cv::Vec3b(255,0,0),1), //red
		std::make_pair(cv::Vec3b(255,102,0),1), //d_orange
		std::make_pair(cv::Vec3b(255,148,0),1), //m_orange
		std::make_pair(cv::Vec3b(255,197,0),1), //l_orange
		std::make_pair(cv::Vec3b(254,255,0),1), //yellow
		std::make_pair(cv::Vec3b(140,199,0),1), //l_green
		std::make_pair(cv::Vec3b(15,173,0),1), //d_green
		std::make_pair(cv::Vec3b(0,163,199),1), //l_blue
		std::make_pair(cv::Vec3b(0,100,181),1), //m_blue
		std::make_pair(cv::Vec3b(0,16,165),1), //d_blue
		std::make_pair(cv::Vec3b(99,0,165),1), //vieolett
		std::make_pair(cv::Vec3b(197,0,124),1), //pink

	};


	class Segmentation {

	public:

        cv::Ptr<cv::ximgproc::SuperpixelSEEDS> seeds;

        Segmentation();
        
		//Color reduction

		static void reduceColor_Quantization(const cv::Mat3b& src, cv::Mat3b& dst);
		static void reduceColor_kmeans(const cv::Mat3b& src, cv::Mat3b& dst, int k = 3);
		static void reduceColor_spatialkmeans(const cv::Mat3b& src, cv::Mat3b& dst, int k = 8);
		static void reduceColor_Stylization(const cv::Mat3b& src, cv::Mat3b& dst);
		static void reduceColor_EdgePreserving(const cv::Mat3b& src, cv::Mat3b& dst);

		void get_colorpalette(const cv::Mat3b& src, std::vector<std::pair<cv::Vec3b, float>>& output);

		void quantize_image(const cv::Mat& labels, const cv::Mat& image, int numberOfSuperpixels, std::vector<std::pair<cv::Vec3b, float>> &palette, cv::Mat& output);

		static cv::Scalar ScalarHSV2BGR(uchar H, uchar S, uchar V);
		static cv::Scalar ScalarRGB2LAB(uchar R, uchar G, uchar B);
		static cv::Scalar ScalarLAB2BGR(uchar R, uchar G, uchar B);

		//@TODO change
		static std::map<cv::Vec3b, int, lessVec3b> getPalette(const cv::Mat3b& src);
		static void sortPaletteByArea(std::map<cv::Vec3b, int, lessVec3b> input, std::vector<std::pair<cv::Vec3b, int>>& output);
		static void sortPaletteByArea(std::vector<std::pair<cv::Vec3b, int>> input, std::vector<std::pair<cv::Vec3b, int>>& output);

		//Utils for superpixels
		static void meanImage(cv::Mat& labels, cv::Mat& image, int numberOfSuperpixels, cv::Mat& output);

	};
}


#endif //_SEGMENTATION_HPP_
