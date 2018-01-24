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
		std::make_pair(cv::Vec3b(254,39,18),1), //red
		std::make_pair(cv::Vec3b(253,83,8),1), //dark_orange
		std::make_pair(cv::Vec3b(251,153,2),1), //orange
		std::make_pair(cv::Vec3b(250,188,2),1), //light_orange
		std::make_pair(cv::Vec3b(254,254,51),1), //yellow
        
		std::make_pair(cv::Vec3b(208,234,43),1), //ligh_green
		std::make_pair(cv::Vec3b(102,176,50),1), //green
		std::make_pair(cv::Vec3b(3,145,206),1), //light_blue
		std::make_pair(cv::Vec3b(2,71,254),1), //blue
        std::make_pair(cv::Vec3b(61,1,164),1), //dark_blue
		std::make_pair(cv::Vec3b(134,1,175),1), //violett
		std::make_pair(cv::Vec3b(167,25,75),1), //purpel
		std::make_pair(cv::Vec3b(0,0,0),1), //black
		std::make_pair(cv::Vec3b(64,64,64),1), //dark_gray
		std::make_pair(cv::Vec3b(128,128,128),1), //light_gray
		std::make_pair(cv::Vec3b(255,255,255),1), //white
	};

    static std::vector<std::pair<cv::Vec3b, float>> default_palette_sorted_rgb = {
		std::make_pair(cv::Vec3b(110,73,49),1), //brown
        std::make_pair(cv::Vec3b(173,131,104),1), //light_brown
        std::make_pair(cv::Vec3b(254,39,18),1), //red
		std::make_pair(cv::Vec3b(253,83,8),1), //dark_orange
		std::make_pair(cv::Vec3b(251,153,2),1), //orange
		std::make_pair(cv::Vec3b(250,188,2),1), //light_orange
		std::make_pair(cv::Vec3b(254,254,51),1), //yellow

		std::make_pair(cv::Vec3b(208,234,43),1), //ligh_green
		std::make_pair(cv::Vec3b(102,176,50),1), //green
        //std::make_pair(cv::Vec3b(42,120,0),1), //dark_green
        std::make_pair(cv::Vec3b(0,161,160),1), //türkis
		std::make_pair(cv::Vec3b(3,145,206),1), //light_blue
        //std::make_pair(cv::Vec3b(142,183,208),1), //light_light_blue
        //std::make_pair(cv::Vec3b(0,219,219),1), //light_türkis
		std::make_pair(cv::Vec3b(2,71,254),1), //blue
		std::make_pair(cv::Vec3b(61,1,164),1), //dark_blue
		std::make_pair(cv::Vec3b(134,1,175),1), //violett
        //std::make_pair(cv::Vec3b(230,200,189),1), //rose
		std::make_pair(cv::Vec3b(167,25,75),1), //purpel
		std::make_pair(cv::Vec3b(0,0,0),1), //black
		std::make_pair(cv::Vec3b(64,64,64),1), //dark_gray
		std::make_pair(cv::Vec3b(128,128,128),1), //light_gray
		std::make_pair(cv::Vec3b(255,255,255),1), //white
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

		/**
		 *
		 * @return returns the default color palette in lab, sorted by colors
		 */
		void get_default_palette_lab(std::vector<std::pair<cv::Vec3b, float>>& _output);

		void quantize_image(const cv::Mat& labels, const cv::Mat& image, int numberOfSuperpixels, std::vector<std::pair<cv::Vec3b, float>> &palette, cv::Mat& output);

		static cv::Scalar ScalarHSV2BGR(uchar H, uchar S, uchar V);
		static cv::Scalar ScalarRGB2LAB(uchar R, uchar G, uchar B);
		static cv::Scalar ScalarLAB2BGR(uchar R, uchar G, uchar B);
		static cv::Scalar ScalarBGR2LAB(uchar B, uchar G, uchar R);

		//@TODO change
		static std::map<cv::Vec3b, int, lessVec3b> getPalette(const cv::Mat3b& src);
		static void sortPaletteByArea(std::map<cv::Vec3b, int, lessVec3b> input, std::vector<std::pair<cv::Vec3b, int>>& output);
		static void sortPaletteByArea(std::vector<std::pair<cv::Vec3b, int>> input, std::vector<std::pair<cv::Vec3b, int>>& output);

		//Utils for superpixels
		static void meanImage(cv::Mat& labels, cv::Mat& image, int numberOfSuperpixels, cv::Mat& output);

	};
}


#endif //_SEGMENTATION_HPP_
