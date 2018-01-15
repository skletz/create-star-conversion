#include "segmentation.hpp"
#include "../libs/CIEDE2000/CIEDE2000.h"

void vbs::Segmentation::reduceColor_Quantization(const cv::Mat3b& src, cv::Mat3b& dst)
{
	uchar N = 64;
	dst = src / N;
	dst *= N;
}

void vbs::Segmentation::reduceColor_kmeans(const cv::Mat3b& src, cv::Mat3b& dst, int k)
{
	int n = src.rows * src.cols;
	cv::Mat data = src.reshape(1, n);
	data.convertTo(data, CV_32F);

	std::vector<int> labels;
	cv::Mat1f colors;
	kmeans(data, k, labels, cv::TermCriteria(), 1, cv::KMEANS_PP_CENTERS, colors);

	for (int i = 0; i < n; ++i)
	{
		data.at<float>(i, 0) = colors(labels[i], 0);
		data.at<float>(i, 1) = colors(labels[i], 1);
		data.at<float>(i, 2) = colors(labels[i], 2);
	}

	cv::Mat reduced = data.reshape(3, src.rows);
	reduced.convertTo(dst, CV_8U);
}

void vbs::Segmentation::reduceColor_spatialkmeans(const cv::Mat3b& src, cv::Mat3b& dst, int k)
{
	int n = src.rows * src.cols;
	cv::Mat data = src.reshape(1, n);
	data.convertTo(data, CV_32F);

	std::vector<int> labels;
	cv::Mat1f colors;
	kmeans(data, k, labels, cv::TermCriteria(), 1, cv::KMEANS_PP_CENTERS, colors);

	for (int i = 0; i < n; ++i)
	{
		data.at<float>(i, 0) = colors(labels[i], 0);
		data.at<float>(i, 1) = colors(labels[i], 1);
		data.at<float>(i, 2) = colors(labels[i], 2);
	}

	cv::Mat reduced = data.reshape(3, src.rows);
	reduced.convertTo(dst, CV_8U);
}

void vbs::Segmentation::reduceColor_Stylization(const cv::Mat3b & src, cv::Mat3b & dst)
{
	stylization(src, dst);
}

void vbs::Segmentation::reduceColor_EdgePreserving(const cv::Mat3b & src, cv::Mat3b & dst)
{
	edgePreservingFilter(src, dst);
}

std::map<cv::Vec3b, int, vbs::lessVec3b> vbs::Segmentation::getPalette(const cv::Mat3b& src)
{
	std::map<cv::Vec3b, int, lessVec3b> palette;
	for (int r = 0; r < src.rows; ++r)
	{
		for (int c = 0; c < src.cols; ++c)
		{
			cv::Vec3b color = src(r, c);
			if (palette.count(color) == 0)
			{
				palette[color] = 1;
			}
			else
			{
				palette[color] = palette[color] + 1;
			}
		}
	}
	return palette;
}

void vbs::Segmentation::meanImage(cv::Mat & labels, cv::Mat & image, int numberOfSuperpixels, cv::Mat& output)
{
	int width = image.cols;
	int height = image.rows;

	std::cout << "Image Size: " << width << "x" << height << std::endl;
	std::cout << "Label Size: " << labels.cols << "x" << labels.rows << std::endl;

	cv::Mat newImage = image.clone();

	int meanB = 0;
	int meanG = 0;
	int meanR = 0;
	int count = 0;

	for (int label = 0; label < numberOfSuperpixels; label++) {
		meanB = 0;
		meanG = 0;
		meanR = 0;
		count = 0;

		for (int i = 0; i < newImage.rows; i++) {
			for (int j = 0; j < newImage.cols; j++) {
				if (labels.at<int>(i, j) == label) {
					meanB += image.at<cv::Vec3b>(i, j)[0];
					meanG += image.at<cv::Vec3b>(i, j)[1];
					meanR += image.at<cv::Vec3b>(i, j)[2];

					count++;
				}
			}
		}

		if (count > 0) {
			meanB = (int)meanB / count;
			meanG = (int)meanG / count;
			meanR = (int)meanR / count;
		}

		for (int i = 0; i < newImage.rows; i++) {
			for (int j = 0; j < newImage.cols; j++) {
				if (labels.at<int>(i, j) == label) {
					newImage.at<cv::Vec3b>(i, j)[0] = meanB;
					newImage.at<cv::Vec3b>(i, j)[1] = meanG;
					newImage.at<cv::Vec3b>(i, j)[2] = meanR;
				}
			}
		}
	}

	newImage.copyTo(output);
}

cv::Scalar vbs::Segmentation::ScalarHSV2BGR(uchar H, uchar S, uchar V) {
	cv::Mat rgb;
	cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(H, S, V));
	cv::cvtColor(hsv, rgb, CV_HSV2BGR);
	return cv::Scalar(rgb.data[0], rgb.data[1], rgb.data[2]);
}

cv::Scalar vbs::Segmentation::ScalarRGB2LAB(uchar R, uchar G, uchar B) {
	cv::Mat rgb;
	cv::Mat lab(1, 1, CV_8UC3, cv::Scalar(R, G, B));
	cv::cvtColor(lab, rgb, CV_RGB2Lab);
	return cv::Scalar(rgb.data[0], rgb.data[1], rgb.data[2]);
}

cv::Scalar vbs::Segmentation::ScalarLAB2BGR(uchar L, uchar A, uchar B) {
	cv::Mat bgr;
	cv::Mat lab(1, 1, CV_8UC3, cv::Scalar(L, A, B));
	cv::cvtColor(lab, bgr, CV_Lab2BGR);
	return cv::Scalar(bgr.data[0], bgr.data[1], bgr.data[2]);
}

void vbs::Segmentation::quantizedImage(cv::Mat& labels, cv::Mat& image, int numberOfSuperpixels,
	std::map<cv::Vec3b, int, lessVec3b> palette, cv::Mat& output)
{

	std::map<cv::Vec3b, int, lessVec3b> default_pallette_lab;

	if (palette.empty())
	{
		
		for (auto color : default_pallette_rgb)
		{
			cv::Scalar lab = ScalarRGB2LAB(color.first[0], color.first[1], color.first[2]);
			cv::Vec3b tmp = cv::Vec3b(lab[0], lab[1], lab[2]);
			default_pallette_lab.insert(std::make_pair(tmp,1));
		}

		palette = default_pallette_lab;
	}


	int width = image.cols;
	int height = image.rows;

	std::cout << "Image Size: " << width << "x" << height << std::endl;
	std::cout << "Label Size: " << labels.cols << "x" << labels.rows << std::endl;

	cv::Mat newImage = image.clone();

	int meanL = 0;
	int meanA = 0;
	int meanB = 0;
	int count = 0;

	for (int label = 0; label < numberOfSuperpixels; label++) {
		meanL = 0;
		meanA = 0;
		meanB = 0;
		count = 0;

		
		for (int i = 0; i < newImage.rows; i++) {
			for (int j = 0; j < newImage.cols; j++) {
				if (labels.at<int>(i, j) == label) {
					meanL += image.at<cv::Vec3b>(i, j)[0];
					meanA += image.at<cv::Vec3b>(i, j)[1];
					meanB += image.at<cv::Vec3b>(i, j)[2];

					count++;

				}
			}
		}

		if (count > 0) {
			meanL = (int)meanL / count;
			meanA = (int)meanA / count;
			meanB = (int)meanB / count;
		}

		double min = std::numeric_limits<double>::max();
		cv::Vec3b min_color = cv::Vec3b(0, 0, 0);
		cv::Mat tmin;
		for (auto color : palette)
		{
			double idx_l = color.first[0] / 100.0;
			double idx_a = (color.first[1] + 127.0) / 255.0;
			double idx_b = (color.first[2] + 127.0) / 255.0;

			double cmp_l = meanL / 100.0;
			double cmp_a = (meanA + 127.0) / 255.0;
			double cmp_b = (meanB + 127.0) / 255.0;
			//double idx_l = color.first[0] ;
			//double idx_a = (color.first[1]) ;
			//double idx_b = (color.first[2]);

			//double cmp_l = meanL;
			//double cmp_a = (meanA);
			//double cmp_b = (meanB);

			if (idx_l < 0 && idx_l > 1 || idx_a < 0 && idx_a > 1 || idx_b < 0 && idx_b > 1)
				std::cout << "Not normalized: " << color.first << std::endl;

			if (cmp_l < 0 && cmp_l > 1 || cmp_a < 0 && cmp_a > 1 || cmp_b < 0 && cmp_b > 1)
				std::cout << "Not normalized: " << cv::Vec3b(meanL,meanA,meanB) << std::endl;

			//double l = idx_l - cmp_l;
			//double a = idx_a - cmp_a;
			//double b = idx_b - cmp_b;
			double dist;
			//dist = std::pow(l, 2) + std::pow(a, 2) + std::pow(b, 2);
			//dist = std::sqrt(dist);

			CIEDE2000::LAB lab1, lab2;
			lab1.l = idx_l;
			lab1.a = idx_a;
			lab1.b = idx_b;

			lab2.l = cmp_l;
			lab2.a = cmp_a;
			lab2.b = cmp_b;

			dist = CIEDE2000::CIEDE2000(lab1, lab2);

			//cv::Mat tidx(1, 1, CV_8UC3, cv::Scalar(idx_l, idx_a, idx_b));
			//cv::Mat cmpidx(1, 1, CV_8UC3, cv::Scalar(cmp_l, cmp_a, cmp_b));

			if (dist < min)
			{
				min = dist;
				min_color = color.first;
				//cv::Mat t(1, 1, CV_8UC3, cv::Scalar(min_color[0], min_color[1], min_color[2]));
				//tmin = t;
			}

		}

		for (int i = 0; i < newImage.rows; i++) {
			for (int j = 0; j < newImage.cols; j++) {
				if (labels.at<int>(i, j) == label) {
					newImage.at<cv::Vec3b>(i, j)[0] = min_color[0];
					newImage.at<cv::Vec3b>(i, j)[1] = min_color[1];
					newImage.at<cv::Vec3b>(i, j)[2] = min_color[2];
				}
			}
		}
	}

	newImage.copyTo(output);

}
