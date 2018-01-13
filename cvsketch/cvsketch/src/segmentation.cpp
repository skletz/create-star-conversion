#include "segmentation.hpp"

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
	//int colorTable[8][3] = {
	//	{ 230,25,75 },{ 60,180,75 },{ 230,25,75 },{ 60,180,75 },{ 230,25,75 },{ 60,180,75 },{ 60,180,75 },{ 60,180,75 }
	//};

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
