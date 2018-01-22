#include "segmentation.hpp"
#include "../libs/CIEDE2000/CIEDE2000.h"


vbs::Segmentation::Segmentation()
{

}

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
//    cv::cvtColor(dst, dst, CV_Lab2BGR);
//    cv::imshow("Dst", dst);
//    cv::waitKey(0);
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

void vbs::Segmentation::get_colorpalette(const cv::Mat3b& src, std::vector<std::pair<cv::Vec3b, float>>& output)
{
    std::map<cv::Vec3b, float, lessVec3b> palette;
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

    std::vector<std::pair<cv::Vec3b, int>> results;

    for(auto color : palette)
    {
        results.push_back(std::make_pair(color.first, color.second));

    }

    std::sort(results.begin(), results.end(), [](const std::pair<cv::Vec3b, int>& s1, const std::pair<cv::Vec3b, int>& s2)
    {
        cv::Vec3b c1 = s1.first;
        cv::Vec3b c2 = s2.first;
        return (s1.second > s2.second);

    });

    output.assign(results.begin(), results.end());
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

void vbs::Segmentation::get_default_palette_lab(std::vector<std::pair<cv::Vec3b, float>>& _output)
{
	std::vector<std::pair<cv::Vec3b, float>> output;

	for (auto color : default_palette_sorted_rgb)
	{
		cv::Scalar lab = ScalarRGB2LAB(color.first[0], color.first[1], color.first[2]);
		cv::Vec3b tmp = cv::Vec3b(lab[0], lab[1], lab[2]);
		output.push_back(std::make_pair(tmp, -1.0));
	}
	_output.assign(output.begin(), output.end());
}

//std::map<cv::Scalar, int, vbs::lessVec3b> vbs::Segmentation::getPaletteNormalized(const cv::Mat3b& src)
//{
//	std::map<cv::Scalar, int, lessVec3b> palette;
//	for (int r = 0; r < src.rows; ++r)
//	{
//		for (int c = 0; c < src.cols; ++c)
//		{
//			cv::Scalar color = src(r, c);
//
//			color[0] = color[0] / 100.0;
//			color[1] = (color[1] + 127.0) / 255.0;
//			color[2] = (color[2] + 127.0) / 255.0;
//
//			if ((color[0] < 0 && color[0] > 1) || (color[1] < 0 && color[1] > 1) || (color[2] < 0 && color[2] > 1))
//				std::cout << "Not normalized: " << color << std::endl;
//
//
//
//			if (palette.count(color) == 0)
//			{
//				palette[color] = 1;
//			}
//			else
//			{
//				palette[color] = palette[color] + 1;
//			}
//		}
//	}
//	return palette;
//}

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

cv::Scalar vbs::Segmentation::ScalarBGR2LAB(uchar B, uchar G, uchar R) {
	cv::Mat bgr;
	cv::Mat lab(1, 1, CV_8UC3, cv::Scalar(B, G, R));
	cv::cvtColor(lab, bgr, CV_BGR2Lab);
	return cv::Scalar(bgr.data[0], bgr.data[1], bgr.data[2]);
}

cv::Scalar vbs::Segmentation::ScalarLAB2BGR(uchar L, uchar A, uchar B) {
	cv::Mat bgr;
	cv::Mat lab(1, 1, CV_8UC3, cv::Scalar(L, A, B));
	cv::cvtColor(lab, bgr, CV_Lab2BGR);
	return cv::Scalar(bgr.data[0], bgr.data[1], bgr.data[2]);
}

void vbs::Segmentation::quantize_image(const cv::Mat& labels, const cv::Mat& image, int numberOfSuperpixels, std::vector<std::pair<cv::Vec3b, float>> &palette, cv::Mat& output)
{
	std::map<cv::Vec3b, float, lessVec3b> colormap;
    std::vector<std::pair<cv::Vec3b, float>> default_colors;
    
	if (palette.empty())
	{

		for (auto color : default_palette_rgb)
		{
			cv::Scalar lab = ScalarRGB2LAB(color.first[0], color.first[1], color.first[2]);
			cv::Vec3b tmp = cv::Vec3b(lab[0], lab[1], lab[2]);
			colormap.insert(std::make_pair(tmp,1));
		}

        
        for(auto color : palette)
        {
            default_colors.push_back(std::make_pair(color.first, color.second));
            
        }

		palette = default_colors;
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

			if ((idx_l < 0 && idx_l > 1) || (idx_a < 0 && idx_a > 1) || (idx_b < 0 && idx_b > 1))
				std::cout << "Not normalized: " << color.first << std::endl;

			if ((cmp_l < 0 && cmp_l > 1) || (cmp_a < 0 && cmp_a > 1) || (cmp_b < 0 && cmp_b > 1))
				std::cout << "Not normalized: " << cv::Vec3b(meanL,meanA,meanB) << std::endl;


			double dist;

			CIEDE2000::LAB lab1, lab2;
			lab1.l = idx_l;
			lab1.a = idx_a;
			lab1.b = idx_b;

			lab2.l = cmp_l;
			lab2.a = cmp_a;
			lab2.b = cmp_b;

			dist = CIEDE2000::CIEDE2000(lab1, lab2);

			if (dist < min)
			{
				min = dist;
				min_color = color.first;
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

void vbs::Segmentation::sortPaletteByArea(std::map<cv::Vec3b, int, lessVec3b> input, std::vector<std::pair<cv::Vec3b, int>>& output)
{

    std::vector<std::pair<cv::Vec3b, int>> results;

    for(auto color : input)
    {
       output.push_back(std::make_pair(color.first, color.second));

    }

    std::sort(output.begin(), output.end(), [](const std::pair<cv::Vec3b, int>& s1, const std::pair<cv::Vec3b, int>& s2)
    {
        cv::Vec3b c1 = s1.first;
        cv::Vec3b c2 = s2.first;
        return (s1.second > s2.second);

    });

}

void vbs::Segmentation::sortPaletteByArea(std::vector<std::pair<cv::Vec3b, int>> input, std::vector<std::pair<cv::Vec3b, int>>& output)
{

    for(auto color : input)
    {
       output.push_back(std::make_pair(color.first, color.second));

    }

    std::sort(output.begin(), output.end(), [](const std::pair<cv::Vec3b, int>& s1, const std::pair<cv::Vec3b, int>& s2)
    {
      cv::Vec3b c1 = s1.first;
      cv::Vec3b c2 = s2.first;
      return (s1.second > s2.second);

    });

}
