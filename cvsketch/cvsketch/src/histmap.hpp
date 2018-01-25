//
//  HistMapExtractor.hpp
//  dive
//
//  Created by Sabrina Kletz on 24/02/18.
//  Copyright © 2018 Sabrina Kletz. All rights reserved.
//

#ifndef HistMapExtractor_hpp
#define HistMapExtractor_hpp

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <boost/tokenizer.hpp>
#include <iostream>
#include <string>
#include <stdlib.h>

namespace vbs {


    /**
     * @brief The Histogram Map Extractor for comparing rough drawn color sketches in a grid-based manner
     *
     */
    class HistMapExtractor {

    private:

        std::string mImage_path;

        int mPatchSizeX;
        int mPatchSizeY;
        int mCellSizeX;
        int mCellSizeY;
        int mPatchStepX;
        int mPatchStepY;

    public:

        static const std::vector<std::pair<cv::Vec3b, float>> color_palette_rgb;

        HistMapExtractor(std::string _imagepath);

        ~HistMapExtractor();

        void process();

        static void compute_histmap(cv::Mat _src, cv::Mat& _dst, int _width = 320, int _height = 240, int _psize = 80, int _csize = 40, int _pstep = 40);

        void writeToFile(const std::string& _filepath, const cv::Mat& _descriptor);

        static void readFromFile(const std::string _filepath, std::vector<std::vector<float>>& _descriptor);

        static void readFromFile(const std::string _filepath, cv::Mat& _descriptor);

        static float calculateSimilarity(std::vector<std::vector<float>> _d1, std::vector<std::vector<float>> _d2);

        static float calculateSimilarity(const cv::Mat& _d1, const cv::Mat& _d2);

        void set_imagepath(std::string _imagepath);

        std::string get_imagepath();

    private:

        static void compute_histmap_patch(const cv::Mat& _patch, std::vector<std::pair<cv::Vec3b, float>>& _palette, int _cstep, cv::Mat& _hist);

        static void compute_histmap_cell(const cv::Mat& _cell, std::vector<std::pair<cv::Vec3b, float>>& _palette, cv::Mat& _hist);

        static cv::Scalar ScalarLAB2BGR(uchar L, uchar A, uchar B);

        static cv::Scalar ScalarBGR2LAB(uchar B, uchar G, uchar R);

    };

}

#endif /* HistMapExtractor_hpp */
