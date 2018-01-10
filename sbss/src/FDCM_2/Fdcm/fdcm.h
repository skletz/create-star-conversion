//
// Created by bns on 1/8/18.
//

#ifndef SBSS_FDCM_H
#define SBSS_FDCM_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

class FDCM
{
public:
    void fdcm_detect(const std::string &templateTxt, const std::string &targetImagePath, const cv::Mat &targetEdgeMap, const std::string &resultOutPath);
    void fdcm(int argc, const char *argv[]);
};

#endif //SBSS_FDCM_H
