//
// Created by bns on 1/8/18.
//

#ifndef SBSS_FITLINE_H
#define SBSS_FITLINE_H

#include <opencv2/core/mat.hpp>

// cfg (FDCMChamferdetect.m)
const double SIGMA_FIT_A_LINE = 0.5;
const double SIGMA_FIND_SUPPORT = 0.5;
const double MAX_GAP = 2.0;
const int N_LINES_TO_FIT_IN_STAGE_1 = 300;
const int N_LINES_TO_FIT_IN_STAGE_2  = 100000;
const int N_TRIALS_PER_LINE_IN_STAGE_2 = 1;

class Fitline
{
public:
    void fitline(int argc, char *argv[]);
    void fitlineToLineRep(const cv::Mat &templateEdgemap, const std::string &outFileName);
};

#endif //SBSS_FITLINE_H
