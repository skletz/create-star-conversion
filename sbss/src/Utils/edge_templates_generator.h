//
// Created by bns on 1/7/18.
//

#ifndef SBSS_EDGE_TEMPLATES_GENERATOR_H
#define SBSS_EDGE_TEMPLATES_GENERATOR_H

class EdgeTemplatesGenerator
{

private:
    CvPoint2D32f project3Dto2D(CvPoint3D32f pt3, CvMat *pose, CvMat *param_intrinsic);

    CvRect drawModel(IplImage *img, std::vector<CvPoint3D32f> ep1, std::vector<CvPoint3D32f> ep2, CvMat *pose,
                     CvMat *param_intrinsic, CvScalar color);

public:
    EdgeTemplatesGenerator();

    int generateTemplate(int argc, const char **argv);

};

#endif //SBSS_EDGE_TEMPLATES_GENERATOR_H
