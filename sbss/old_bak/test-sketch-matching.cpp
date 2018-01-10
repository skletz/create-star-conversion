//
// Created by bns on 12/27/17.
//

#include <iostream>
#include <limits>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "FDCM_1/Image/Image.h"
#include "FDCM_1/Image/ImageIO.h"
#include "FDCM_1/Fitline/LFLineFitter.h"
#include "FDCM_1/Fdcm/LMLineMatcher.h"

#include <opencv2/opencv.hpp>
#include "chamfer/Chamfer.hpp"
#include "chamfer/Utils.hpp"
#include "Utils/Timer.hpp"

//TODO: create template  (currently not properly working)
#include "FDCM_1/Utils/edge_templates_generator.h"

//std::string DATA_LOCATION_PREFIX = "../data/";
std::string DATA_LOCATION_PREFIX = "../../cv.sketch/cv.sketch/data/sketch_sample/";
std::string OUT_DIR_PREFIX = "../out";


int testOriginalChamfer()
{
//    cv::Mat img_template = cv::imread(DATA_LOCATION_PREFIX + "Inria_logo_template.jpg");
//  cv::Mat img_query = cv::imread(DATA_LOCATION_PREFIX + "Inria_scene.jpg");
//  cv::Mat img_query = cv::imread(DATA_LOCATION_PREFIX + "Inria_scene2.jpg");
//  cv::Mat img_query = cv::imread(DATA_LOCATION_PREFIX + "Inria_scene3.jpg");
//  cv::Mat img_query = cv::imread(DATA_LOCATION_PREFIX + "Inria_scene4.jpg");
//    cv::Mat img_query = cv::imread(DATA_LOCATION_PREFIX + "Inria_scene5.jpg");

    cv::Mat img_template = cv::imread(DATA_LOCATION_PREFIX + "sketches/0.png");
    cv::Mat img_query = cv::imread(DATA_LOCATION_PREFIX + "images/0/641602006.jpg");

    Timer timer(true);

    std::map<int, cv::Mat> mapOfTemplates;
    std::map<int, std::pair<cv::Rect, cv::Rect> > mapOfTemplateRois;
    mapOfTemplates[1] = img_template;
    mapOfTemplateRois[1] = std::pair<cv::Rect, cv::Rect>(cv::Rect(0, 0, -1, -1), cv::Rect(0, 0, -1, -1));

    ChamferMatcher chamfer(mapOfTemplates, mapOfTemplateRois);
    std::cout << "Setup time = " << timer.getTimeMS() << " ms" << std::endl;


    std::vector<Detection_t> detections;
    bool useOrientation = true;
    float distanceThreshold = 100.0, lambda = 100.0f;
    float weight_forward = 1.0f, weight_backward = 1.0f;
    bool useNonMaximaSuppression = true, useGroupDetections = true;

    chamfer.setCannyThreshold(70.0);
    chamfer.setMatchingType(ChamferMatcher::edgeMatching);
//  chamfer.setMatchingType(ChamferMatcher::edgeForwardBackwardMatching);
//  chamfer.setMatchingType(ChamferMatcher::fullMatching);
//  chamfer.setMatchingType(ChamferMatcher::lineMatching);
//  chamfer.setMatchingType(ChamferMatcher::lineForwardBackwardMatching);

    timer.reset();
//  chamfer.detect(img_query, detections, useOrientation, distanceThreshold, lambda, weight_forward,
//  		weight_backward, useGroupDetections);
    chamfer.detectMultiScale(img_query, detections, useOrientation, distanceThreshold, lambda, weight_forward,
                             weight_backward, useNonMaximaSuppression, useGroupDetections);
//    t = ((double) cv::getTickCount() - t) / cv::getTickFrequency() * 1000.0;
    std::cout << "Processing time = " << timer.getTimeMS() << " ms" << std::endl;
    std::cout << "detections = " << detections.size() << std::endl;

    cv::Mat result;
    img_query.convertTo(result, CV_8UC3);


    for (std::vector<Detection_t>::const_iterator it = detections.begin(); it != detections.end(); ++it)
    {
        cv::rectangle(result, it->m_boundingBox, cv::Scalar(0, 0, 255), 2);

        std::stringstream ss;
        //Chamfer distance
        ss << it->m_chamferDist;
        cv::Point ptText = it->m_boundingBox.tl() + cv::Point(10, 20);
        cv::putText(result, ss.str(), ptText, cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 0, 0), 2);

        //Scale
        ss.str("");
        ss << it->m_scale;
        ptText = it->m_boundingBox.tl() + cv::Point(10, 40);
        cv::putText(result, ss.str(), ptText, cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 0, 0), 2);

        cv::imshow("result", result);
        cv::waitKey(0);
    }

    //XXX:
//  cv::imwrite("Simple_test_result_single_scale.png", result);

    cv::waitKey(0);
    return 0;
}


void getDetections(const std::string &queryImgPath, const std::vector<boost::filesystem::path> &imgDatabasePaths,
                   std::vector<std::pair<int, boost::filesystem::path>> &detectionsMap)
{
    Timer timer(true);


    cv::Mat img_sketch = cv::imread(queryImgPath);
    std::map<int, cv::Mat> mapOfTemplates;
    std::map<int, std::pair<cv::Rect, cv::Rect> > mapOfTemplateRois;
    mapOfTemplates[1] = img_sketch;
    mapOfTemplateRois[1] = std::pair<cv::Rect, cv::Rect>(cv::Rect(0, 0, -1, -1), cv::Rect(0, 0, -1, -1));

    ChamferMatcher chamfer(mapOfTemplates, mapOfTemplateRois);
    timer.printTime("Setup");

    cv::Mat img_query;



//    for (boost::filesystem::path file: files)
//    {
//        std::cout << file.string() << std::endl;
//    }

    std::cout << "Sketch: " << queryImgPath << std::endl;
    std::cout << "Total dataset files: " << imgDatabasePaths.size() << std::endl;


    int count = 0;


    // perform chamfer matching on all images
    for (boost::filesystem::path file: imgDatabasePaths)
    {

        std::vector<Detection_t> detections;
        bool useOrientation = true;
        float distanceThreshold = 100.0, lambda = 100.0f;
        float weight_forward = 1.0f, weight_backward = 1.0f;
        bool useNonMaximaSuppression = true, useGroupDetections = true;

        chamfer.setCannyThreshold(70.0);
        chamfer.setMatchingType(ChamferMatcher::edgeMatching);
        //  chamfer.setMatchingType(ChamferMatcher::edgeForwardBackwardMatching);
        //  chamfer.setMatchingType(ChamferMatcher::fullMatching);
        //  chamfer.setMatchingType(ChamferMatcher::lineMatching);
        //  chamfer.setMatchingType(ChamferMatcher::lineForwardBackwardMatching);

        timer.reset();

        img_query.release();
        img_query = cv::imread(file.string());

//        chamfer.detect(img_query, detections, useOrientation, distanceThreshold, lambda, weight_forward,
//                       weight_backward, useGroupDetections);


        chamfer.detectMultiScale(img_query, detections, useOrientation, distanceThreshold, lambda, weight_forward,
                                 weight_backward, useNonMaximaSuppression, useGroupDetections);

        timer.printTime(
                "Processing " + std::to_string(count + 1) + "/" + std::to_string(imgDatabasePaths.size()) + ", " +
                file.string());
        std::cout << "detections = " << detections.size() << std::endl;

        std::pair<int, boost::filesystem::path> detectPair(detections.size(), file);

        detectionsMap.push_back(detectPair);


        count++;

        // result image
        //cv::Mat result;
        //img_query.convertTo(result, CV_8UC3);

    }

    std::sort(detectionsMap.begin(), detectionsMap.end());
}

void DrawDetWind(IplImage *image,int x,int y,int detWindWidth,int detWindHeight,CvScalar scalar,int thickness)
{
    cvLine(image,cvPoint( x,y),cvPoint( x+detWindWidth,y),scalar,thickness);
    cvLine(image,cvPoint( x+detWindWidth,y),cvPoint( x+detWindWidth, y+detWindHeight),scalar,thickness);
    cvLine(image,cvPoint( x+detWindWidth,y+detWindHeight),cvPoint( x, y+detWindHeight),scalar,thickness);
    cvLine(image,cvPoint( x, y+detWindHeight),cvPoint( x, y),scalar,thickness);
}

int testFastDirectionalChamfer(const std::string &templateFileName, const std::string &displayImageName, const std::string &edgeMapName = "" )
{

    LFLineFitter lf;
    LMLineMatcher lm;
    lf.Configure("../cfg/para_line_fitter_template.txt");
    lm.Configure("../cfg/para_line_matcher.txt");


    //Image *inputImage=NULL;
    IplImage *inputImage=NULL;
    //Image<uchar> *inputImage=NULL;
    IplImage *edgeImage = NULL;

//    string templateFileName(argv[1]);
//    string displayImageName(argc==4?argv[3]:argv[2]);

    //string templateFileName("Exp_Smoothness/template_list.txt");
    //string edgeMapName("Exp_Smoothness/device5-20_edge_cluttered.pgm");
    //string displayImageName("Exp_Smoothness/device5-20_edge_cluttered.pgm");

    //string templateFileName("data/template_giraffe.txt");
    //string edgeMapName("data/looking_edges.pgm");
    //string displayImageName("data/looking.ppm");

    //string templateFileName("data/template_applelogo.txt");
    //string edgeMapName("data/hat_edges.pgm");
    //string displayImageName("data/hat.jpg");


    //inputImage = cvLoadImage(edgeMapName.c_str(),0);
    //inputImage = ImageIO::LoadPGM(edgeMapName.c_str());

    if(edgeMapName != "")
    {
//        std::cout << argc << std::endl;
//        string edgeMapName(argv[2]);
        inputImage = cvLoadImage(edgeMapName.c_str(),0);
        if(inputImage==NULL)
        {
            std::cerr<<"[ERROR] Fail in reading image "<<edgeMapName<<std::endl;
            exit(0);
        }
    }
    else //if(argc == 3)
    {
        IplImage* img = cvLoadImage(displayImageName.c_str(),0);
        inputImage = cvCloneImage(img);
        //cvCanny(img, inputImage, 20, 40, 3);
        cvCanny(img, inputImage, 20, 80, 3);
        //cvCanny(img, inputImage, 80, 120, 3);
        cvReleaseImage(&img);

        edgeImage = cvCloneImage(inputImage);
    }

    lf.Init();
    lm.Init(templateFileName.c_str());


    // Line Fitting
    lf.FitLine(inputImage);

    // FDCM Matching
    vector<LMDetWind> detWind;
    //lm.Match(lf, detWind);

    vector<vector<LMDetWind> > detWinds(lm.ndbImages_);
    vector<LMDetWind> detWindAll;
    double maxThreshold = 0.30;
    for(int i=0; i<lm.ndbImages_; i++)
    {
        std::cout << "[" << i << "]-th template ..." << std::endl;
        lm.SingleShapeDetectionWithVaryingQuerySize(lf, i, maxThreshold, detWinds[i]);
        for(size_t j=0; j<detWinds[i].size(); j++)
        {
            detWindAll.push_back(detWinds[i][j]);
        }
    }

    // Sort the window array in the ascending order of matching cost.
    LMDetWind *tmpWind = new LMDetWind[detWindAll.size()];
    for(size_t i=0;i<detWindAll.size();i++)
        tmpWind[i] = detWindAll[i];
    MMFunctions::Sort(tmpWind, detWindAll.size());
    for(size_t i=0;i<detWindAll.size();i++)
        detWind.push_back(tmpWind[i]);
    delete [] tmpWind;

    //MMFunctions::Sort(detWindAll,detWindAll.size());
    //detWind = detWindAll;

    //int last = detWind2.size()-1;
    //int nDetWindows = detWind2[last].size();
    //detWind = detWind2[last];



    //lm.MatchCostMap(lf,outputCostMapName.c_str());

    // Display best matcher in edge map
    //if(displayImageName.c_str())
    //{
    //	Image<RGBMap> *debugImage = ImageIO::LoadPPM(displayImageName.c_str());
    //	LMDisplay::DrawDetWind(debugImage,detWind[0].x_,detWind[0].y_,detWind[0].width_,detWind[0].height_,RGBMap(0,255,0),4);
    //	char outputname[256];
    //	sprintf(outputname,"%s.output.ppm",displayImageName.c_str());
    //	ImageIO::SavePPM(debugImage,outputname);
    //	delete debugImage;
    //}

    std::cout << detWind.size() << " detections..." << std::endl;


    IplImage* dispImage = cvLoadImage(displayImageName.c_str());

    for(size_t i=0; i<detWind.size(); i++)
    {
        std::cout << detWind[i].x_ << " " << detWind[i].y_ << " " << detWind[i].width_ << " " << detWind[i].height_ << " " << detWind[i].cost_ << " " << detWind[i].count_ << " " << detWind[i].scale_ << " " << detWind[i].aspect_ << " " << detWind[i].tidx_ << std::endl;
    }

    for(size_t i=1; i<(detWind.size()<10?detWind.size():10); i++)
        DrawDetWind(dispImage, detWind[i].x_, detWind[i].y_, detWind[i].width_, detWind[i].height_, cvScalar(255,255,0), i==1?2:1);

    if(detWind.size() > 0)
        DrawDetWind(dispImage, detWind[0].x_, detWind[0].y_, detWind[0].width_, detWind[0].height_, cvScalar(0,255,255), 2);

    cvNamedWindow("edge", 1);
    cvNamedWindow("output", 1);
    cvShowImage("edge", edgeImage);
    cvShowImage("output", dispImage);
    cvSaveImage("result.png", dispImage);
    cvWaitKey(0);

    cvDestroyWindow("edge");
    cvDestroyWindow("output");

    if(inputImage)  cvReleaseImage(&inputImage);
    if(dispImage)   cvReleaseImage(&dispImage);
    if(edgeImage)   cvReleaseImage(&edgeImage);

    return 0;
}



int main()
{

    // query image path
    std::string sketchPath = DATA_LOCATION_PREFIX + "sketches/0.png";

    // database paths
    std::vector<boost::filesystem::path> files;
    static const std::string extArray[] = {".PNG", ".png", ".JPG", ".jpg", ".JPEG", ".jpeg", ".BMP", ".bmp"};
    std::vector<std::string> extVector(extArray, extArray + sizeof(extArray) / sizeof(extArray[0]));
    getAllFilesWithExtensions(DATA_LOCATION_PREFIX + "images/", extVector, files);


//    // test original implementation (chamfer folder) -> https://github.com/s-trinh/Chamfer-Matching
//    testOriginalChamfer();

//    // test original implementation with query image and database
//    std::vector<std::pair<int, boost::filesystem::path>> detectionsMap; // detections -> path
//    getDetections(sketchPath, files, detectionsMap);

//    // fast directional chamfer matching 1 -> https://github.com/CognitiveRobotics/object_tracking_2D/tree/master/3rdparty/Fdcm
//    // [Syntax] fdcm template.txt input_edgeMap.pgm input_realImage.jpg [OR]
//    // [Syntax] fdcm template.txt input_realImage.jpg
//    //const char *argv[] = {"edge_templates_generator", "-h", NULL};
//    const char *argv[] = {"edge_templates_generator", "-o", sketchPath.c_str(), "-p", OUT_DIR_PREFIX.c_str(), NULL };
//    int argc = sizeof(argv) / sizeof(char*) - 1;
//    EdgeTemplatesGenerator edgeTemplatesGenerator;
//    edgeTemplatesGenerator.generateTemplate(argc, argv);
//    testFastDirectionalChamfer("../src/cfg/hingeTemplate.txt", sketchPath);

    // fast directional chamfer matching 2 (https://github.com/whitelok/fast-directional-chamfer-matching)

}




