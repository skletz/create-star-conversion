//
// Created by bns on 12/27/17.
//

#include <iostream>
#include <limits>

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <opencv2/opencv.hpp>
#include <Fitline/fitline.h>
#include "chamfer/Chamfer.hpp"
#include "chamfer/Utils.hpp"
#include "Utils/Timer.hpp"
#include "Fdcm/fdcm.h"

// PATHS
//std::string DATA_LOCATION_PREFIX = "../data/";
std::string SKETCH_DATA_LOCATION_PREFIX = "../../data/sketch_sample/";
std::string ETHZ_DATA_LOCATION_PREFIX = "../../data/ETHZShapeClasses-V1.2/";
std::string OUT_DIR_PREFIX = "../out/";

// CANNY CFG
//a. We establish a ratio of lower:upper threshold of 3:1 (with the variable *ratio*)
//b. We set the kernel size of :math:`3` (for the Sobel operations to be performed internally by the Canny function)
//c. We set a maximum value for the lower Threshold of :math:`100`.
const int CANNY_KERNEL_SIZE = 3;
const int CANNY_RATIO = 3; // 1:3
const int CANNY_THRESH = 40;


int testOriginalChamfer()
{
//    cv::Mat img_template = cv::imread(SKETCH_DATA_LOCATION_PREFIX + "Inria_logo_template.jpg");
//  cv::Mat img_query = cv::imread(SKETCH_DATA_LOCATION_PREFIX + "Inria_scene.jpg");
//  cv::Mat img_query = cv::imread(SKETCH_DATA_LOCATION_PREFIX + "Inria_scene2.jpg");
//  cv::Mat img_query = cv::imread(SKETCH_DATA_LOCATION_PREFIX + "Inria_scene3.jpg");
//  cv::Mat img_query = cv::imread(SKETCH_DATA_LOCATION_PREFIX + "Inria_scene4.jpg");
//    cv::Mat img_query = cv::imread(SKETCH_DATA_LOCATION_PREFIX + "Inria_scene5.jpg");

    cv::Mat img_template = cv::imread(SKETCH_DATA_LOCATION_PREFIX + "sketches/0.png");
    cv::Mat img_query = cv::imread(SKETCH_DATA_LOCATION_PREFIX + "images/0/641602006.jpg");

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

/**
 * @function CannyThreshold
 * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
 */
void myCanny(const cv::Mat &inputImg, cv::Mat &resultEdgeMap, int lowerthresh)
{
    /// Reduce noise with a kernel 3x3
    cv::blur( inputImg, resultEdgeMap, cv::Size(CANNY_KERNEL_SIZE,CANNY_KERNEL_SIZE) );

    /// Canny detector
    cv::Canny( resultEdgeMap, resultEdgeMap, lowerthresh, lowerthresh * CANNY_RATIO, CANNY_KERNEL_SIZE );

    //cv::src.copyTo( dst, detected_edges);
    //imshow( window_name, dst );
}

void saveImage(std::string &path, cv::Mat &img)
{
    if(cv::imwrite(path, img))
    {
        std::cout << "Saved " << path << std::endl;
    }

}

void testFDCM(const boost::filesystem::path sourceImgPath, const boost::filesystem::path &targetImgPath)
{
    // template
    cv::Mat sourceImg = cv::imread(sourceImgPath.string());
    cv::Mat templateImg;
    cv::cvtColor( sourceImg, templateImg, cv::COLOR_BGR2GRAY );
    cv::Mat templateEdgeMap;
    myCanny(templateImg, templateEdgeMap, CANNY_THRESH);

    // query
    cv::Mat targetImageColor = cv::imread(targetImgPath.string());
    cv::Mat targetImgEdgeMap;
    cv::cvtColor( targetImageColor, targetImgEdgeMap, cv::COLOR_BGR2GRAY );
    myCanny(targetImgEdgeMap, targetImgEdgeMap, CANNY_THRESH);

//    cv::namedWindow( "as window", cv::WINDOW_AUTOSIZE ); // Create a window for display.
//    cv::imshow( "as window", targetImgEdgeMap );                // Show our image inside it.
//    cv::waitKey(0);

    // convert edge map into line representation (template)
    Fitline fl;
    std::string outFileName = OUT_DIR_PREFIX + sourceImgPath.stem().string();
    std::string fdcmTemplatePgm = outFileName + ".pgm";
    fl.fitlineToLineRep(templateEdgeMap, outFileName );
    saveImage(fdcmTemplatePgm, templateEdgeMap);
    // write template file (should contain enumerated references to contours?)
    std::ofstream templateFile;
    std::string fdcmTemplateTxt = outFileName + "_template.txt";
    templateFile.open (fdcmTemplateTxt);
    templateFile << "1\n";
    templateFile << outFileName + "_lf.txt" + "\n";
    templateFile.close();
    std::cout << "Saved " << fdcmTemplateTxt << std::endl;

    // convert edge map into line representation (target)
    std::string outFileNameTarget = OUT_DIR_PREFIX + targetImgPath.stem().string();
    std::string fdcmTargetPgm = outFileNameTarget + ".pgm";
//    fl.fitlineToLineRep(targetImgEdgeMap, outFileNameTarget );
    saveImage(fdcmTargetPgm, targetImgEdgeMap);


    const char *argv[] = {"fdcm", fdcmTemplateTxt.c_str(), fdcmTargetPgm.c_str(), targetImgPath.c_str(), NULL };
    int argc = sizeof(argv) / sizeof(char*) - 1;
    FDCM fdcm;
//    fdcm.fdcm(argc, argv);
    const std::string resultOutpath = OUT_DIR_PREFIX + "results/"+ targetImgPath.stem().string() + "_result.jpg";
    std::cout << "out" <<resultOutpath << std::endl;
//    cv::Mat tempEdge;
//    cv::imread(OUT_DIR_PREFIX + "black2_edges.tif");
//    cv::cvtColor( tempEdge, tempEdge, CV_BGR2GRAY );
//    cv::Mat binaryMat(tempEdge.size(), tempEdge.type());
//    cv::threshold(tempEdge, binaryMat, 100, 255, cv::THRESH_BINARY);
//    cv::namedWindow("Output", cv::WINDOW_AUTOSIZE);
//    cv::imshow("Output", binaryMat);
//
//    cv::waitKey(0);


    fdcm.fdcm_detect(fdcmTemplateTxt, targetImgPath.string(), targetImgEdgeMap, resultOutpath);

    // cleanup
    sourceImg.release();
    templateImg.release();
    templateEdgeMap.release();
    targetImageColor.release();
    targetImgEdgeMap.release();
}

void createDir(boost::filesystem::path path)
{
    if (!boost::filesystem::is_directory(path))
    {
        if (boost::filesystem::create_directories(path))
        {
            std::cout << "Created dir: " + path.string() << std::endl;
        }
        else
        {
            std::cout << "Failed to create dir: " + path.string() << std::endl;
        }
    }
}

int main()
{

    // create out paths
    boost::filesystem::path out{OUT_DIR_PREFIX};
    boost::filesystem::path out_results{OUT_DIR_PREFIX + "results/"};
    createDir(out);
    createDir(out_results);


    // query image path
//    boost::filesystem::path sketchPath{SKETCH_DATA_LOCATION_PREFIX + "sketches/0.png"};
//    boost::filesystem::path temptargetPath{SKETCH_DATA_LOCATION_PREFIX + "images/0/1429542167.jpg"};
    boost::filesystem::path sketchPath{"../../sbss/src/FDCM_2/DemoImg/template.png"};
    boost::filesystem::path temptargetPath{"../../sbss/src/FDCM_2/DemoImg/matching_target.jpg"};
//    boost::filesystem::path sketchPath{ETHZ_DATA_LOCATION_PREFIX + "Swans/big_swans_outlines.png"};
//    boost::filesystem::path temptargetPath{ETHZ_DATA_LOCATION_PREFIX + "Swans/big.jpg"};

    // database paths
    std::vector<boost::filesystem::path> files;
    static const std::string extArray[] = {".PNG", ".png", ".JPG", ".jpg", ".JPEG", ".jpeg", ".BMP", ".bmp"};
    std::vector<std::string> extVector(extArray, extArray + sizeof(extArray) / sizeof(extArray[0]));
    getAllFilesWithExtensions(SKETCH_DATA_LOCATION_PREFIX + "images/", extVector, files);


//    // test original implementation (chamfer folder) -> https://github.com/s-trinh/Chamfer-Matching
//    testOriginalChamfer();

    // original fast directioonal chamfer matching FDCM -> https://github.com/mingyuliutw/FastDirectionalChamferMatching


//    // test original implementation with query image and database
//    std::vector<std::pair<int, boost::filesystem::path>> detectionsMap; // detections -> path
//    getDetections(sketchPath.string(), files, detectionsMap);


    // fast directional chamfer matching 2 (https://github.com/whitelok/fast-directional-chamfer-matching)
//    testFDCM(sketchPath, files.at(5));
    testFDCM(sketchPath, temptargetPath);
}




