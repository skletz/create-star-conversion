/*
Copyright 2011, Ming-Yu Liu

All Rights Reserved 

Permission to use, copy, modify, and distribute this software and 
its documentation for any non-commercial purpose is hereby granted 
without fee, provided that the above copyright notice appear in 
all copies and that both that copyright notice and this permission 
notice appear in supporting documentation, and that the name of 
the author not be used in advertising or publicity pertaining to 
distribution of the software without specific, written prior 
permission. 

THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, 
INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
ANY PARTICULAR PURPOSE. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR 
ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES 
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN 
AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING 
OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE. 
*/


//#include <cxcore.h>
#include "fdcm.h"
#include "Image/Image.h"
#include "Image/ImageIO.h"
#include "Fitline/LFLineFitter.h"
#include "LMLineMatcher.h"
#include "../../Utils/Timer.hpp"

#include <iostream>
#include <string>

void FDCM::fdcm_detect(const std::string &templateTxt, const std::string &targetImagePath, const cv::Mat &targetEdgeMap, const std::string &resultOutPath)
{

    cv::Size s = targetEdgeMap.size();
    int templateCols = s.width;
    int templateRows = s.height;

    Timer timer(true);
    // Create Image
    Image<uchar> inputImage;
    inputImage.Resize(templateCols,templateRows,false);

    int channels = targetEdgeMap.channels();
    int nRows = targetEdgeMap.rows;
    int nCols = targetEdgeMap.cols * channels;
    if (targetEdgeMap.isContinuous())
    {
        nCols *= nRows;
        nRows = 1;
    }
    int i,j;
    const uchar* p;
    for( i = 0; i < nRows; ++i)
    {
        p = targetEdgeMap.ptr<uchar>(i);
        for ( j = 0; j < nCols; ++j)
        {
            inputImage.Access(j, i) = p[j];
        }
    }
    timer.printTime("FDCM: Image Creation");
    // Load Image
    //inputImage = ImageIO::LoadPGM(edgeMapName.c_str());


    LFLineFitter lf;
    LMLineMatcher lm;
    lf.Configure("../cfg/para_line_fitter_target.txt");
    lm.Configure("../cfg/para_line_matcher.txt");


    lf.Init();
    // Line Fitting
    lf.FitLine(&inputImage);
    timer.printTime("FDCM: Fitline");



    lm.Init(templateTxt.c_str());
    timer.printTime("FDCM: LM init");
    vector< vector<LMDetWind> > detWindArrays;
    detWindArrays.clear();
    lm.SingleShapeDetectionWithVaryingQuerySize(lf, 0.12, detWindArrays);
    int last = detWindArrays.size()-1;
    int nDetWindows = detWindArrays[last].size();
    timer.printTime("FDCM: LM Shape Detection");



    cv::Mat debugImage;
    debugImage = cv::imread(targetImagePath.c_str(), cv::IMREAD_COLOR); // Read the file
    for(int i=0;i<nDetWindows;i++)
    {
//        out[i+0*nDetWindows] = 1.0*detWindArrays[last][i].x_;
//        out[i+1*nDetWindows] = 1.0*detWindArrays[last][i].y_;
//        out[i+2*nDetWindows] = 1.0*detWindArrays[last][i].width_;
//        out[i+3*nDetWindows] = 1.0*detWindArrays[last][i].height_;
//        out[i+4*nDetWindows] = 1.0*detWindArrays[last][i].cost_;
//        out[i+5*nDetWindows] = 1.0*detWindArrays[last][i].count_;

        std::cout << "x " + std::to_string(detWindArrays[last][i].x_) + " Y " + std::to_string(detWindArrays[last][i].y_) + ", " +  std::to_string(detWindArrays[last][i].width_) + " x " + std::to_string(detWindArrays[last][i].height_)  << std::endl;

        cv::rectangle(debugImage, cv::Point( detWindArrays[last][i].x_, detWindArrays[last][i].y_ ), cv::Point(detWindArrays[last][i].x_ + detWindArrays[last][i].width_, detWindArrays[last][i].y_ + detWindArrays[last][i].height_),cv::Scalar( 0, 255, 0 ),3);

    }
//    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE ); // Create a window for display.
//    cv::imshow( "Display window", debugImage );                // Show our image inside it.
//    cv::waitKey(0);
    cv::imwrite(resultOutPath, debugImage);
    debugImage.release();
}

// Andi: renamed main method
void FDCM::fdcm(int argc, const char *argv[])
{
	
	//if(argc < 4)
	//{
	//	//std::cerr<<"[Syntax] fdcm template.txt input_edgeMap.pgm input_realImage.jpg"<<std::endl;
	//	std::cerr<<"[Syntax] fdcm template.txt input_edgeMap.pgm input_realImage.ppm"<<std::endl;
	//	exit(0);
	//}

	LFLineFitter lf;
	LMLineMatcher lm;
	lf.Configure("../cfg/para_line_fitter_target.txt");
	lm.Configure("../cfg/para_line_matcher.txt");

	
	//Image *inputImage=NULL;
	Image<uchar> *inputImage=NULL;
	
	string templateFileName(argv[1]);
	string edgeMapName(argv[2]);
	string displayImageName(argv[3]);
	
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
	inputImage = ImageIO::LoadPGM(edgeMapName.c_str());

	
	if(inputImage==NULL)
	{
		std::cerr<<"[ERROR] Fail in reading image "<<edgeMapName<<std::endl;
		exit(0);
	}

	lf.Init();

	lm.Init(templateFileName.c_str());

	// Line Fitting
	lf.FitLine(inputImage);

	// FDCM Matching (original)
	vector<LMDetWind> detWind;
	lm.Match(lf,detWind);
    std::cout << "Found Matches: " + std::to_string(detWind.size()) << std::endl;
    std::cout << "x " + std::to_string(detWind[0].x_) + " Y " + std::to_string(detWind[0].y_) + ", " +  std::to_string(detWind[0].width_) + " x " + std::to_string(detWind[0].height_)  << std::endl;


	//lm.MatchCostMap(lf,outputCostMapName.c_str());
	// Display best matcher in edge map
	if(displayImageName.c_str())
	{
//		Image<RGBMap> *debugImage = ImageIO::LoadPPM(displayImageName.c_str());
//		LMDisplay::DrawDetWind(debugImage,detWind[0].x_,detWind[0].y_,detWind[0].width_,detWind[0].height_,RGBMap(0,255,0),4);
//		char outputname[256];
//		sprintf(outputname,"../out/%s.output.ppm",displayImageName.c_str());
//		ImageIO::SavePPM(debugImage,outputname);
//        delete debugImage;

        cv::Mat debugImage;
        debugImage = cv::imread(displayImageName.c_str(), cv::IMREAD_COLOR); // Read the file
        cv::rectangle(debugImage, cv::Point( detWind[0].x_, detWind[0].y_ ), cv::Point(detWind[0].x_+detWind[0].width_,detWind[0].y_+detWind[0].height_),cv::Scalar( 0, 255, 0 ),3);

        cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE ); // Create a window for display.
        cv::imshow( "Display window", debugImage );                // Show our image inside it.
        cv::waitKey(0);
        debugImage.release();

	}



	//cvReleaseImage(&inputImage);
	delete inputImage;
};
