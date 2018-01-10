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

#include "fitline.h"
#include <iostream>
#include <string>
#include "Image/Image.h"
#include "Image/ImageIO.h"
#include "LFLineFitter.h"
#include "../../Utils/Timer.hpp"

using namespace std;

void Fitline::fitlineToLineRep(const cv::Mat &templateEdgemap, const string &outFileName)
{
    cv::Size s = templateEdgemap.size();
    int templateCols = s.width;
    int templateRows = s.height;

    // input Mat to 1D double vector
//    std::vector<double> template1D;
//    if (templateEdgemap.isContinuous()) {
//        template1D.assign(templateEdgemap.datastart, templateEdgemap.dataend);
//    } else {
//        for (int i = 0; i < templateEdgemap.rows; ++i) {
//            template1D.insert(template1D.end(), templateEdgemap.ptr<uchar>(i), templateEdgemap.ptr<uchar>(i)+templateEdgemap.cols);
//        }
//    }

    //float * testData1D = (float*)templateEdgemap; // Direct convert from 2D array to 1D array:

    // outputs
//    lineRep // prhs[0]
//    lineMap // prhs[1]

    Timer timer(true);

    // Create Image
    Image<uchar> inputImage;
    inputImage.Resize(templateCols,templateRows,false);

    int channels = templateEdgemap.channels();
    int nRows = templateEdgemap.rows;
    int nCols = templateEdgemap.cols * channels;
    if (templateEdgemap.isContinuous())
    {
        nCols *= nRows;
        nRows = 1;
    }
    int i,j;
    const uchar* p;
    for( i = 0; i < nRows; ++i)
    {
        p = templateEdgemap.ptr<uchar>(i);
        for ( j = 0; j < nCols; ++j)
        {
            inputImage.Access(j, i) = p[j];
        }
    }
    timer.printTime("Fitline: Image Creation");


//    int    row,col;
//    for (col=0; col < templateCols; col++)
//    {
//        for (row=0; row < templateRows; row++)
//        {
////            inputImage.Access(col,row) = template1D.at(row+col*templateRows);
////            inputImage.Access(col,row) = templateEdgemap.at<uchar>(col, row);
//        }
//    }

//    int nLayer = 2;
//    int nLinesToFitInStage[2];
//    int nTrialsPerLineInStage[2];
//    nLinesToFitInStage[0] = (int)floor(N_LINES_TO_FIT_IN_STAGE_1);
//    nLinesToFitInStage[1] = (int)floor(N_LINES_TO_FIT_IN_STAGE_2);
//    nTrialsPerLineInStage[0] = (int)floor(N_TRIALS_PER_LINE_IN_STAGE_1);
//    nTrialsPerLineInStage[1] = (int)floor(N_TRIALS_PER_LINE_IN_STAGE_2);

    LFLineFitter lf;
//    lf.Configure(SIGMA_FIT_A_LINE,SIGMA_FIND_SUPPORT,MAX_GAP,nLayer,nLinesToFitInStage,nTrialsPerLineInStage);
    lf.Configure("../cfg/para_line_fitter_template.txt");
    lf.Init();
    lf.FitLine(&inputImage);
    timer.printTime("Fitline: Line Fitting");

    std::string outFilename = outFileName + "_lf.txt";

    lf.SaveEdgeMap(outFilename.c_str());

    // debug
//    std::string outputImageName = outFileName + ".pgm";
//    Image<uchar> *debugImage = lf.ComputeOuputLineImage(&inputImage);
//    lf.DisplayEdgeMap(debugImage,outputImageName.c_str());
//    delete debugImage;



}

// Andi: renamed main method
void Fitline::fitline(int argc, char *argv[])
{
	//IplImage *inputImage=NULL;
	Image<uchar> *inputImage=NULL;
	LFLineFitter lf;

	if(argc != 4)
	{
		std::cerr<<"[Syntax] fitline   input_edgeMap.pgm   output_line.txt   output_edgeMap.pgm"<<std::endl;
		exit(0);
	}

	string imageName(argv[1]);
	string outFilename(argv[2]);
	string outputImageName(argv[3]);
	
	//string imageName("data/hat_edges.pgm");
	//string outFilename("data/hat_edges.txt");
	//string outputImageName("data/hat_edges_display.pgm");

	
	//inputImage = cvLoadImage(imageName.c_str(),0);
	inputImage = ImageIO::LoadPGM(imageName.c_str());
	if(inputImage==NULL)
	{
		std::cerr<<"[ERROR] Fail in reading image "<<imageName<<std::endl;
		exit(0);
	}
	
	lf.Init();
	
	lf.Configure("../../cfg/para_template_line_fitter.txt");
	
	lf.FitLine(inputImage);

	lf.DisplayEdgeMap(inputImage,outputImageName.c_str());	

	lf.SaveEdgeMap(outFilename.c_str());

	//cvReleaseImage(&inputImage);
	delete inputImage;
};