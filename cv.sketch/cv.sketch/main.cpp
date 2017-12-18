//
//  Shape context
//  main.cpp
//  cv.sketch
//
//  Created by Sabrina Kletz on 15.12.17.
//  Copyright © 2017 Sabrina Kletz. All rights reserved.
//
#include "opencv2/shape.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/core/utility.hpp>
#include <iostream>
#include <string>
#include <dirent.h>
#include <algorithm>


using namespace std;
using namespace cv;


static void help()
{
    printf("\n"
           "This program demonstrates methods for shape comparisson based on shape context\n"
           "Run the program providing a number between 0 and 48 for selecting a sketch in the folder ../sketches/.\n"
           "Call\n"
           "./main [number between 0 and 48, 0 default]\n\n");
}

static int MAX_QUERYW = 512;
static int SCREEN_HEIGHT = 1900;
static int ROWS_RESULTS = 10;
static int MAX_RESULTS = 100;

//How many directories?
static int MAX_IN = 10;
//How many images per directory?
static int NR_DIR = 10;

static int POINTS = 1500;


struct Match{
    std::string path;
    float dist;
    cv::Mat image;
};

/**
 * @brief makeCanvas Makes composite image from the given images
 * @param vecMat Vector of Images.
 * @param windowHeight The height of the new composite image to be formed.
 * @param nRows Number of rows of images. (Number of columns will be calculated
 *              depending on the value of total number of images).
 * @return new composite image.
 */
cv::Mat makeCanvas(std::vector<cv::Mat>& vecMat, int windowHeight, int nRows) {
    int N = int(vecMat.size());
    nRows  = nRows > N ? N : nRows;
    int edgeThickness = 10;
    int imagesPerRow = ceil(double(N) / nRows);
    int resizeHeight = floor(2.0 * ((floor(double(windowHeight - edgeThickness) / nRows)) / 2.0)) - edgeThickness;
    int maxRowLength = 0;
    
    std::vector<int> resizeWidth;
    for (int i = 0; i < N;) {
        int thisRowLen = 0;
        for (int k = 0; k < imagesPerRow; k++) {
            double aspectRatio = double(vecMat[i].cols) / vecMat[i].rows;
            int temp = int( ceil(resizeHeight * aspectRatio));
            resizeWidth.push_back(temp);
            thisRowLen += temp;
            if (++i == N) break;
        }
        if ((thisRowLen + edgeThickness * (imagesPerRow + 1)) > maxRowLength) {
            maxRowLength = thisRowLen + edgeThickness * (imagesPerRow + 1);
        }
    }
    int windowWidth = maxRowLength;
    cv::Mat canvasImage(windowHeight, windowWidth, CV_8UC3, Scalar(0, 0, 0));
    
    for (int k = 0, i = 0; i < nRows; i++) {
        int y = i * resizeHeight + (i + 1) * edgeThickness;
        int x_end = edgeThickness;
        for (int j = 0; j < imagesPerRow && k < N; k++, j++) {
            int x = x_end;
            cv::Rect roi(x, y, resizeWidth[k], resizeHeight);
            cv::Size s = canvasImage(roi).size();
            // change the number of channels to three
            cv::Mat target_ROI(s, CV_8UC3);
            if (vecMat[k].channels() != canvasImage.channels()) {
                if (vecMat[k].channels() == 1) {
                    cv::cvtColor(vecMat[k], target_ROI, CV_GRAY2BGR);
                }
            } else {
                vecMat[k].copyTo(target_ROI);
            }
            cv::resize(target_ROI, target_ROI, s);
            if (target_ROI.type() != canvasImage.type()) {
                target_ROI.convertTo(target_ROI, canvasImage.type());
            }
            target_ROI.copyTo(canvasImage(roi));
            x_end += resizeWidth[k] + edgeThickness;
        }
    }
    return canvasImage;
}

bool compareMatchesByDist(const Match &a, const Match &b)
{
    return a.dist < b.dist;
}

static vector<Point> getSimpleContours(const Mat& currentQuery, int points = POINTS)
{
    vector<vector<Point> > _contoursQuery;
    vector <Point> contoursQuery;
    cv::Mat edges;
    Canny(currentQuery, edges, 100, 200);
    cv::findContours(edges, _contoursQuery, RETR_LIST, CHAIN_APPROX_NONE);
    
    for (size_t border=0; border<_contoursQuery.size(); border++)
    {
        for (size_t p=0; p<_contoursQuery[border].size(); p++)
        {
            contoursQuery.push_back( _contoursQuery[border][p] );
        }
    }
    // In case actual number of points is less than n
    int dummy=0;
    for (int add=(int)contoursQuery.size()-1; add < points; add++)
    {
        contoursQuery.push_back(contoursQuery[dummy++]); //adding dummy values
    }
    // Uniformly sampling
    random_shuffle(contoursQuery.begin(), contoursQuery.end());
    vector<Point> cont;
    for (int i= 0; i < points; i++)
    {
        cont.push_back(contoursQuery[i]);
    }
    return cont;
}

static void drawPoints(cv::Mat bg, std::vector<cv::Point> cont, cv::Mat& output)
{
    cv::Mat contours(bg.rows, bg.cols, CV_8UC3);
    //bg.copyTo(contours);
    int radius = 3;
    for (int i = 0; i < cont.size(); i++) {
        cv::circle(contours, cv::Point(cont[i].x,cont[i].y), radius, CV_RGB(255, 255, 255));
    }
    
    contours.copyTo(output);
}

int main(int argc, char** argv)
{
    string path = "/Applications/XAMPP/xamppfiles/htdocs/GameJam/cv.sketch/cv.sketch/data/sketch_sample";
    string path_sketches = path + "/sketches/";
    string path_images = path + "/images/";
    
    cv::CommandLineParser parser(argc, argv, "{help h||}{@input|0|}");
    if (parser.has("help"))
    {
        help();
        return 0;
    }
    int indexQuery = parser.get<int>("@input");
    if (!parser.check())
    {
        parser.printErrors();
        help();
        return 1;
    }
    if (indexQuery < 0 || indexQuery > 48)
    {
        help();
        return 1;
    }
    
    printf("Sketch-based matching running ...\n");
    
    cv::Ptr <cv::HausdorffDistanceExtractor> scde = cv::createHausdorffDistanceExtractor();
    stringstream queryName;
    queryName << path_sketches << indexQuery << ".png";
    printf("Input sketch %s \n", queryName.str().c_str());
    
    cv::Mat input = cv::imread(queryName.str(), IMREAD_GRAYSCALE);
    cv::Mat input_small;
    float imgScale = MAX_QUERYW / float(input.cols);
    int width = input.cols * imgScale;
    int height = input.rows * imgScale;
    cv::resize(input, input_small, cv::Size(width, height));
    
    cv::namedWindow("Sketch Query", WINDOW_NORMAL);
    cv::moveWindow("Sketch Query", 0, 0);
    cv::imshow("Sketch Query", input_small);
    
    
    //Get contours of input sketch (query)
    std::vector<cv::Point> contoursQuery = getSimpleContours(input);
    
    cv::Mat contours, contours_small;
    drawPoints(input, contoursQuery, contours);
    cv::resize(contours, contours_small, cv::Size(width, height));
    
    cv::namedWindow("Sketch Contours");
    cv::moveWindow("Sketch Contours", 0, input_small.rows);
    cv::imshow("Sketch Contours", contours_small);
    
    
    std::vector<Match> matches;
    int counter;
    for(int i = 0; i < MAX_IN; i++)
    {
        stringstream dbDir;
        dbDir << path_images << i << "/";
        
        DIR *dp;
        struct dirent * dirp;

        if((dp  = opendir(dbDir.str().c_str())) == NULL)
        {
            printf("Cannot open directory! %s \n", dbDir.str().c_str());
            continue;
        }
        
        std::vector<std::string> files;
        counter = NR_DIR;
        while(counter != 0 && (dirp = readdir(dp)) != NULL)
        {
            std::string file = std::string(dirp->d_name);
            size_t pos = file.find(".jpg");
            
            if(pos != std::string::npos){
                files.push_back(std::string(dirp->d_name));
                counter--;
            }

        }
        
        closedir(dp);
        
        for(int j = 0; j < files.size(); j++)
        {
            printf("File: %s\n", files.at(j).c_str());
         
            stringstream dbName;
            dbName << path_images << i << "/" << files.at(j);
            printf("Read file: %s\n", dbName.str().c_str());
            cv::Mat db = imread(dbName.str());
            cv::Mat db_gray;
            cv::cvtColor(db, db_gray, CV_RGB2GRAY);
            std::vector<cv::Point> contoursDb = getSimpleContours(db_gray);
            cv::Mat contours;
            drawPoints(db, contoursDb, contours);
            
            float start = cv::getTickCount();
            float dist = scde->computeDistance(contoursQuery, contoursDb);
            //float dist = 0;
            float end = cv::getTickCount();
            float t = ((end - start) / cv::getTickCount()) * 1000;
            printf("Time: %lf\n", t);
            printf("Dist: %lf\n", dist);
            
            Match match;
            match.path = files.at(j);
            match.dist = dist;
            match.image = db;
            matches.push_back(match);
            Match features;
            features.path = files.at(j);
            features.dist = dist;
            features.image = contours;
            matches.push_back(features);
        }
    }
    
    std::sort(matches.begin(), matches.end(), compareMatchesByDist);

    std::vector<cv::Mat> images;
    
    for(int i = 0; i < MAX_RESULTS; i++)
    {
        images.push_back(matches.at(i).image);
    }

    cv::Mat results = makeCanvas(images, SCREEN_HEIGHT, ROWS_RESULTS);
    cv::namedWindow("Results");
    cv::moveWindow("Results", MAX_QUERYW, 0);

    cv::imshow("Results", results);
    cv::waitKey(0);
}
