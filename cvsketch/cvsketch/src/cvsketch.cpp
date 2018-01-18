#include "cvsketch.hpp"

//Third-Party libraries
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <boost/filesystem.hpp>

//Project dependencies
#include "segmentation.hpp"
#include "matching.hpp"

//External source code
#include "../libs/imageSegmentation/imageSegmentation.hpp"


vbs::cvSketch::cvSketch()
{

#if DEBUG
    std::cout << "cvSketch contstructor ..." << std::endl;
#endif

}

cv::Ptr<cv::ximgproc::SuperpixelSEEDS> vbs::Segmentation::seeds;
std::map<cv::Vec3b, int, vbs::lessVec3b> vbs::Segmentation::query_colorpalette;
std::vector<std::pair<cv::Vec3b, int>> vbs::Segmentation::sorted_query_colorpalette;

std::string vbs::cvSketch::help(const boost::program_options::options_description& desc)
{
    std::stringstream help;

    help << "============== Help ==============" << std::endl;
    help << "INFO: This program tests sketch-based image retrival methods ..." << std::endl;
    help << "INFO: Call ./cvsketch_demo --input [path]" << std::endl;
    help << "============== Help ==============" << std::endl;
    help << desc << std::endl;
    return help.str();
}

std::string vbs::cvSketch::getInfo()
{
    std::stringstream info;
    info << "============== Info ==============" << std::endl;
    info << "============= OpenCV: ============" << std::endl;
    info << "INFO: Using OpenCV " << CV_VERSION << std::endl;
    info << "============= Boost: =============" << std::endl;

    info << "INFO: Using Boost "
    << BOOST_VERSION / 100000     << "."  // major version
    << BOOST_VERSION / 100 % 1000 << "."  // minor version
    << BOOST_VERSION % 100                // patch level
    << std::endl;
    info << "============== Info ==============" << std::endl;

    return info.str();
}

bool vbs::cvSketch::init(boost::program_options::variables_map _args)
{
	if(verbose)
		std::cout << "cvSketch init ..." << std::endl;

	if (_args.find("verbose") != _args.end())
	{
		verbose = true;
	}

	if (_args.find("display") != _args.end())
	{
		display = true;
	}

	if(_args.find("searchin") != _args.end())
	{
		searchin = _args["searchin"].as<std::string>();
		std::cout << "SEARCH IN: " << searchin << std::endl;
	}

    input = _args["input"].as<std::string>();
	output = _args["output"].as<std::string>();

	if (!boost::filesystem::is_directory(output)) {
		boost::filesystem::create_directory(output);
	}

    std::cout << "INPUT: " << input << std::endl;
    std::cout << "OUTPUT: " << output << std::endl;

    return true;
}

void on_trackbar_colorReduction_kMeans(int, void* data)
{
    vbs::SettingsKmeansCluster settings = *static_cast<vbs::SettingsKmeansCluster*>(data);
    std::string winname = settings.winname;
    cv::Mat src, dst;
    settings.image.copyTo(src);
    int k = settings.kvalue;

    if(settings.initDone)
    {
        settings.reducedImage.copyTo(dst);
    }else
    {
        if(settings.kvalue > 0)
        {
            vbs::cvSketch::reduceColors(src, k, dst);
        }
        else{
            src.copyTo(dst);
        }

        //Update also color chart
        //get color-palette of the image
        vbs::Segmentation::query_colorpalette = vbs::Segmentation::getPalette(dst);
        std::vector<std::pair<cv::Vec3b, int>> sorted_colorpalette;
        vbs::Segmentation::sortPaletteByArea(vbs::Segmentation::getPalette(dst), sorted_colorpalette);
        vbs::Segmentation::sorted_query_colorpalette = sorted_colorpalette;

        cv::Mat colorchart;
        vbs::cvSketch::getColorchart(vbs::Segmentation::sorted_query_colorpalette, colorchart, src.cols, 50, (dst.cols * dst.rows));
        cv::cvtColor(colorchart, colorchart, COLOR_Lab2BGR);
        cv::imshow(settings.winnameColorchart, colorchart);
        dst.copyTo(settings.reducedImage);

        //Update also quantized color image
        cv::Mat quantizedImage;
        vbs::cvSketch::quantizeColors(src, settings.labels, settings.num_labels, quantizedImage, vbs::Segmentation::query_colorpalette);
        cv::cvtColor(quantizedImage, quantizedImage, cv::COLOR_Lab2BGR);
        cv::imshow(settings.winnameQuantizedColors, quantizedImage);
    }

    cv::cvtColor(dst, dst, cv::COLOR_Lab2BGR);

    cv::Mat dst_show;
    dst.copyTo(dst_show);
    std::stringstream text;
    text << "Colors: " << settings.kvalue;

    cv::putText(dst_show, text.str(), cvPoint(30, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.4, cv::Scalar(200, 200, 250), 1, CV_AA);

    cv::imshow(winname, dst_show);

}

void on_trackbar_superpixel_SEEDS(const int, void* data)
{
//	const cv::Mat src = *static_cast<cv::Mat*>(data);
    vbs::SettingsSuperpixelSEEDS settings = *static_cast<vbs::SettingsSuperpixelSEEDS*>(data);
    std::string winname = settings.winname;

    cv::Mat src, labels, dst, mask;
    int num_superpixel_found;

    settings.image.copyTo(src);
    int num_superpixels = settings.num_superpixels;
    int prior = settings.prior;
    int num_levels = settings.num_levels;
    bool double_step = bool(settings.double_step);
    int num_iterations = settings.num_iterations;
    int num_histogram_bins = settings.num_histogram_bins;

    if(settings.initDone)
    {
        settings.superpixelImage.copyTo(dst);
    }else
    {
        vbs::cvSketch::extractSuperpixels(src, labels, mask, num_superpixel_found, num_superpixels, num_levels, prior, num_histogram_bins, double_step, num_iterations);
        src.copyTo(dst);
        dst.setTo(cv::Scalar(0,0,255), mask);

        dst.copyTo(settings.superpixelImage);

        //Update also quantized color image
        cv::Mat quantizedImage;
        vbs::cvSketch::quantizeColors(src, labels, num_superpixel_found, quantizedImage, vbs::Segmentation::query_colorpalette);
        cv::cvtColor(quantizedImage, quantizedImage, cv::COLOR_Lab2BGR);
        cv::imshow(settings.winnameQuantizedColors, quantizedImage);
    }

    cv::cvtColor(dst, dst, cv::COLOR_Lab2BGR);

    cv::Mat dst_show;
    dst.copyTo(dst_show);
    std::stringstream text;
    text << "Superpixels: " << num_superpixels << ", ";
    text << "Prior: " << prior  << ", ";
    text << "Levels: " << num_levels;

    cv::putText(dst_show, text.str(), cvPoint(30, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.4, cv::Scalar(200, 200, 250), 1, CV_AA);

    text.str("");
    text << "Double: " << double_step  << ", ";
    text << "Iter: " << num_iterations  << ", ";
    text << "Hist Bins: " << num_histogram_bins;

    cv::putText(dst_show, text.str(), cvPoint(30, 40), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.4, cv::Scalar(200, 200, 250), 1, CV_AA);

    cv::imshow(winname, dst_show);
}

void vbs::cvSketch::testColorSegmentation(cv::Mat& image, cv::Mat& colorSegments, cv::Mat& colorLabels, std::map<cv::Vec3b, int, lessVec3b>& palette)
{

    if(verbose)
        std::cout << "cvSketch testColorSegmentation ..." << std::endl;

    cv::Mat input;
    image.copyTo(input);

    //exection time measure
    double t = 0.0;

    const int width = image.cols;
    const int height = image.rows;

    std::cout << "Convert colors to LAB ..." << std::endl;
    t = double(cv::getTickCount());

    //Convert to LAB color space
    cv::cvtColor(image, input, cv::COLOR_BGR2Lab);

    t = (double(cv::getTickCount()) - t) / cv::getTickFrequency();
    printf("Color conversion took %i ms with %3ix%3i px resoultion \n",int(t * 1000), width, height);

    std::cout << "Reduce colors using k-means ..." << std::endl;
    int initKValue = 4;
    cv::Mat reducedColorImage;
    //Cluster colors in LAB color space using k-means
    reduceColors(input, initKValue, reducedColorImage);

    std::map<cv::Vec3b, int, lessVec3b> colorpalette;
    //get color-palette of the image
    colorpalette = vbs::Segmentation::getPalette(reducedColorImage);

    std::vector<std::pair<cv::Vec3b, int>> sorted_colorpalette;
    vbs::Segmentation::sortPaletteByArea(colorpalette, sorted_colorpalette);

    std::cout << "Create barchart of k dominant colors ..." << std::endl;
    cv::Mat colorchart;
    getColorchart(sorted_colorpalette, colorchart, input.cols, 50, (width * height));

    std::cout << "Create superpixels using SEEDS ..." << std::endl;
    int num_init_superpixels = 1000;
    int prior = 5;
    int num_levels = 10;
    int num_iterations = 12;
    bool double_step = false;
    int num_histogram_bins = 10;

    cv::Mat superpixelImage, mask, labels;
    int num_superpixels_found;
    //Extract superpixels using Energy Driven Sampling
    extractSuperpixels(input, labels, mask, num_superpixels_found, num_init_superpixels, num_levels, prior, num_histogram_bins, double_step, num_iterations);
    input.copyTo(superpixelImage);
    superpixelImage.setTo(cv::Scalar(0,0,255), mask);


    cv::Mat quantizedColorImage;

    std::map<cv::Vec3b, int, lessVec3b> empty;
    quantizeColors(reducedColorImage, labels, num_superpixels_found, quantizedColorImage, colorpalette);

    //cv::Mat colorchart_default;
    //getDefaultColorchart(vbs::default_palette_rgb, colorchart_default, input.cols, 50);


    if(display){

        //int win_h = input.rows;
        int win_w = input.cols;

        std::string winnameOrig = "Original";
        std::string winnameChart = "Color Chart";
        std::string winnameQuantizedColors = "Quantized Color Image";

        cv::namedWindow(winnameOrig);
        cv::moveWindow(winnameOrig, pad_left, pad_top);
        cv::imshow(winnameOrig, image);
        //cv::cvtColor(reducedColorImage, reducedColorImage, cv::COLOR_Lab2BGR);

        cv::namedWindow(winnameChart);
        cv::moveWindow(winnameChart, pad_left + win_w * 4, pad_top);

        cv::cvtColor(colorchart, colorchart, COLOR_Lab2BGR);
        cv::imshow(winnameChart, colorchart);

        //cv::cvtColor(colorchart_default, colorchart_default, COLOR_Lab2BGR);
        //cv::imshow("Default Color Chart", colorchart_default);


        //TESTBED for k-means clustering
        SettingsKmeansCluster* set_kmeans = new SettingsKmeansCluster();
        set_kmeans->winname = "Reduced Colors";
        set_kmeans->kvalue = initKValue;
        input.copyTo(set_kmeans->image);
        set_kmeans->kvalue_max = 10;
        set_kmeans->initDone = true;
        reducedColorImage.copyTo(set_kmeans->reducedImage);

        set_kmeans->winnameColorchart = winnameChart;
        set_kmeans->winnameQuantizedColors = winnameQuantizedColors;
        set_kmeans->labels = labels;
        set_kmeans->num_labels = num_superpixels_found;

        cv::namedWindow(set_kmeans->winname, 1);
        cv::moveWindow(set_kmeans->winname, pad_left + win_w, pad_top);
        //cv::imshow(set_kmeans->winname, reducedColorImage);
        cv::createTrackbar("Colors", set_kmeans->winname, &set_kmeans->kvalue, set_kmeans->kvalue_max, on_trackbar_colorReduction_kMeans, set_kmeans);
        on_trackbar_colorReduction_kMeans(0, set_kmeans);
        set_kmeans->initDone = false;

        //TESTBED for SEEDS superpixles
        SettingsSuperpixelSEEDS* set_seeds = new SettingsSuperpixelSEEDS();
        set_seeds->winname = "Superpixels";
        set_seeds->num_superpixels = num_init_superpixels;
        set_seeds->num_superpixels_max = 1000;
        set_seeds->prior = prior;
        set_seeds->prior_max = 10;
        set_seeds->num_levels = num_levels;
        set_seeds->num_levels_max = 10;
        set_seeds->num_iterations = num_iterations;
        set_seeds->num_iterations_max = 25;
        set_seeds->double_step = int(double_step);
        set_seeds->num_histogram_bins = num_histogram_bins;
        set_seeds->num_histogram_bins_max = 10;

        input.copyTo(set_seeds->image);
        set_seeds->initDone = true;
        superpixelImage.copyTo(set_seeds->superpixelImage);
        set_seeds->winnameQuantizedColors = winnameQuantizedColors;

        cv::namedWindow(set_seeds->winname, 1);
        cv::moveWindow(set_seeds->winname, pad_left + win_w * 2, pad_top);
        //cv::imshow(set_kmeans->winname, superpixels);
        cv::createTrackbar("Superpixels", set_seeds->winname, &set_seeds->num_superpixels, set_seeds->num_superpixels_max, on_trackbar_superpixel_SEEDS, set_seeds);
        cv::createTrackbar("Prior", set_seeds->winname, &set_seeds->prior, set_seeds->prior_max, on_trackbar_superpixel_SEEDS, set_seeds);
        cv::createTrackbar("Levels", set_seeds->winname, &set_seeds->num_levels, set_seeds->num_levels_max, on_trackbar_superpixel_SEEDS, set_seeds);
        cv::createTrackbar("Double Step", set_seeds->winname, &set_seeds->double_step, 1, on_trackbar_superpixel_SEEDS, set_seeds);
        cv::createTrackbar("Hist Bins", set_seeds->winname, &set_seeds->num_histogram_bins, set_seeds->num_histogram_bins_max, on_trackbar_superpixel_SEEDS, set_seeds);
        cv::createTrackbar("Interations", set_seeds->winname, &set_seeds->num_iterations, set_seeds->num_iterations_max, on_trackbar_superpixel_SEEDS, set_seeds);

        on_trackbar_superpixel_SEEDS(0, set_seeds);
        set_seeds->initDone = false;

        cv::namedWindow(winnameQuantizedColors, 1);
        cv::moveWindow(winnameQuantizedColors, pad_left + win_w * 3, pad_top);
        cv::cvtColor(quantizedColorImage, quantizedColorImage, cv::COLOR_Lab2BGR);
        cv::imshow(winnameQuantizedColors, quantizedColorImage);


        int c = cv::waitKey(0);
        while((c & 255) != 'q' && c != 'Q' && (c & 255) != 27)
        {
            if (c == 's')
            {
                cv::cvtColor(set_kmeans->reducedImage, set_kmeans->reducedImage, cv::COLOR_Lab2BGR);
                cv::imshow("Save: ", set_kmeans->reducedImage);
            }
            c = cv::waitKey(0);
        }

        labels.copyTo(colorLabels);
        quantizedColorImage.copyTo(colorSegments);
        palette = colorpalette;
    }
}

void vbs::cvSketch::describeColorSegmentation(cv::Mat& image, cv::Mat& colorSegments, cv::Mat& colorLabels, std::map<cv::Vec3b, int, lessVec3b>& palette, cv::Mat& descriptors)
{
    cv::Mat describtor;

    //hist(image);
    //int winSize=20;
    //pair<Mat, vector<pair<pair<Point, Vec3b> ,int > > >retVal=
    //cv::Mat combined = colSeg(image, winSize);

    //cv::cvtColor(colorSegments, colorSegments, cv::COLOR_BGR2Lab);
    //cv::imshow("Color Segments", colorSegments);
    //crop(colorSegments, image);
    RNG rng(12345);
    for(auto color : palette)
    {
        cv::Mat mask;
        cv::Vec3b lab = color.first;
        cv::Scalar rgb_scalar = vbs::Segmentation::ScalarLAB2BGR(color.first[0], color.first[1], color.first[2]);

        cv::inRange(colorSegments, rgb_scalar, rgb_scalar, mask);

        //colorSegments.setTo(cv::Scalar(255,255,255), colorSegments = bgr);
        cv::Mat c(50,50, CV_8UC3, rgb_scalar);

        cv::imshow("Color Segments", colorSegments);
        cv::imshow("Color", c);
        cv::imshow("Mask", mask);

        std::vector<std::vector<cv::Point> > contours;
        cv::findContours(mask, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);


        std::vector<std::vector<cv::Point> > contours_poly( contours.size() );
        std::vector<cv::Rect> boundRect( contours.size() );
        std::vector<Point2f>center( contours.size() );
        std::vector<float>radius( contours.size() );

        for( int i = 0; i < contours.size(); i++ )
        {
            approxPolyDP( Mat(contours[i]), contours_poly[i], 10, true );
            boundRect[i] = boundingRect( Mat(contours_poly[i]) );
            minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
        }

        cv::Mat drawing = cv::Mat::zeros( mask.size(), CV_8UC3 );
        for( int i = 0; i< contours.size(); i++ )
        {
            cv::Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            drawContours( drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
            rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
            //circle( drawing, center[i], (int)radius[i], color, 2, 8, 0 );
        }

        cv::imshow("Contours", drawing);

        waitKey(0);
    }


}

//Input - Lab color space
void vbs::cvSketch::reduceColors(const cv::Mat& image, int kvalue, cv::Mat& output)
{
    cv::Mat3b reduced;
    double t = double(cv::getTickCount());
    vbs::Segmentation::reduceColor_kmeans(image, reduced, kvalue);
    t = (double(cv::getTickCount()) - t) / cv::getTickFrequency();

    printf("Color reduction using k-means took %i ms with %3i colors\n", int(t * 1000), kvalue);
    reduced.copyTo(output);
//    cv::cvtColor(reduced, reduced, COLOR_Lab2BGR);
//    cv::imshow("Test", reduced);
//    cv::waitKey(0);
}

void vbs::cvSketch::getColorchart(std::map<cv::Vec3b, int, lessVec3b>& palette, cv::Mat& output, int chartwidth, int chartheight, int area)
{
    // Print palette
    cv::Mat chart(chartheight, chartwidth, CV_8UC3);
    int coloridx = 0;
    int maxidx = 0;

    cv::Vec3b lastcolor;

    for (auto color : palette)
    {
        std::cout << "Color: " << color.first << " \t - Area: " << 100.f * float(color.second) / float(area) << "%" << std::endl;
        int max_width = chartwidth * (float(color.second) / float(area));
        maxidx += max_width;
        for(int i = 0; i <  chart.rows; i++)
        {
            for (int j = coloridx; j < maxidx; j++)
            {
                cv::Vec3b lab = color.first;
                //cv::Scalar bgr = vbs::Segmentation::ScalarLAB2BGR(color.first[0], color.first[1], color.first[2]);
                chart.at<cv::Vec3b>(i, j) = cv::Vec3b(lab[0], lab[1], lab[2]);
            }
        }
        coloridx = coloridx + max_width;

        lastcolor = color.first;
    }

    if(maxidx < chartwidth){
        for(int i = 0; i <  chart.rows; i++)
        {
            for (int j = maxidx; j < chartwidth; j++)
            {
                chart.at<cv::Vec3b>(i, j) = cv::Vec3b(lastcolor[0], lastcolor[1], lastcolor[2]);
            }
        }
    }

    chart.copyTo(output);
}

void vbs::cvSketch::getColorchart(std::vector<std::pair<cv::Vec3b, int>>& palette, cv::Mat& output, int chartwidth, int chartheight, int area)
{
    // Print palette
    cv::Mat chart(chartheight, chartwidth, CV_8UC3);
    int coloridx = 0;
    int maxidx = 0;

    cv::Vec3b lastcolor;

    for (auto color : palette)
    {
        std::cout << "Color: " << color.first << " \t - Area: " << 100.f * float(color.second) / float(area) << "%" << std::endl;
        int max_width = chartwidth * (float(color.second) / float(area));
        maxidx += max_width;
        for(int i = 0; i <  chart.rows; i++)
        {
            for (int j = coloridx; j < maxidx; j++)
            {
                cv::Vec3b lab = color.first;
                chart.at<cv::Vec3b>(i, j) = cv::Vec3b(lab[0], lab[1], lab[2]);

                if(j == maxidx)
                    chart.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 127, 127);
            }



        }

//        cv::Mat space(chart.rows,5,CV_8UC3,cv::Scalar(0,127,127));
//        cv::hconcat(chart, space, chart);


        coloridx = coloridx + max_width;

        lastcolor = color.first;
    }

    if(maxidx < chartwidth)
    {
        for(int i = 0; i <  chart.rows; i++)
        {
            for (int j = maxidx; j < chartwidth; j++)
            {
                chart.at<cv::Vec3b>(i, j) = cv::Vec3b(lastcolor[0], lastcolor[1], lastcolor[2]);
            }
        }
    }

    chart.copyTo(output);
}

void vbs::cvSketch::getDefaultColorchart(std::map<cv::Vec3b, int, lessVec3b>& palette, cv::Mat& output, int chartwidth, int chartheight)
{
    // Print palette
    int area = chartwidth * chartheight;
    cv::Mat chart(chartheight, chartwidth, CV_8UC3);
    int coloridx = 0;
    int maxidx = 0;

    cv::Vec3b lastcolor;

    for (auto color : palette)
    {
        std::cout << "Color: " << color.first << " \t - Area: " << 100.f * float(color.second) / float(area) << "%" << std::endl;
        int max_width = chartwidth  / float(palette.size());
        maxidx += max_width;
        for(int i = 0; i <  chart.rows; i++)
        {
            for (int j = coloridx; j < maxidx; j++)
            {
                cv::Vec3b bgr = color.first;
                cv::Scalar lab = vbs::Segmentation::ScalarRGB2LAB(color.first[0], color.first[1], color.first[2]);
                chart.at<cv::Vec3b>(i, j) = cv::Vec3b(lab[0], lab[1], lab[2]);
            }
        }
        coloridx = coloridx + max_width;

        lastcolor = color.first;
    }


    chart.copyTo(output);
}

void vbs::cvSketch::quantizeColors(cv::Mat& image, cv::Mat& lables, int num_labels, cv::Mat& output, std::map<cv::Vec3b, int, lessVec3b> colorpalette)
{
    cv::Mat quantizedImage;
    vbs::Segmentation::quantizedImage(lables, image, num_labels, colorpalette, quantizedImage);
    quantizedImage.copyTo(output);

}

void vbs::cvSketch::extractSuperpixels(cv::Mat& image, cv::Mat& output, cv::Mat& mask, int& num_output, int num_superpixels, int num_levels, int prior, int num_histogram_bins, bool double_step, int num_iterations)
{
    cv::Mat superpixels;
    image.copyTo(superpixels);

    int widht = superpixels.cols;
    int height = superpixels.rows;
    int channels = superpixels.channels();

    Segmentation::seeds = cv::ximgproc::createSuperpixelSEEDS(widht, height, channels, num_superpixels, num_levels, prior, num_histogram_bins, double_step);

    double t = double(cv::getTickCount());
    Segmentation::seeds->iterate(superpixels, num_iterations);
    t = (double(cv::getTickCount()) - t) / cv::getTickFrequency();
    printf("SEEDS segmentation took %i ms with %3i superpixels\n", int(t * 1000), Segmentation::seeds->getNumberOfSuperpixels());

    cv::Mat mask_labels, labels;
    Segmentation::seeds->getLabels(labels);
    Segmentation::seeds->getLabelContourMask(mask_labels, false);

    num_output = Segmentation::seeds->getNumberOfSuperpixels();
    labels.copyTo(output);
    mask_labels.copyTo(mask);
}

void vbs::cvSketch::run()
{
	if (verbose)
		std::cout << "cvSketch run ..." << std::endl;

    cv::Mat image = cv::imread(input,1);
    cv::Mat reduced;
    int max_width = 352, max_height = 240;
    ////int max_width = 720, max_height = 480;
    cv::resize(image, reduced, cv::Size(max_height, max_width));

//    cv::Mat colorSegments;
//    testColorSegmentation(reduced, colorSegments);

//    cv::Mat colorSegments, colorLabels;
//    std::map<cv::Vec3b, int, lessVec3b> palette;
//    testColorSegmentation(reduced, colorSegments, colorLabels, palette);

    //cv::Mat descriptors;
    //describeColorSegmentation(image, colorSegments, colorLabels, palette, descriptors);


	searchimagein(input);

}

bool compareMatchesByDist(const vbs::Match & a, const vbs::Match & b)
{
	return a.dist < b.dist;
}



void vbs::cvSketch::searchimagein(std::string query_path)
{
    double t;
    t = double(cv::getTickCount());

	using namespace boost::filesystem;
	const path dir(searchin);
	const recursive_directory_iterator it(dir), end;

	std::vector<std::string> files;
	for (auto& entry : boost::make_iterator_range(it, end))
	{
        boost::filesystem::path t(entry.path());
        std::string filepath = t.string();

		if ((t.filename() != ".DS_Store") && is_regular(entry))
		{
			files.push_back(filepath);
		}
	}

    path query(query_path);
    cv::Mat query_image = cv::imread(query.string(),1);
	//cv::Mat q_reduced;
	int max_width = 352, max_height = 240; //int max_width = 720, max_height = 480;
	//int area = max_width * max_height;

    std::string orientation = "";

	const int numberOfColors = 4;

    cv::Mat q_colorchart;
    std::vector<std::pair<cv::Vec3b, int>> q_colorpalette;
    process_image(query_image, max_width, max_height, numberOfColors, q_colorchart, q_colorpalette);

    std::vector<std::pair<cv::Vec3b, float>> q_colorpalette_weights(q_colorpalette.size());
    for (int i = 0; i < q_colorpalette.size(); i++)
    {
        float weight = float(q_colorpalette[i].second) / float(max_height * max_width);
        q_colorpalette_weights[i].second = weight;
        q_colorpalette_weights[i].first = q_colorpalette[i].first;
        std::cout << "Color: " << q_colorpalette_weights[i].first << " \t - Area: " << float(q_colorpalette_weights[i].second) << "%" << std::endl;
    }

    cv::destroyAllWindows();
    std::string winnameQuery = "Query Image: " + query.filename().string();
    cv::namedWindow(winnameQuery, WINDOW_NORMAL);
    show_image(q_colorchart, winnameQuery, 25, 50);


    std::vector<Match> matches;
    cv::Mat image;
    cv::Mat colorchart;
    int area = max_width * max_height;
    
    int max_files = 0;
    if(nr_input_images == -1){
        max_files = int(files.size());
    }else{
        max_files = nr_input_images;
    }
    
	for(int i = 0; i < max_files; i++)
	{
        path db(files.at(i));

        image = cv::imread(files.at(i), 1);

        std::vector<std::pair<cv::Vec3b, int>> db_colorpalette;
        process_image(image, max_width, max_height, numberOfColors, colorchart, db_colorpalette);

        std::vector<std::pair<cv::Vec3b, float> > db_colorpalette_weights(db_colorpalette.size());
        for (int i = 0; i < db_colorpalette.size(); i++)
        {
            float weight = float(db_colorpalette[i].second) / float(max_height * max_width);
            db_colorpalette_weights[i].second = weight;
            db_colorpalette_weights[i].first = db_colorpalette[i].first;

            std::cout << "Sorted Color: " << db_colorpalette_weights[i].first << " \t - Area: " << float(db_colorpalette_weights[i].second) << "%" << std::endl;

        }

//        if(query.filename().string() == db.filename().string())
//        {
//            std::cout << "DB Image is the Query image at position: " << i << std::endl;
//
//            std::cout << "DB Image Colors: " << std::endl;
//            for (auto color : sorted_colorpalette)
//            {
//                std::cout << "DB Color: " << color.first << " \t - Area: " << 100.f * float(color.second) / float(max_height * max_width) << "%" << std::endl;
//            }
//
//            std::cout << "Query Image Colors: " << std::endl;
//            for (auto color : sorted_q_colorpalette)
//            {
//                std::cout << "Query Color: " << color.first << " \t - Area: " << 100.f * float(color.second) / float(max_height * max_width) << "%" << std::endl;
//            }
//
//            std::cout << "====================================" << i << std::endl;
//        }

        //double dist = vbs::Matching::compareWithOCCD(q_colorpalette_weights, db_colorpalette_weights, area);

        double dist = vbs::Matching::compareWithEuclid(q_colorpalette_weights, db_colorpalette_weights);

        
        //Display the retrieval results
        cv::Mat result;
        colorchart.copyTo(result);

        cv::putText(result, std::to_string(dist), cvPoint(30, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 2.0, cv::Scalar(0, 0, 255), 1, CV_AA);
		Match match;
		match.path = files.at(i);
        match.dist = dist;
		match.image = result;
		matches.push_back(match);

        image.release();
	}

	std::sort(matches.begin(), matches.end(), compareMatchesByDist);

	std::vector<cv::Mat> images;

    int max_results = 0;
    if(top_kresults == -1){
        max_results = max_files;
    }else{
        max_results = top_kresults;
    }
    
	for (int i = 0; i < top_kresults; i++)
	{
		images.push_back(matches.at(i).image);
	}

    t = (double(cv::getTickCount()) - t) / cv::getTickFrequency();
    printf("Searching took %i ms %3ix%3i px resoultion; %i colos; %i files \n", int(t * 1000), max_width, max_height, numberOfColors, int(files.size()));
    printf("Searching took %i ms per file \n", int( float(t * 1000) / float(files.size())));


	cv::Mat results = makeCanvas(images, 700, 7);
    std::string winnameResults = "Retrieval Results Top: " + std::to_string(top_kresults) + " of " + std::to_string(files.size());
	cv::namedWindow(winnameResults);
    show_image(results, winnameResults, q_colorchart.cols + 25, 50);

	cv::waitKey(0);
}


void vbs::cvSketch::show_image(const cv::Mat& image, std::string winname, int x, int y)
{
    cv::Mat result;
    image.copyTo(result);
    cv::cvtColor(result, result, CV_Lab2BGR);
    cv::moveWindow(winname, x, y);
    cv::imshow(winname, result);
}

void vbs::cvSketch::process_image(const cv::Mat& image, int width, int height, int colors, cv::Mat& image_withbarchart, std::vector<std::pair<cv::Vec3b, int>>& sorted_colorpalette)
{
    cv::Mat reduced;
    std::string orientation = "";
    int bar_width = 0;
    if (image.rows > image.cols)
    {
        bar_width = height;
        orientation = "Portrait";
        cv::resize(image, reduced, cv::Size(height, width));
    }
    else
    {
        bar_width = width;
        orientation = "Landscape";
        cv::resize(image, reduced, cv::Size(width, height));
    }

    if(verbose)
        std::cout << "Image: "<< orientation <<  "..." << std::endl;

    cv::Mat q_colorSegments, q_colorLabels;
    //Convert to LAB color space
    if(verbose)
        std::cout << "Image: Convert colors to LAB ..." << std::endl;
    cv::cvtColor(reduced, reduced, cv::COLOR_BGR2Lab);

    if(verbose)
        std::cout << "Image: Reduce colors using k-means ..." << std::endl;

    cv::Mat3b reducedColorImage;
    //Cluster colors in LAB color space using k-means
    reduceColors(reduced, colors, reducedColorImage);

    std::map<cv::Vec3b, int, lessVec3b> q_colorpalette = vbs::Segmentation::getPalette(reducedColorImage);

    //std::vector<std::pair<cv::Vec3b, int>> sorted_colorpalette;
    vbs::Segmentation::sortPaletteByArea(q_colorpalette, sorted_colorpalette);

    cv::Mat colorchart;
    vbs::cvSketch::getColorchart(sorted_colorpalette, colorchart, bar_width, 50, (reduced.cols * reduced.rows));
    cv::Mat space(20,colorchart.cols,CV_8UC3,cv::Scalar(0,127,127));

    cv::vconcat(reduced, space, reduced);
    cv::vconcat(reduced, colorchart, reduced);
    reduced.copyTo(image_withbarchart);
}

cv::Mat vbs::cvSketch::makeCanvas(std::vector<cv::Mat>& vecMat, int windowHeight, int nRows)
{
	int N = int(vecMat.size());
	nRows = nRows > N ? N : nRows;
	int edgeThickness = 10;
	int imagesPerRow = ceil(double(N) / nRows);
	int resizeHeight = floor(2.0 * ((floor(double(windowHeight - edgeThickness) / nRows)) / 2.0)) - edgeThickness;
	int maxRowLength = 0;

	std::vector<int> resizeWidth;
	for (int i = 0; i < N;) {
		int thisRowLen = 0;
		for (int k = 0; k < imagesPerRow; k++) {
			double aspectRatio = double(vecMat[i].cols) / vecMat[i].rows;
			int temp = int(ceil(resizeHeight * aspectRatio));
			resizeWidth.push_back(temp);
			thisRowLen += temp;
			if (++i == N) break;
		}
		if ((thisRowLen + edgeThickness * (imagesPerRow + 1)) > maxRowLength) {
			maxRowLength = thisRowLen + edgeThickness * (imagesPerRow + 1);
		}
	}
	int windowWidth = maxRowLength;
    cv::Mat canvasImage(windowHeight, windowWidth, CV_8UC3, cv::Scalar(0, 127, 127));

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
			}
			else {
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

bool vbs::cvSketch::storeImage(std::string originalfile, std::string append, std::string extension, cv::Mat& image)
{
	boost::filesystem::path in(originalfile);
	boost::filesystem::path ext = in.filename().extension();
    size_t position = in.filename().string().find(ext.string());
	std::string store = output + DIRECTORY_SEPARATOR + in.filename().string().substr(0, position) + append + extension;

	cv::imwrite(store, image);
	return false;
}

vbs::cvSketch::~cvSketch()
{
	if(verbose)
		std::cout << "cvSketch destructor ..." << std::endl;
}

boost::program_options::variables_map vbs::cvSketch::processProgramOptions(const int argc, const char *const argv[])
{
    boost::program_options::options_description generic("Generic options");
    generic.add_options()
    ("help,h", "Print options")
	("verbose", "show additional information while processing (default false)")
	("display", "show output while processing (default false)")
	("input", boost::program_options::value<std::string>(), "the image to process")
	("output", boost::program_options::value<std::string>()->default_value("output"), "specify the output (default is ./output/input.jpg,input.png)")
	("searchin", boost::program_options::value<std::string>(), "the directory containing images")
	;

    boost::program_options::options_description config("Configuration");
    config.add_options()
    ("Param.threshold", boost::program_options::value<float>()->default_value(0.1),
     "test param for possible config files")
    ;

    boost::program_options::positional_options_description positional;
    positional.add("input", 1);

    boost::program_options::options_description cmdlineOptions;
    cmdlineOptions.add(generic).add(config);

    boost::program_options::options_description configfileOptions;
    configfileOptions.add(config);

    boost::program_options::options_description visible("Allowed options");
    visible.add(generic).add(config);

    if (argc < 2)
    {
        std::cout << "ERROR: For execution there are too few arguments!" << std::endl;
        std::cout << help(visible) << std::endl;
        exit(EXIT_SUCCESS);
    }

    boost::program_options::variables_map args;
    boost::program_options::variables_map configfileArgs;

    try
    {
        store(boost::program_options::command_line_parser(argc, argv).options(cmdlineOptions).positional(positional).run(), args);
    }
    catch (boost::program_options::error const& e)
    {
        std::cout << "ERROR: Try store program options ..." << std::endl;

        std::cout << e.what() << std::endl;
        std::cout << help(visible) << std::endl;
        exit(EXIT_FAILURE);
    }

    std::ifstream ifs;

    try
    {
        ifs.open(args["config"].as< std::string >());
        if (!ifs.is_open())
        {
            std::cout << "Configuration file  " << args["config"].as< std::string >() << " cannot be found" << std::endl;
            std::cout << help(visible) << std::endl;
            exit(EXIT_SUCCESS);
        }
    }
    catch (boost::bad_any_cast const& e)
    {
        std::cout << "ERROR: No configuraiton defined ..." << std::endl;

    }

    try
    {
        store(parse_config_file(ifs, visible), args);
    }
    catch (boost::program_options::error const& e)
    {
        std::cerr << "ERROR: Try store configfile ..." << std::endl;
        std::cerr << e.what() << std::endl;
        std::cout << help(visible) << std::endl;
        exit(EXIT_FAILURE);
    }

    if (args.count("help")) {
        std::cout << help(visible) << std::endl;
    }

    notify(args);

    return args;
}
