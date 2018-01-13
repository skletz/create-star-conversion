#include "cvsketch.hpp"
#include "opencv2/opencv.hpp"

#include "../libs/imageSegmentation/imageSegmentation.hpp"
#include <opencv2/ximgproc.hpp>

#include "../libs/seeds-revised/lib/SeedsRevised.h"
#include "../libs/seeds-revised/lib/Tools.h"
#include "segmentation.hpp"

vbs::cvSketch::cvSketch()
{
    
#if DEBUG
    std::cout << "cvSketch contstructor ..." << std::endl;
#endif
    
}

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

	if (_args.find("verbose") != _args.end()) {
		verbose = true;
	}

	if (_args.find("display") != _args.end()) {
		display = true;
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

cv::Mat img;
cv::Mat3b dst;
const int cluster_slider_max = 16;
int cluster_slider;
int K;
std::string m1 = "Color Reduction using k-means";
void on_trackbar_kMeansCluster(int, void*)
{
	if(K>0)
	{
		vbs::Segmentation::reduceColor_kmeans(img, dst, K);
		cv::imshow(m1, dst);
	}else
	{
		cv::imshow(m1, img);
	}

}

void vbs::cvSketch::run()
{
	if (verbose)
		std::cout << "cvSketch run ..." << std::endl;

    cv::Mat image = cv::imread(input);
	std::cout << "Convert colors to LAB ..." << std::endl;
	cv::Mat original, convertedLAB;
	original = image;
	cv::cvtColor(image, convertedLAB, COLOR_BGR2Lab);
	img = image;
	std::vector<cv::Mat> outputcanvas;
	cv::putText(original, "Original", cvPoint(30, 30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, CV_AA);
	outputcanvas.push_back(original);

	if(verbose)
		std::cout << "Reduce colors using k-means ..." << std::endl;

	K = 5;
	
	cv::namedWindow(m1, 1);
	//char TrackbarName[100];
	//std::sprintf(TrackbarName, "C x %d", cluster_slider_max);
	std::string TrackbarName = "Cx" + std::to_string(cluster_slider_max);
	cv::createTrackbar(TrackbarName, m1, &K, cluster_slider_max, on_trackbar_kMeansCluster);
	on_trackbar_kMeansCluster(cluster_slider, 0);

	//cv::Mat3b reduced;
	//vbs::Segmentation::reduceColor_kmeans(image, reduced, K);
	
	//if(display)
	//	cv::imshow("Reduced", reduced);

	//std::string append = "_reduced_kmeans_k=" + std::to_string(K);
	//storeImage(input, append, ".png", reduced);

	//cv::Ptr<cv::ximgproc::SuperpixelSEEDS> seeds;
	//int num_iterations = 4;
	//int prior = 5;
	//bool double_step = false;
	//int num_superpixels = 400;
	//int num_levels = 4;
	//int num_histogram_bins = 5;

	//int width = image.size().width;
	//int height = image.size().height;

	//seeds = cv::ximgproc::createSuperpixelSEEDS(width, height, image.channels(), num_superpixels, num_levels, prior, num_histogram_bins, double_step);
	//

	//Mat result, converted;
	//cvtColor(image, converted, COLOR_BGR2Lab);
	//double t = (double)getTickCount();
	//seeds->iterate(converted, num_iterations);
	//result = image;
	//t = ((double)getTickCount() - t) / getTickFrequency();
	//printf("SEEDS segmentation took %i ms with %3i superpixels\n",(int)(t * 1000), seeds->getNumberOfSuperpixels());

	//
	//Mat labels, mask;
	//seeds->getLabels(labels);
	//
	//seeds->getLabelContourMask(mask, false);
	//result.setTo(Scalar(0, 0, 255), mask);
	//cv::imshow("Result", result);
	////cv::imshow("Mask", mask);

	//cv::Mat mImg = meanImage(labels, image, seeds->getNumberOfSuperpixels());
	//cv::imshow("Mean Image", mImg);
	cv::waitKey(0);
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
	int position = in.filename().string().find(ext.string());
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
