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

cv::Mat3b dst_crimagelab;
cv::Mat3b dst_crimagebgr;
std::string winname1 = "Color Reduction using k-means";
void on_trackbar_colorReduction_kMeans(const int kvalue, void* data)
{
	const cv::Mat src = *static_cast<cv::Mat*>(data);

	if(kvalue > 0)
	{
		double t = double(getTickCount());
		vbs::Segmentation::reduceColor_kmeans(src, dst_crimagelab, kvalue);
		t = (double(getTickCount()) - t) / getTickFrequency();
		printf("Color reduciton took %i ms with %3i colors\n", int(t * 1000), kvalue);

		cv::cvtColor(dst_crimagelab, dst_crimagebgr, COLOR_Lab2BGR);
	}else
	{
		cv::cvtColor(src, dst_crimagebgr, COLOR_Lab2BGR);
	}
	cv::imshow(winname1, dst_crimagebgr);
}

std::string winname2 = "Superpixel using SEEDS";
cv::Ptr<cv::ximgproc::SuperpixelSEEDS> seeds;
int num_iterations = 4;
int prior = 5;
bool double_step = false;
int num_superpixels = 400;
int num_levels = 4;
int num_histogram_bins = 5;
void on_trackbar_superpixel_SEEDS(const int kvalue, void* data)
{
	const cv::Mat src = *static_cast<cv::Mat*>(data);
	seeds = cv::ximgproc::createSuperpixelSEEDS(src.cols, src.rows, src.channels(), num_superpixels, num_levels, prior, num_histogram_bins, double_step);

	double t = double(getTickCount());
	seeds->iterate(src, num_iterations);
	t = (double(getTickCount()) - t) / getTickFrequency();
	printf("SEEDS segmentation took %i ms with %3i superpixels\n", int(t * 1000), seeds->getNumberOfSuperpixels());

	Mat labels, mask, result;

	seeds->getLabels(labels);
	seeds->getLabelContourMask(mask, false);
	vbs::Segmentation::meanImage(labels, dst_crimagebgr, seeds->getNumberOfSuperpixels(), result);

	result.setTo(Scalar(0, 0, 255), mask);
	cv::imshow(winname2, result);
}

void vbs::cvSketch::run()
{
	if (verbose)
		std::cout << "cvSketch run ..." << std::endl;

	double t = 0.0;

	cv::Mat image = cv::imread(input);
	cv::Mat source;
	image.copyTo(source);

	const int width = image.cols;
	const int height = image.rows;

	cv::putText(source, "Original", cvPoint(30, 30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, CV_AA);

	std::cout << "Convert colors to LAB ..." << std::endl;
	t = double(getTickCount());
	cv::cvtColor(image, image, COLOR_BGR2Lab);
	t = (double(getTickCount()) - t) / getTickFrequency();
	printf("Color conversion took %i ms with %3ix%3i resoultion \n",int(t * 1000), width, height);

	std::cout << "Reduce colors using k-means ..." << std::endl;
	int kvalue_init = 8;
	const int kMean_max = 16;
	cv::namedWindow(winname1, 1);
	const std::string trackbarname1 = "Number of Colors" + std::to_string(kMean_max) + ": ";
	cv::createTrackbar(trackbarname1, winname1, &kvalue_init, kMean_max, on_trackbar_colorReduction_kMeans, &image);
	on_trackbar_colorReduction_kMeans(kvalue_init, &image);

	std::cout << "Create superpixels using SEEDS ..." << std::endl;
	//SEED Superpixels
	num_superpixels = 400;
	prior = 5;
	num_levels = 4;
	num_iterations = 4;
	double_step = false;
	num_histogram_bins = 5;
	cv::namedWindow(winname2, 1);

	const std::string trackbarname2 = "Number of Superpixels " + std::to_string(1000) + ": ";
	const std::string trackbarname3 = "Smoothing Prior " + std::to_string(5) + ": ";
	const std::string trackbarname4 = "Number of Levels " + std::to_string(10) + ": ";
	const std::string trackbarname5 = "Iterations " + std::to_string(12) + ": ";
	const std::string trackbarname6 = "Number of Histogram Bins " + std::to_string(10) + ": ";
	cv::createTrackbar(trackbarname2, winname2, &num_superpixels, 1000, on_trackbar_superpixel_SEEDS, &image);
	cv::createTrackbar(trackbarname3, winname2, &prior, 5, on_trackbar_superpixel_SEEDS, &image);
	cv::createTrackbar(trackbarname4, winname2, &num_levels, 10, on_trackbar_superpixel_SEEDS, &image);
	cv::createTrackbar(trackbarname5, winname2, &num_iterations, 12, on_trackbar_superpixel_SEEDS, &image);
	cv::createTrackbar(trackbarname6, winname2, &num_histogram_bins, 10, on_trackbar_superpixel_SEEDS, &image);
	on_trackbar_superpixel_SEEDS(0, &image);


	int c = waitKey(0);
	while((c & 255) != 'q' && c != 'Q' && (c & 255) != 27)
	{
		if (c == 's')
		{
			std::string append = "_reduced_kmeans_rgb_k=" + std::to_string(kvalue_init);
			storeImage(input, append, ".png", dst_crimagebgr);
		}
		c = waitKey(0);
	}

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
