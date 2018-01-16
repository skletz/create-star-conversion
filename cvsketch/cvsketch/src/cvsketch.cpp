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

std::map<cv::Vec3b, int, vbs::lessVec3b> colorpalette;

struct SettingsKmeansCluster
{
    std::string winname;
    int kvalue;
    int kvalue_max;
    cv::Mat image;
    bool initDone;
    cv::Mat reducedImage;
    
    //update additional windows
    std::string winnameColorchart;
    std::string winnameQuantizedColors;
    cv::Mat labels;
    int num_labels;
};

struct SettingsSuperpixelSEEDS
{
    std::string winname;
    int num_superpixels;
    int num_superpixels_max;
    int prior;
    int prior_max;
    int num_levels;
    int num_levels_max;
    int num_iterations;
    int num_iterations_max;
    int double_step;
    int num_histogram_bins;
    int num_histogram_bins_max;
    cv::Mat image;
    bool initDone;
    cv::Mat superpixelImage;
    cv::Mat mask;
    
    //update additional windows
    std::string winnameQuantizedColors;
};

void on_trackbar_colorReduction_kMeans(int, void* data)
{
    SettingsKmeansCluster settings = *static_cast<SettingsKmeansCluster*>(data);
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
        colorpalette = vbs::Segmentation::getPalette(dst);
        
        cv::Mat colorchart;
        vbs::cvSketch::getColorchart(dst, colorpalette, colorchart, src.cols, 50);
        cv::imshow(settings.winnameColorchart, colorchart);
        dst.copyTo(settings.reducedImage);
        
        //Update also quantized color image
        cv::Mat quantizedImage;
        vbs::cvSketch::quantizeColors(src, settings.labels, settings.num_labels, quantizedImage, colorpalette);
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

cv::Ptr<cv::ximgproc::SuperpixelSEEDS> seeds;

void on_trackbar_superpixel_SEEDS(const int, void* data)
{
//	const cv::Mat src = *static_cast<cv::Mat*>(data);
    SettingsSuperpixelSEEDS settings = *static_cast<SettingsSuperpixelSEEDS*>(data);
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
        vbs::cvSketch::quantizeColors(src, labels, num_superpixel_found, quantizedImage, colorpalette);
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

void vbs::cvSketch::testColorSegmentation(cv::Mat& image)
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
    
    std::cout << "Create barchart of k dominant colors ..." << std::endl;
    cv::Mat colorchart;
    getColorchart(reducedColorImage, colorpalette, colorchart, input.cols, 50);
    
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
    quantizeColors(reducedColorImage, labels, num_superpixels_found, quantizedColorImage, colorpalette);
    
    if(display){
        int pad_top = 250;
        int pad_left = 50;
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
        cv::imshow(winnameChart, colorchart);
        
        colorpalette = colorpalette;
        
        //TESTBED for k-means clustering
        SettingsKmeansCluster* set_kmeans = new SettingsKmeansCluster();
        set_kmeans->winname = "Reduced Colors";
        set_kmeans->kvalue = initKValue;
        input.copyTo(set_kmeans->image);
        set_kmeans->kvalue_max = 16;
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
    }
}

//Input - Lab color space
void vbs::cvSketch::reduceColors(cv::Mat& image, int kvalue, cv::Mat& output)
{
    cv::Mat3b reduced;
    double t = double(cv::getTickCount());
    vbs::Segmentation::reduceColor_kmeans(image, reduced, kvalue);
    t = (double(cv::getTickCount()) - t) / cv::getTickFrequency();
    printf("Color reduction using k-means took %i ms with %3i colors\n", int(t * 1000), kvalue);
    reduced.copyTo(output);
}

void vbs::cvSketch::getColorchart(cv::Mat& image, std::map<cv::Vec3b, int, lessVec3b>& palette, cv::Mat& output, int chartwidth, int chartheight)
{
    // Print palette
    int area = image.rows * image.cols;
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
                cv::Scalar bgr = vbs::Segmentation::ScalarLAB2BGR(color.first[0], color.first[1], color.first[2]);
                chart.at<cv::Vec3b>(i, j) = cv::Vec3b(bgr[0], bgr[1], bgr[2]);
            }
        }
        coloridx = coloridx + max_width;
        
        lastcolor = color.first;
    }
    
    if(maxidx < image.cols){
        for(int i = 0; i <  chart.rows; i++)
        {
            for (int j = maxidx; j < image.cols; j++)
            {
                cv::Scalar bgr = vbs::Segmentation::ScalarLAB2BGR(lastcolor[0], lastcolor[1], lastcolor[2]);
                chart.at<cv::Vec3b>(i, j) = cv::Vec3b(bgr[0], bgr[1], bgr[2]);
            }
        }
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
    
    seeds = cv::ximgproc::createSuperpixelSEEDS(widht, height, channels, num_superpixels, num_levels, prior, num_histogram_bins, double_step);
    
    double t = double(cv::getTickCount());
    seeds->iterate(superpixels, num_iterations);
    t = (double(cv::getTickCount()) - t) / cv::getTickFrequency();
    printf("SEEDS segmentation took %i ms with %3i superpixels\n", int(t * 1000), seeds->getNumberOfSuperpixels());
    
    cv::Mat mask_labels, labels;
    seeds->getLabels(labels);
    seeds->getLabelContourMask(mask_labels, false);
    
    num_output = seeds->getNumberOfSuperpixels();
    labels.copyTo(output);
    mask_labels.copyTo(mask);
}

void vbs::cvSketch::run()
{
	if (verbose)
		std::cout << "cvSketch run ..." << std::endl;

    cv::Mat image = cv::imread(input);
    cv::Mat reduced;
    int max_width = 352, max_height = 240;
    //int max_width = 720, max_height = 480;
    cv::resize(image, reduced, cv::Size(max_height, max_width));
    testColorSegmentation(reduced);

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
    cv::Mat canvasImage(windowHeight, windowWidth, CV_8UC3, cv::Scalar(0, 0, 0));

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



