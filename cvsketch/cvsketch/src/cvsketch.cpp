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


vbs::cvSketch::cvSketch(bool _verbose, bool _display, int _max_width, int _max_height)
: set_kmeans(nullptr), set_seeds(nullptr)
{
	this->verbose = _verbose;
	if (verbose)
		std::cout << "Create cvSketch ..." << std::endl;

	this->display = _display;
	this->max_width = _max_width;
	this->max_height = _max_height;
    
    
    this->segmentation = new vbs::Segmentation();
    this->set_kmeans = new SettingsKmeansCluster();
    this->set_seeds = new SettingsSuperpixelSEEDS();
    this->set_exchange = new KmeansClusterSEEDSExchange();
    

    if (verbose)
        std::cout << "cvSketch created ..." << std::endl;
}

//cv::Ptr<cv::ximgproc::SuperpixelSEEDS> vbs::Segmentation::seeds;
//std::map<cv::Vec3b, int, vbs::lessVec3b> vbs::Segmentation::query_colorpalette;
//std::vector<std::pair<cv::Vec3b, int>> vbs::Segmentation::sorted_query_colorpalette;

std::string vbs::cvSketch::help(const boost::program_options::options_description& _desc)
{
    std::stringstream help;

    help << "============== Help ==============" << std::endl;
    help << "INFO: This program tests sketch-based image retrival methods ..." << std::endl;
    help << "INFO: Call ./cvsketch_demo --input [path]" << std::endl;
    help << "============== Help ==============" << std::endl;
    help << _desc << std::endl;
    return help.str();
}

std::string vbs::cvSketch::get_info()
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
		std::cout << "Initialize cvSketch parameter ..." << std::endl;

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
		in_dataset = _args["searchin"].as<std::string>();
		std::cout << "SEARCH IN: " << in_dataset << std::endl;
	}

	in_query = _args["input"].as<std::string>();
	output = _args["output"].as<std::string>();

	if (!boost::filesystem::is_directory(output)) {
		boost::filesystem::create_directory(output);
	}

    std::cout << "INPUT: " << in_query << std::endl;
    std::cout << "OUTPUT: " << output << std::endl;

    
    this->set_kmeans->kvalue = 4;
    this->set_kmeans->kvalue_max = 10;

    this->set_seeds->num_superpixels = 400;
    this->set_seeds->num_superpixels_max = 1000;
    this->set_seeds->prior = 5;
    this->set_seeds->prior_max = 10;
    this->set_seeds->num_levels = 5;
    this->set_seeds->num_levels_max = 10;
    this->set_seeds->num_iterations = 15;
    this->set_seeds->num_iterations_max = 25;
    this->set_seeds->double_step = int(true);
    this->set_seeds->num_histogram_bins = 2;
    this->set_seeds->num_histogram_bins_max = 10;
    
    if(verbose)
        std::cout << "cvSketch initialized ..." << std::endl;
    
    return true;
}


void vbs::cvSketch::run()
{
	if (verbose)
		std::cout << "Run cvSketch ..." << std::endl;

	cv::Mat image = cv::imread(in_query, IMREAD_COLOR);
	cv::Mat reduced;
	cv::resize(image, reduced, cv::Size(max_height, max_width));

	//    cv::Mat colorSegments;
	//    testColorSegmentation(reduced, colorSegments);

	cv::Mat colorSegments, colorLabels;
	std::map<cv::Vec3b, int, lessVec3b> palette;
	testColorSegmentation(reduced, colorSegments, colorLabels, palette);

	//cv::Mat descriptors;
	//describeColorSegmentation(image, colorSegments, colorLabels, palette, descriptors);


	search_image(in_query, in_dataset);
    
    if (verbose)
        std::cout << "cvSketch finished ..." << std::endl;
}

void vbs::cvSketch::on_trackbar_colorReduction_kMeans(int, void* _object)
{
    //global callback
    //vbs::SettingsKmeansCluster settings = *static_cast<vbs::SettingsKmeansCluster*>(_object);
    
    vbs::cvSketch* sketch = (vbs::cvSketch*)(_object);
    vbs::SettingsKmeansCluster* settings = sketch->set_kmeans;
    vbs::KmeansClusterSEEDSExchange* inout = sketch->set_exchange;
    
    std::string winname = settings->winname;
    cv::Mat src, dst;
    inout->image.copyTo(src);
    int k = settings->kvalue;

    if(settings->kvalue > 0)
    {
        sketch->reduce_colors(src, k, dst);
    }
    else{
        src.copyTo(dst);
    }

    //Update also color chart
    //get color-palette of the image
    std::vector<std::pair<cv::Vec3b, float>> colorpalette;
    sketch->segmentation->get_colorpalette(dst, colorpalette);

    inout->colors = colorpalette;
    
    cv::Mat colorchart;
    sketch->get_colorchart(colorpalette, colorchart, src.cols, 50, (dst.cols * dst.rows));


    dst.copyTo(inout->reduced_image);

    //Update also quantized color image
    if(inout->num_labels != 0){
        cv::Mat quantized_image;
        sketch->quantize_colors(src, inout->labels, inout->num_labels, quantized_image, colorpalette);
        sketch->show_image(quantized_image, inout->winnameQuantizedColors);
    }

    cv::Mat dst_show;
    dst.copyTo(dst_show);
    std::stringstream text;
    if(settings->kvalue != 0)
    {
        text << "Colors: " << settings->kvalue;
    }else
    {
        text << "Colors: " << "All";
    }

    //Dispaly
    sketch->set_label(dst_show, text.str(), cvPoint(20, 20));
    sketch->show_image(dst_show, winname);
    sketch->show_image(colorchart, inout->winnameColorchart);

}

void vbs::cvSketch::on_trackbar_superpixel_SEEDS(const int, void* _object)
{
    //pass mat object
    //const cv::Mat src = *static_cast<cv::Mat*>(data);
    //Global callback
    //vbs::SettingsSuperpixelSEEDS settings = *static_cast<vbs::SettingsSuperpixelSEEDS*>(data);
    
    vbs::cvSketch* sketch = (vbs::cvSketch*)(_object);
    vbs::SettingsSuperpixelSEEDS* settings = sketch->set_seeds;
    vbs::KmeansClusterSEEDSExchange* inout = sketch->set_exchange;
    
    std::string winname = settings->winname;

    cv::Mat src, labels, dst, mask;
    inout->image.copyTo(src);
    
    int num_superpixel_found;
    int num_superpixels = settings->num_superpixels;
    int prior = settings->prior;
    int num_levels = settings->num_levels;
    bool double_step = bool(settings->double_step);
    int num_iterations = settings->num_iterations;
    int num_histogram_bins = settings->num_histogram_bins;

    sketch->extract_superpixels(src, labels, mask, num_superpixel_found, num_superpixels, num_levels, prior, num_histogram_bins, double_step, num_iterations);
    src.copyTo(dst);
    dst.setTo(cv::Scalar(0,0,255), mask);

    //Update also quantized color image
    cv::Mat quantized_image;
    
    std::vector<std::pair<cv::Vec3b, float>> colorpalette;
    colorpalette = inout->colors;
    
    sketch->quantize_colors(src, labels, num_superpixel_found, quantized_image, colorpalette);

    cv::Mat dst_show;
    dst.copyTo(dst_show);
    std::stringstream text;
    text << "Superpixels: " << num_superpixels << ", ";
    text << "Prior: " << prior;
    sketch->set_label(dst_show, text.str(), cvPoint(30, 30));
    
    text.str("");
    text << "Levels: " << num_levels  << ", ";
    text << "Double Step: " << ((double_step = 1) ? "true" : "false");
    sketch->set_label(dst_show, text.str(), cvPoint(30, 45));
    
    text.str("");
    text << "Iterations: " << num_iterations  << ", ";
    text << "Hist Bins: " << num_histogram_bins;
    sketch->set_label(dst_show, text.str(), cvPoint(30, 60));
    
    sketch->show_image(dst_show, winname);
    sketch->show_image(quantized_image, inout->winnameQuantizedColors);
    
    dst.copyTo(inout->superpixel_image);
    mask.copyTo(inout->mask);
    labels.copyTo(inout->labels);
    inout->num_labels = num_superpixel_found;
}



void vbs::cvSketch::testColorSegmentation(cv::Mat& image, cv::Mat& colorSegments, cv::Mat& colorLabels, std::map<cv::Vec3b, int, lessVec3b>& palette)
{

    if(verbose)
        std::cout << "Test cvSketch color segmentation ..." << std::endl;

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

    
    if(display)
    {
        //int win_h = input.rows;
        int win_w = input.cols;

        std::string winnameOrig = "Original";
        std::string winnameChart = "Color Chart";
        std::string winnameQuantizedColors = "Quantized Color Image";
        this->set_kmeans->winname = "Reduced Colors";
        this->set_seeds->winname = "Superpixels";
        
        cv::namedWindow(winnameOrig);
        cv::moveWindow(winnameOrig, pad_left, pad_top);
        cv::imshow(winnameOrig, image);

        cv::namedWindow(winnameChart);
        cv::moveWindow(winnameChart, pad_left + win_w * 4, pad_top);

        cv::namedWindow(winnameQuantizedColors, 1);
        cv::moveWindow(winnameQuantizedColors, pad_left + win_w * 3, pad_top);
        
        cv::namedWindow(this->set_kmeans->winname, 1);
        cv::moveWindow(this->set_kmeans->winname, pad_left + win_w, pad_top);
        
        cv::namedWindow(this->set_seeds->winname, 1);
        cv::moveWindow(this->set_seeds->winname, pad_left + win_w * 2, pad_top);
        
        input.copyTo(this->set_exchange->image);//
        this->set_exchange->winnameColorchart = winnameChart;
        this->set_exchange->winnameQuantizedColors = winnameQuantizedColors;

        cv::createTrackbar("Colors", this->set_kmeans->winname, &this->set_kmeans->kvalue, this->set_kmeans->kvalue_max, &vbs::cvSketch::on_trackbar_colorReduction_kMeans, this);
        on_trackbar_colorReduction_kMeans(0, this);

        cv::createTrackbar("Superpixels", set_seeds->winname, &set_seeds->num_superpixels, set_seeds->num_superpixels_max, &vbs::cvSketch::on_trackbar_superpixel_SEEDS, this);
        cv::createTrackbar("Prior", set_seeds->winname, &set_seeds->prior, set_seeds->prior_max, &vbs::cvSketch::on_trackbar_superpixel_SEEDS, this);
        cv::createTrackbar("Levels", set_seeds->winname, &set_seeds->num_levels, set_seeds->num_levels_max, &vbs::cvSketch::on_trackbar_superpixel_SEEDS, this);
        cv::createTrackbar("Double Step", set_seeds->winname, &set_seeds->double_step, 1, &vbs::cvSketch::on_trackbar_superpixel_SEEDS, this);
        cv::createTrackbar("Hist Bins", set_seeds->winname, &set_seeds->num_histogram_bins, set_seeds->num_histogram_bins_max, &vbs::cvSketch::on_trackbar_superpixel_SEEDS, this);
        cv::createTrackbar("Interations", set_seeds->winname, &set_seeds->num_iterations, set_seeds->num_iterations_max, &vbs::cvSketch::on_trackbar_superpixel_SEEDS, this);
        on_trackbar_superpixel_SEEDS(0, this);

        int c = cv::waitKey(0);
        while((c & 255) != 'q' && c != 'Q' && (c & 255) != 27)
        {
            if (c == 's')
            {
                show_image(this->set_exchange->reduced_image, "Saved ... ");
                cv::waitKey(0);
                cv::destroyWindow("Saved ... ");
            }
            c = cv::waitKey(0);
        }

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
void vbs::cvSketch::reduce_colors(const cv::Mat& image, int kvalue, cv::Mat& output)
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

void vbs::cvSketch::get_colorchart(std::vector<std::pair<cv::Vec3b, float>>& colors, cv::Mat& output, int chartwidth, int chartheight, int area)
{
    // Print palette
    cv::Mat chart(chartheight, chartwidth, CV_8UC3);
    int coloridx = 0;
    int maxidx = 0;
    
    cv::Vec3b lastcolor;
    
    for (auto color : colors)
    {
        float scope = 0.0;
        if(area != -1) //weighted values
            scope = float(color.second) / float(area);
        else
            scope = float(color.second);
            
        std::cout << "Color: " << color.first << " \t - Area: " << 100.f * scope << "%" << std::endl;
        int max_width = chartwidth * scope;
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
        
        coloridx = coloridx + max_width;
        
        lastcolor = color.first;
    }
    
    //fill up barchart
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

void vbs::cvSketch::quantize_colors(const cv::Mat& image, cv::Mat& lables, int num_labels, cv::Mat& output, std::vector<std::pair<cv::Vec3b, float>>& colorpalette)
{
    cv::Mat quantized_image;
    this->segmentation->quantize_image(lables, image, num_labels, colorpalette, quantized_image);
    quantized_image.copyTo(output);

}

void vbs::cvSketch::extract_superpixels(cv::Mat& image, cv::Mat& output, cv::Mat& mask, int& num_output, int num_superpixels, int num_levels, int prior, int num_histogram_bins, bool double_step, int num_iterations)
{
    cv::Mat superpixels;
    image.copyTo(superpixels);

    int widht = superpixels.cols;
    int height = superpixels.rows;
    int channels = superpixels.channels();

    this->segmentation->seeds = cv::ximgproc::createSuperpixelSEEDS(widht, height, channels, num_superpixels, num_levels, prior, num_histogram_bins, double_step);

    double t = double(cv::getTickCount());
    this->segmentation->seeds->iterate(superpixels, num_iterations);
    t = (double(cv::getTickCount()) - t) / cv::getTickFrequency();
    printf("SEEDS segmentation took %i ms with %3i superpixels\n", int(t * 1000), this->segmentation->seeds->getNumberOfSuperpixels());

    cv::Mat mask_labels, labels;
    this->segmentation->seeds->getLabels(labels);
    this->segmentation->seeds->getLabelContourMask(mask_labels, false);

    num_output = this->segmentation->seeds->getNumberOfSuperpixels();
    labels.copyTo(output);
    mask_labels.copyTo(mask);
}



bool compareMatchesByDist(const vbs::Match & a, const vbs::Match & b)
{
	return a.dist < b.dist;
}



void vbs::cvSketch::search_image(std::string query_path, std::string dataset_path)
{
    double t;
    t = double(cv::getTickCount());

	using namespace boost::filesystem;
	const path dir(dataset_path);
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

        double dist = vbs::Matching::compareWithOCCD(q_colorpalette_weights, db_colorpalette_weights, area);

        //double dist = vbs::Matching::compareWithEuclid(q_colorpalette_weights, db_colorpalette_weights);

        
        //Display the retrieval results
        cv::Mat result;
        colorchart.copyTo(result);
  
        set_label(result, std::to_string(dist),cvPoint(10, 40), 1.5);
        
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


	cv::Mat results = make_canvas(images, 700, 7);
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
    
    if(x != -1 && y != -1)
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
    reduce_colors(reduced, colors, reducedColorImage);

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

boost::program_options::variables_map vbs::cvSketch::process_program_options(const int argc, const char *const argv[])
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

vbs::cvSketch::~cvSketch()
{
    if(verbose)
        std::cout << "cvSketch destructor ..." << std::endl;
    
    delete set_kmeans;
    delete set_seeds;
    delete set_exchange;
}


cv::Mat vbs::cvSketch::make_canvas(std::vector<cv::Mat>& vecMat, int windowHeight, int nRows)
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

void vbs::cvSketch::set_label(cv::Mat& _im, const std::string _label, const cv::Point& _point, float _scale)
{
    int fontface = cv::FONT_HERSHEY_DUPLEX;
    double scale = _scale;
    int thickness = 1;
    int baseline = 0;
    
    cv::Size text = cv::getTextSize(_label, fontface, scale, thickness, &baseline);
    cv::rectangle(_im, _point + cv::Point(0, baseline), _point + cv::Point(text.width, -text.height), cv::Scalar(0,127,127), CV_FILLED);
    cv::putText(_im, _label, _point, fontface, scale, CV_RGB(200, 200, 250), thickness, 8);
}

bool vbs::cvSketch::store_image(std::string originalfile, std::string append, std::string extension, cv::Mat& image)
{
    boost::filesystem::path in(originalfile);
    boost::filesystem::path ext = in.filename().extension();
    size_t position = in.filename().string().find(ext.string());
    std::string store = output + DIRECTORY_SEPARATOR + in.filename().string().substr(0, position) + append + extension;
    
    cv::imwrite(store, image);
    return false;
}
