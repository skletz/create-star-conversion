#include "histmap.hpp"

#include <iomanip>
#include <fstream>

namespace vbs{
    
    const std::vector<std::pair<cv::Vec3b, float>> HistMapExtractor::color_palette_rgb = {
        std::make_pair(cv::Vec3b(110,73,49),1), //brown
        std::make_pair(cv::Vec3b(173,131,104),1), //light_brown
        std::make_pair(cv::Vec3b(254,39,18),1), //red
        std::make_pair(cv::Vec3b(253,83,8),1), //dark_orange
        std::make_pair(cv::Vec3b(251,153,2),1), //orange
        std::make_pair(cv::Vec3b(250,188,2),1), //light_orange
        std::make_pair(cv::Vec3b(254,254,51),1), //yellow
        std::make_pair(cv::Vec3b(208,234,43),1), //ligh_green
        std::make_pair(cv::Vec3b(102,176,50),1), //green
        std::make_pair(cv::Vec3b(0,161,160),1), //türkis
        std::make_pair(cv::Vec3b(3,145,206),1), //light_blue
        std::make_pair(cv::Vec3b(2,71,254),1), //blue
        std::make_pair(cv::Vec3b(61,1,164),1), //dark_blue
        std::make_pair(cv::Vec3b(134,1,175),1), //violett
        std::make_pair(cv::Vec3b(167,25,75),1), //purpel
        std::make_pair(cv::Vec3b(0,0,0),1), //black
        std::make_pair(cv::Vec3b(64,64,64),1), //dark_gray
        std::make_pair(cv::Vec3b(128,128,128),1), //light_gray
        std::make_pair(cv::Vec3b(255,255,255),1), //white
    };

    HistMapExtractor::HistMapExtractor(std::string _imagepath)
    {
        this->mImage_path = _imagepath;
        
    }

    HistMapExtractor::~HistMapExtractor()
    {
        
    }
    
    void HistMapExtractor::process()
    {
        std::cout << "HistMapExtractor started." << std::endl;
        
        //Filename
        std::ostringstream filename;
        filename << this->mImage_path << "_" << std::setfill('0') << std::setw(7)  << "_histmap.bin";
        
        cv::Mat image = cv::imread(this->mImage_path, cv::IMREAD_UNCHANGED);
        
        if(image.empty())
        {
            std::cerr << "Error: The image file cannot be read." << this->mImage_path << std::endl;
            return;
        }
        
        //Process image and compute the histogram map
        cv::Mat descriptor;
        compute_histmap(image, descriptor);
        
        writeToFile(filename.str(), cv::Mat());

        std::cout << "HistMapExtractor completed." << std::endl;
    }
 
	void HistMapExtractor::compute_histmap_grid(cv::Mat _src, cv::Mat& _dst, int _width, int _height, int _gsize)
	{
		cv::Mat reduced_image;
		cv::resize(_src, reduced_image, cv::Size(_width, _height));

		std::vector<std::pair<cv::Vec3b, float>> palette_lab;
		for (auto color : HistMapExtractor::color_palette_rgb)
		{
			cv::Scalar lab = HistMapExtractor::ScalarBGR2LAB(color.first[2], color.first[1], color.first[0]);
			cv::Vec3b tmp = cv::Vec3b(lab[0], lab[1], lab[2]);
			palette_lab.push_back(std::make_pair(tmp, -1.0));
		}


		int patch_size_x = (_width / float(_gsize - 1));
		int patch_size_y = (_height / float(_gsize - 1));

		int cell_size_x = (patch_size_x / float(2));
		int cell_size_y = (patch_size_y / float(2));

		int num_colors = int(palette_lab.size());
		int num_hists_x = (_width / float(patch_size_x) * 2) - 1; //overlapping sliding window cols
		int num_hists_y = (_height / float(patch_size_y) * 2) - 1; //overlapping sliding window rows
		int hist_dim = num_hists_x * num_hists_y;



		cv::Mat descriptor, patch;


    	if (_src.channels() == 4)
		{
			patch = cv::Mat::zeros(patch_size_y, patch_size_x, CV_8UC4);
		}
		else
		{
			patch = cv::Mat::zeros(patch_size_y, patch_size_x, CV_8UC3);
		}

		//each row represents one histogram
		descriptor = cv::Mat::zeros(hist_dim, num_colors, CV_32F);
		int counter = 0;
		for (int iPatchx = 0; iPatchx < _width - cell_size_x; iPatchx = iPatchx + cell_size_x)
		{
			for (int iPatchy = 0; iPatchy < _height - cell_size_y; iPatchy = iPatchy + cell_size_y)
			{
				cv::Mat subhist(1, num_colors, CV_32F);
				cv::Rect mask = cv::Rect(iPatchx, iPatchy, patch_size_x, patch_size_y);
				reduced_image(mask).copyTo(patch);
				compute_histmap_patch(patch, palette_lab, cell_size_x, cell_size_y, subhist);

				if (iPatchx == 0 && iPatchy == 0)
					descriptor = subhist;
				else
					cv::vconcat(subhist, descriptor, descriptor);

				counter++;
			}
		}

		cv::Mat norm_descriptor;
		cv::normalize(descriptor, norm_descriptor, 1, 0, cv::NORM_L2, -1);
		norm_descriptor.copyTo(_dst);

	}

	void HistMapExtractor::compute_histmap_patch(const cv::Mat& _patch, std::vector<std::pair<cv::Vec3b, float>>& _palette, int _cstep_x, int _cstep_y, cv::Mat& _hist)
	{

		int dim = int(_palette.size());

		cv::Mat cell, subhist, hist;
		hist = cv::Mat::zeros(1, dim, CV_32F);
		subhist = cv::Mat::zeros(1, dim, CV_32F);

		for (int iCellx = 0; iCellx < _patch.cols; iCellx = iCellx + _cstep_x)
		{
			for (int iCelly = 0; iCelly < _patch.rows; iCelly = iCelly + _cstep_y)
			{
				cv::Rect mask = cv::Rect(iCellx, iCelly, _cstep_x, _cstep_y);
				_patch(mask).copyTo(cell);

				compute_histmap_cell(cell, _palette, subhist);

				hist = hist + subhist;
			}

			subhist.release();
		}

		hist.copyTo(_hist);
	}

    void HistMapExtractor::compute_histmap(cv::Mat _src, cv::Mat& _dst, int _width, int _height, int _psize, int _csize, int _pstep)
    {
        cv::Mat reduced_image;
        cv::resize(_src, reduced_image, cv::Size(_width, _height));
        
		std::vector<std::pair<cv::Vec3b, float>> palette_lab;
		for (auto color : HistMapExtractor::color_palette_rgb)
		{
			cv::Scalar lab = HistMapExtractor::ScalarBGR2LAB(color.first[2], color.first[1], color.first[0]);
			cv::Vec3b tmp = cv::Vec3b(lab[0], lab[1], lab[2]);
			palette_lab.push_back(std::make_pair(tmp, -1.0));
		}

        int num_colors = int(palette_lab.size());
        int num_hists_x = (_width / float(_psize) * 2) - 1; //overlapping sliding window cols
        int num_hists_Y = (_height / float(_psize) * 2) - 1; //overlapping sliding window rows
        int hist_dim = num_hists_x * num_hists_Y;
        
        cv::Mat descriptor, patch;
        
        if(_src.channels() == 4)
        {
            patch = cv::Mat::zeros(_psize, _psize, CV_8UC4);
        }
        else
        {
            patch = cv::Mat::zeros(_psize, _psize, CV_8UC3);
        }
        
        //each row represents one histogram
        descriptor = cv::Mat::zeros(hist_dim, num_colors, CV_32F);
		int counter = 0;
        for (int iPatchx = 0; iPatchx < _width - _pstep; iPatchx = iPatchx + _pstep)
        {
            for (int iPatchy = 0; iPatchy < _height - _pstep; iPatchy = iPatchy + _pstep)
            {
                cv::Mat subhist(1, num_colors, CV_32F);
                cv::Rect mask = cv::Rect(iPatchx, iPatchy, _psize, _psize);
                reduced_image(mask).copyTo(patch);
                
                compute_histmap_patch(patch, palette_lab, _csize, subhist);
                
                if (iPatchx == 0 && iPatchy == 0)
                    descriptor = subhist;
                else
                    cv::vconcat(subhist, descriptor, descriptor);

				counter++;
            }
        }
        
        cv::Mat norm_descriptor;
        cv::normalize(descriptor, norm_descriptor, 1, 0, cv::NORM_L2, -1);
        norm_descriptor.copyTo(_dst);
        
    }

    void HistMapExtractor::compute_histmap_patch(const cv::Mat& _patch, std::vector<std::pair<cv::Vec3b, float>>& _palette, int _cstep, cv::Mat& _hist)
    {

        int dim = int(_palette.size());
        
        cv::Mat cell, subhist, hist;
        hist = cv::Mat::zeros(1, dim, CV_32F);
        subhist = cv::Mat::zeros(1, dim, CV_32F);
        
        for (int iCellx = 0; iCellx < _patch.cols; iCellx = iCellx + _cstep)
        {
            for (int iCelly = 0; iCelly < _patch.rows; iCelly = iCelly + _cstep)
            {
                cv::Rect mask = cv::Rect(iCellx, iCelly, _cstep, _cstep);
                _patch(mask).copyTo(cell);

                compute_histmap_cell(cell, _palette, subhist);
                
                hist = hist + subhist;
            }
            
            subhist.release();
        }
        
        hist.copyTo(_hist);
    }
    
    void HistMapExtractor::compute_histmap_cell(const cv::Mat& _cell, std::vector<std::pair<cv::Vec3b, float>>& _palette, cv::Mat& _hist)
    {
        int dim = int(HistMapExtractor::color_palette_rgb.size());
        
        cv::Mat hist = cv::Mat::zeros(1, dim, CV_32F);
        
        for (int iCol = 0; iCol < _cell.cols; iCol++)
        {
            for (int iRow = 0; iRow < _cell.rows; iRow++)
            {
				cv::Scalar pixel_bgra;
				if (_cell.channels() == 4)
					pixel_bgra = _cell.at<cv::Vec4b>(iRow, iCol);
				else if (_cell.channels() == 3)
					pixel_bgra = _cell.at<cv::Vec3b>(iRow, iCol);
				else
					std::cerr << "Error with number of channels: " << _cell.channels() << ", these/this are/is not supported" << std::endl;

                cv::Scalar pixel_lab = HistMapExtractor::ScalarBGR2LAB(pixel_bgra[0], pixel_bgra[1], pixel_bgra[2]);
                
                cv::Vec4b c1 = cv::Vec4b(pixel_lab[0], pixel_lab[1], pixel_lab[2], pixel_lab[3]);
   
                if(_cell.channels() == 3 || _cell.channels() == 4)
                {
                 
                    //find most similar color
                    if((_cell.channels() == 4 && pixel_bgra[3] != 0) || _cell.channels() == 3)
                    {
                        int min_idx = int(_palette.size() - 1);
                        double min_dist = std::numeric_limits<double>::max();
                        for(int iColor = 0; iColor < dim; iColor++)
                        {
                            cv::Vec3b c2 = _palette[iColor].first;
                            //cv::Scalar lab = HistMapExtractor::ScalarBGR2LAB(rgb[2], rgb[1], rgb[0]);
                            
                            //cv::Vec3b c2 = cv::Vec3b(lab[0], lab[1], lab[2]);
                            //std::cout << "Color: " << iColor << ": " << c2 << std::endl;
                            
                            double l = (c1[0] - c2[0]);
                            double a = (c1[1] - c2[1]);
                            double b = (c1[2] - c2[2]);
                            
                            double tmp_dist = (0.90 * std::pow(l, 2) + 0.20 * std::pow(a, 2) + 0.20 * std::pow(b, 2));
                            tmp_dist = std::sqrt(tmp_dist);
                            
                            if(tmp_dist < min_dist)
                            {
                                min_dist = tmp_dist;
                                min_idx = iColor;
                            }
                        }
                        
                        hist.at<float>(0, min_idx) = hist.at<float>(0, min_idx) + 1.0;
                    }
                    
                }else
                {
                    std::cerr << "Channels: " << _cell.channels() << " not supported." << std::endl;
                }

            }
        }
        hist.copyTo(_hist);
    }
    
    void HistMapExtractor::writeToFile(const std::string& _filepath, const cv::Mat& _descriptor)
    {
        std::ofstream binary;
        binary.open(_filepath.c_str(), std::ios::out | std::ios::binary);
        
        int rows = _descriptor.rows;
        int cols = _descriptor.cols;

        binary.write((const char*)&(rows), sizeof(int));
        binary.write((const char*)&(cols), sizeof(int));
        
        for (int iRows = 0; iRows < rows; iRows++) {
            for(int iCols = 0; iCols < cols; iCols++){
                
                float value = _descriptor.at<float>(iRows, iCols);
                binary.write((char*)&(value), sizeof(float));
            }
        }
        
        binary.close();
    }
    
    void HistMapExtractor::readFromFile(const std::string _filepath, std::vector<std::vector<float>>& _descriptor)
    {
		try {
			std::ifstream read;
			read.open(_filepath, std::ios::in | std::ios::binary);

			int rows = 0;
			int cols = 0;

			read.read((char*)&rows, sizeof(int));
			read.read((char*)&cols, sizeof(int));

			std::vector<std::vector<float>> descriptor;
			for (int iRows = 0; iRows < rows; iRows++)
			{
				std::vector<float> rows;
				for (int iCols = 0; iCols < cols; iCols++)
				{
					float value;
					read.read((char*)&value, sizeof(float));
					rows.push_back(value);
				}
				descriptor.push_back(rows);
			}

			read.close();
			_descriptor.assign(descriptor.begin(), descriptor.end());
		}
		catch (const std::exception &e) {
			std::cout << "Error with file: " << _filepath << std::endl;
			std::fflush(stdout);
		}
    }
    
    void HistMapExtractor::readFromFile(const std::string _filepath, cv::Mat& _descriptor)
    {
        
        try {
            std::ifstream read;
            read.open(_filepath, std::ios::in | std::ios::binary);
            
            int rows = 0;
            int cols = 0;
            
            read.read(reinterpret_cast<char*>(&rows), sizeof(int));
            read.read(reinterpret_cast<char*>(&cols), sizeof(int));
            
            cv::Mat descriptor = cv::Mat::zeros(rows, cols, CV_32F);
            for (int iRows = 0; iRows < rows; iRows++)
            {
                for (int iCols = 0; iCols < cols; iCols++)
                {
                    float value;
                    read.read(reinterpret_cast<char*>(&value), sizeof(float));
                    descriptor.at<float>(iRows, iCols) = value;
                }
            }
            
            read.close();
			descriptor.copyTo(_descriptor);

        } catch (const std::exception &e) {
            std::cout << "Error with file: " << _filepath << std::endl;
            std::fflush(stdout);
        }
    }
    
    float HistMapExtractor::calculateSimilarity(std::vector<std::vector<float>> _d1, std::vector<std::vector<float>> _d2)
    {
        float dist = std::numeric_limits<float>::max();
        if(_d1.size() != _d2.size())
        {
            std::cerr << "Error: Input descriptors have not an equal dimension." << std::endl;
            return dist;
        }
        
        int rows = int(_d1.size());
        int cols = int(_d1[0].size());
        cv::Mat d1(rows, cols, CV_32F);
        cv::Mat d2(rows, cols, CV_32F);
        
        for (int i = 0; i < rows; i++)
        {
            d1.row(i) = cv::Mat(_d1[i]).t();
            d2.row(i) = cv::Mat(_d2[i]).t();
        }
        
        dist = calculateSimilarity(d1, d2);
        return dist;
    }
    
    float HistMapExtractor::calculateSimilarity(const cv::Mat& _d1, const cv::Mat& _d2)
    {
        float dist = std::numeric_limits<float>::max();
        
        if((_d1.cols != _d2.cols) || _d1.rows != _d2.rows)
        {
            std::cerr << "Error: Input descriptors have not an equal dimension." << std::endl;
            return dist;
        }
        
        int num_rows = _d1.rows; //== _d2
        float tmp_dist = 0.0;
        for(int iRow = 0; iRow < num_rows; iRow++)
        {
            cv::Mat row = _d1.row(iRow);
            cv::Scalar sum = cv::sum(row);
            
            if(sum[0] > 0)
            {
                float q1, q2, diff;
                for(int iCol = 0; iCol < _d1.cols; iCol++)
                {
                    q1 = _d1.at<float>(iRow,iCol);
                    q2 = _d2.at<float>(iRow,iCol);
                    diff = q1 - q2;
                    tmp_dist += float(std::pow(diff, 2));
                }
            }/*else{
                std::cout << "Skip transparent Hist: " << iRow << std::endl;
            }*/
            
            
            dist += float(std::sqrt(tmp_dist));
        }
        
        dist = float(std::sqrt(tmp_dist));
        
        return dist;
    }

    cv::Scalar HistMapExtractor::ScalarLAB2BGR(uchar L, uchar A, uchar B)
    {
        cv::Mat bgr;
        cv::Mat lab(1, 1, CV_8UC3, cv::Scalar(L, A, B));
        cv::cvtColor(lab, bgr, CV_Lab2BGR);
        return cv::Scalar(bgr.data[0], bgr.data[1], bgr.data[2]);
    }
    
    cv::Scalar HistMapExtractor::ScalarBGR2LAB(uchar B, uchar G, uchar R)
    {
        cv::Mat bgr;
        cv::Mat lab(1, 1, CV_8UC3, cv::Scalar(B, G, R));
        cv::cvtColor(lab, bgr, CV_BGR2Lab);
        return cv::Scalar(bgr.data[0], bgr.data[1], bgr.data[2]);
    }
    
    void HistMapExtractor::set_imagepath(std::string _imagepath)
    {
        this->mImage_path = _imagepath;
    }
    
    std::string HistMapExtractor::get_imagepath()
    {
        return this->mImage_path;
    }
}
