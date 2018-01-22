#include "matching.hpp"

bool vbs::Matching::compareMatchesByDist(const Match & a, const Match & b)
{
	return a.dist < b.dist;
}

std::vector<cv::Point> vbs::Matching::getSimpleContours(const cv::Mat & currentQuery, int points)
{
	std::vector<std::vector<cv::Point> > _contoursQuery;
	std::vector <cv::Point> contoursQuery;
	cv::Mat edges;
	Canny(currentQuery, edges, 100, 200);
	cv::findContours(edges, _contoursQuery, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

	for (size_t border = 0; border<_contoursQuery.size(); border++)
	{
		for (size_t p = 0; p<_contoursQuery[border].size(); p++)
		{
			contoursQuery.push_back(_contoursQuery[border][p]);
		}
	}
	// In case actual number of points is less than n
	int dummy = 0;
	for (int add = (int)contoursQuery.size() - 1; add < points; add++)
	{
		contoursQuery.push_back(contoursQuery[dummy++]); //adding dummy values
	}
	// Uniformly sampling/Users/skletz/Dropbox/Programming/CUPCakes/vbssketch/cvsketch/cvsketch
	std::random_shuffle(contoursQuery.begin(), contoursQuery.end());
	std::vector<cv::Point> cont;
	for (int i = 0; i < points; i++)
	{
		cont.push_back(contoursQuery[i]);
	}
	return cont;
}

void vbs::Matching::drawPoints(cv::Mat bg, std::vector<cv::Point> cont, cv::Mat & output)
{
	cv::Mat contours(bg.rows, bg.cols, CV_8UC3);
	//bg.copyTo(contours);
	int radius = 3;
	for (int i = 0; i < cont.size(); i++) {
		cv::circle(contours, cv::Point(cont[i].x, cont[i].y), radius, CV_RGB(255, 255, 255));
	}

	contours.copyTo(output);
}

double vbs::Matching::compareWithEuclid(const std::vector<std::pair<cv::Vec3b, float>>& c1, const std::vector<std::pair<cv::Vec3b, float>>& c2)
{
	double dist = 0.0;
	for(int i = 0; i < c1.size(); i++)
	{
		int minIdx;
		double min_dist = std::numeric_limits<double>::max();
		for (int j = 0; j < c2.size(); j++)
		{

			double l = (c1[i].first[0] - c2[j].first[0]);
			double a = (c1[i].first[1] - c2[j].first[1]);
			double b = (c1[i].first[2] - c2[j].first[2]);
			double tmp_dist = (0.01 * std::pow(l, 2) + 0.49 * std::pow(a, 2) + 0.49 * std::pow(b, 2));
			//double tmp_dist = (std::pow(l, 2) + std::pow(a, 2) + std::pow(b, 2));
			tmp_dist = std::sqrt(tmp_dist);


            if(tmp_dist < min_dist)
            {
                min_dist = tmp_dist;
                minIdx = j;
            }
		}

		dist += min_dist;
	}

	dist = dist / double(c1.size());
    return dist;
}

double vbs::Matching::compareWithCIEDE(const std::vector<std::pair<cv::Vec3b, float>>& c1, const std::vector<std::pair<cv::Vec3b, float>>& c2)
{
    double dist = 0.0;
    for(int i = 0; i < c1.size(); i++)
    {
        int minIdx;
        double min_dist = std::numeric_limits<double>::max();
        for (int j = 0; j < c2.size(); j++)
        {
            CIEDE2000::LAB lab1, lab2;
            lab1.l = c1[i].first[0];
            lab1.a = c1[i].first[1];
            lab1.b = c1[i].first[2];

            lab2.l = c2[j].first[0];
            lab2.a = c2[j].first[1];
            lab2.b = c2[j].first[2];

            double tmp_dist = CIEDE2000::CIEDE2000(lab1, lab2);

            if(tmp_dist < min_dist)
            {
                min_dist = tmp_dist;
                minIdx = j;
            }
        }

        dist += min_dist;
    }

    dist = dist / double(c1.size());
    return dist;
}

void vbs::Matching::print_stack(const std::vector<std::pair<cv::Vec3b, float>>& colorpalette, cv::Mat& image)
{
    cv::Mat stack(cv::Size(500, 50), CV_8UC3, cv::Scalar(0,127,127));

    int img_width = stack.cols;
    int height = stack.rows;
    int maxidx = 0, coloridx = 0;
    for(int iColor = 0; iColor < colorpalette.size(); iColor++){

        int max_width = int(img_width * (float(colorpalette[iColor].second)));
        maxidx += max_width;
        for(int i = 0; i <  height; i++)
        {
            for (int j = coloridx; j < maxidx; j++)
            {
                cv::Vec3b lab = colorpalette[iColor].first;
                stack.at<cv::Vec3b>(i, j) = cv::Vec3b(lab[0], lab[1], lab[2]);

                if(j > maxidx - 2 && j < maxidx)
                    stack.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 127, 127);
            }
        }
        coloridx = coloridx + max_width;
    }

    stack.copyTo(image);
}

void vbs::Matching::show_image(const cv::Mat& image, const std::string winname, int x, int y)
{
    cv::Mat result;
    image.copyTo(result);
    cv::cvtColor(result, result, CV_Lab2BGR);
    cv::moveWindow(winname, x, y);
    cv::imshow(winname, result);
}

void vbs::Matching::sortPaletteByArea(std::vector<std::pair<cv::Vec3b, int>> input, std::vector<std::pair<cv::Vec3b, int>>& output)
{

    std::vector<std::pair<cv::Vec3b, int>> result;
    for(auto color : input)
    {
       result.push_back(std::make_pair(color.first, color.second));

    }

    std::sort(result.begin(), result.end(), [](const std::pair<cv::Vec3b, int>& s1, const std::pair<cv::Vec3b, int>& s2)
              {
                  cv::Vec3b c1 = s1.first;
                  cv::Vec3b c2 = s2.first;

                  //return (c1[0] < c2[0]) && ((c1[1] < c2[1]) && (c1[2] < c2[2]));
                  return (s1.second > s2.second);

              });

    output.assign(result.begin(), result.end());

}

double vbs::Matching::compareWithOCCD(const std::vector<std::pair<cv::Vec3b, float>>& c1, const std::vector<std::pair<cv::Vec3b, float>>& c2, int area)
{
    //@TODO Debugging - Change in the release
    //Transform weights into integer representation
    std::vector<std::pair<cv::Vec3b, int>> c1_int(c1.size());
    std::vector<std::pair<cv::Vec3b, int>> c2_int(c2.size());
    for (int i = 0; i < c1.size(); i++)
    {
        int weight = c1[i].second * area;
        c1_int[i].second = weight;
        c1_int[i].first = c1[i].first;
    }
    for (int i = 0; i < c2.size(); i++)
    {
        int weight = c2[i].second * area;
        c2_int[i].second = weight;
        c2_int[i].first = c2[i].first;
    }


    //Print initial stack
    std::string winNameInitialQStack = "Initial Stack Query";
    std::string winNameInitialDBStack = "Initial Stack DB";
    std::string winNameDistQStack = "Distributed Stack Query";
    std::string winNameDistDBStack = "Distributed Stack DB";
    cv::Mat q_initial_stack, db_initial_stack, q_dist_stack, db_dist_stack;
    print_stack(c1, q_initial_stack);
    cv::namedWindow(winNameInitialQStack);
    show_image(q_initial_stack, winNameInitialQStack, 0, 0);

    cv::namedWindow(winNameInitialDBStack);
    print_stack(c2, db_initial_stack);
    show_image(db_initial_stack, winNameInitialDBStack, 0, q_initial_stack.rows + 50);
    //cv::waitKey(0);

//    for (int i = 0; i < c1_int.size(); i++)
//    {
//        std::cout << "Color befor sorting: " << c1_int[i].first << std::endl;
//    }
//
//    std::sort(c1_int.begin(), c1_int.end(), [](const std::pair<cv::Vec3b, int>& s1, const std::pair<cv::Vec3b, int>& s2)
//      {
//          cv::Vec3b c1 = s1.first;
//          cv::Vec3b c2 = s2.first;
//          return ((c1[1] > c2[1]) && (c1[2] > c2[2]));
//      });
//
//    for (int i = 0; i < c1_int.size(); i++)
//    {
//        std::cout << "Color after sorting: " << c1_int[i].first << std::endl;
//    }
//
//    std::sort(c2_int.begin(), c2_int.end(), [](const std::pair<cv::Vec3b, int>& s1, const std::pair<cv::Vec3b, int>& s2)
//      {
//          cv::Vec3b c1 = s1.first;
//          cv::Vec3b c2 = s2.first;
//          return ((c1[1] > c2[1]) && (c1[2] > c2[2]));
//      });


    std::vector<std::pair<cv::Vec3b, int>> q_stack;
    std::vector<std::pair<cv::Vec3b, int>> db_stack;
    std::vector<std::pair<cv::Vec3b, int>> cp_c1(c1_int);

    int iQColor = 0;
    bool colors_empty = false;

    while(!colors_empty)
    {
        sortPaletteByArea(cp_c1, cp_c1);
        sortPaletteByArea(c2_int, c2_int);

        std::pair<cv::Vec3b, int> qc = cp_c1[iQColor];

        int minIdx = 0; //idx of the color with minimal distance
        double min_dist = std::numeric_limits<double>::max();

        //number of colors
        int qp = cp_c1[iQColor].second;

        //search the minimum

        for (int iDBColor = 0; iDBColor < c2_int.size(); iDBColor++)
        {
            std::pair<cv::Vec3b, int> dp = c2_int[iDBColor];

            double l = (qc.first[0] - dp.first[0]);
            double a = (qc.first[1] - dp.first[1]);
            double b = (qc.first[2] - dp.first[2]);
            double tmp_dist = (0.01 * std::pow(l, 2) + 0.49 * std::pow(a, 2) + 0.49 * std::pow(b, 2));
            tmp_dist = std::sqrt(tmp_dist);

            if(c2_int[iDBColor].second != 0 && tmp_dist < min_dist)
            {
                min_dist = tmp_dist;
                minIdx = iDBColor;
            }
        }

        //the minimum to the query
        int dbp = c2_int[minIdx].second;
        int diff, r = 0;
        std::pair<cv::Vec3b, int> rest;

        //either one of the both is zero, nothing is to distribute
        if(cp_c1[minIdx].second != 0 || c2_int[iQColor].second != 0)
        {
            std::cout << "Color Q: " << cp_c1[iQColor].first << "; " << cp_c1[iQColor].second << std::endl;
            std::cout << "Color DB: " << c2_int[minIdx].first << "; " << c2_int[minIdx].second << std::endl;

            //50 > 30
            if(qp > dbp)
            {
                diff = qp - dbp; //20
                r = qp - diff; //30
                cp_c1[iQColor].second = diff; // 20
                db_stack.push_back(c2_int[minIdx]);
                c2_int[minIdx].second = 0;

                rest = std::make_pair(cp_c1[iQColor].first, r);
                q_stack.push_back(rest);

            }
            else if(qp < dbp)
            {
                diff = dbp - qp;
                r = dbp - diff;
                c2_int[minIdx].second = diff;
                q_stack.push_back(cp_c1[iQColor]);
                cp_c1[iQColor].second = 0;

                rest = std::make_pair(c2_int[minIdx].first, r);
                db_stack.push_back(rest);

            }else
            {
                q_stack.push_back(cp_c1[iQColor]);
                db_stack.push_back(c2_int[minIdx]);
                cp_c1[minIdx].second = 0;
                c2_int[iQColor].second = 0;
            }
        }

        int restColors = 0;
        for(auto c : cp_c1){
            restColors += c.second;
        }

        if(restColors == 0)
            colors_empty = true;
        else
        {
            if(iQColor == (cp_c1.size() - 1))
                iQColor = 0;
            else
                iQColor++;
        }

    };

//    sortPaletteByArea(q_stack, q_stack);
//    sortPaletteByArea(db_stack, db_stack);

    std::vector<std::pair<cv::Vec3b, float>> c1_float(q_stack.size());
    std::vector<std::pair<cv::Vec3b, float>> c2_float(db_stack.size());

    for (int i = 0; i < q_stack.size(); i++)
    {
        float weight = float(q_stack[i].second) / float(area);
        c1_float[i].first = q_stack[i].first;
        c1_float[i].second = weight;
        std::cout << "Q_Stack Color: " << c1_float[i].first << " \t - Area: " << float(c1_float[i].second) << "%" << std::endl;
    }

    for (int i = 0; i < db_stack.size(); i++)
    {
        float weight = float(db_stack[i].second) / float(area);
        c2_float[i].first = db_stack[i].first;
        c2_float[i].second = weight;
        std::cout << "DB_Stack Color: " << c2_float[i].first << " \t - Area: " << float(c2_float[i].second) << "%" << std::endl;
    }

    print_stack(c1_float, q_dist_stack);
    cv::namedWindow(winNameDistQStack);
    show_image(q_dist_stack, winNameDistQStack, q_initial_stack.cols, 0);

    print_stack(c2_float, db_dist_stack);
    cv::namedWindow(winNameDistDBStack);
    show_image(db_dist_stack, winNameDistDBStack, q_initial_stack.cols, q_dist_stack.rows + 50);
    //cv::waitKey(0);

    double dist = 0.0;
    int nrOfDists = int(c1_float.size());

    for(int i = 0; i < nrOfDists; i++)
    {
        std::pair<cv::Vec3b, float> qc = c1_float[i];

        int per1 = (qc.second) * 100;

        std::pair<cv::Vec3b, float> dbc = c2_float[i];

        std::cout << "QC: " << qc.first << " DB: " << dbc.first << std::endl;

        double l = (dbc.first[0] - qc.first[0]) / 255.0;
        double a = (dbc.first[1] - qc.first[1]) / 255.0;
        double b = (dbc.first[2] - qc.first[2]) / 255.0;
        double tmp_dist = (0.01 * std::pow(l, 2) + 0.49 * std::pow(a, 2) + 0.49 * std::pow(b, 2));
        tmp_dist = std::sqrt(tmp_dist);

        std::cout << "Dist: " << tmp_dist << std::endl;

        int per2 = (dbc.second) * 100;
        if (per1 != per2)
        {
            std::cout << "Area is not the same" << std::endl;
        }
        dist += (tmp_dist * per1);
        //dist = dist / double(sorted_colorpalette.size());
    }

    dist = dist / double(nrOfDists);
    return dist;
}

double vbs::Matching::compareWithMOCCD(
	const std::vector<std::tuple<cv::Vec3b, float, std::vector<cv::Point>, cv::Rect>>& c1,
	std::vector<std::tuple<cv::Vec3b, float, std::vector<cv::Point>, cv::Rect>>& c2, int area)
{

    return 0.0;
}
