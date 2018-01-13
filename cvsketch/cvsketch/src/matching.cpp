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
	// Uniformly sampling
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
