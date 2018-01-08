//
// Created by bns on 12/27/17.
//

#include <opencv/cxmisc.h>
#include <iostream>
#include "Timer.hpp"


Timer::Timer() : Timer(false)
{
};

Timer::Timer(bool doStart)
{
    stop();
    if (doStart) start();
}

void Timer::start()
{
    reset();
};

void Timer::stop()
{
    startTickCount = -1;
};

double Timer::getTimeMS()
{
    if (startTickCount < 0) return -1;

    return (cv::getTickCount() - startTickCount) / cv::getTickFrequency() * 1000.0;

};

void Timer::printTime(std::string msg)
{
    std::cout << "Time - " << msg << ": " << getTimeMS() << " ms" << std::endl;
}

void Timer::reset()
{
    startTickCount  = cv::getTickCount();
};

