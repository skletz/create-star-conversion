//
// Created by bns on 12/27/17.
//

#ifndef SBSS_TIMER_HPP
#define SBSS_TIMER_HPP

class Timer
{

private:
    int64 startTickCount;

public:
    Timer();

    Timer(bool doStart);

    void start();

    void stop();

    double getTimeMS();

    void printTime(std::string msg, bool reset = false);

    void reset();

};

#endif //SBSS_TIMER_HPP
