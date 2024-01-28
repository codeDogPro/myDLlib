#pragma once

#include <iostream>
#include <chrono>

class Timer{
public:
    Timer(){
        m_StartTimePoint = std::chrono::high_resolution_clock::now();
    }

    ~Timer(){
        stop();
    }

    void stop(){
        m_EndTimePoint = std::chrono::high_resolution_clock::now();

        auto start = std::chrono::time_point_cast<std::chrono::microseconds>(m_StartTimePoint).time_since_epoch().count();
        auto end = std::chrono::time_point_cast<std::chrono::microseconds>(m_EndTimePoint).time_since_epoch().count();

        long long  duration = end - start;
        printResult(duration);
    }

    void printResult(long long& duration){
        double ms = duration * 0.001;
        std::cout << "time took: " << ms << " ms" << " (" << duration << " us)\n";
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_StartTimePoint, m_EndTimePoint;
};
