#pragma once
#include <chrono>

inline double wall_seconds()
{
    using Clock = std::chrono::steady_clock;
    static const auto start = Clock::now();

    return std::chrono::duration<double>(
        Clock::now() - start
    ).count();
}