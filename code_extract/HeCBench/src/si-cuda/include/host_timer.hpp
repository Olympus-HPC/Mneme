
#ifndef HOST_TIMER_HPP
#define HOST_TIMER_HPP

#pragma once

#include <chrono>
#include <algorithm>
#include <fmt/core.h>

struct Interval {
    typedef std::chrono::high_resolution_clock::time_point interval_point;
    interval_point begin;
    interval_point end;
    std::string name;
    Interval(const std::string& name) : name(name) {}
};

struct HostTimer {
    std::vector<Interval*> intervals;
    Interval* add(const std::string& name) {
        auto* interval = new Interval(name);
        interval->begin = std::chrono::high_resolution_clock::now();
        intervals.push_back(interval);
        return interval;
    }
    static void finish(Interval * interval) {
        interval->end = std::chrono::high_resolution_clock::now();
    }
    double sum(std::string const& name) const {
        double sum = 0.0;
        for(auto& interval : intervals) {
            if (interval->name == name) {
                sum += (std::chrono::duration_cast<std::chrono::microseconds>
                               (interval->end - interval->begin).count() / 1000.0);
            }
        }
        return sum;
    }
    double total() const {
        double total = 0.0;
        for(auto& interval : intervals) {
            total += (std::chrono::duration_cast<std::chrono::microseconds>
                             (interval->end - interval->begin).count() / 1000.0);
        }
        return total;
    }
    void print() {
        fmt::print("┌{0:─^{1}}┐\n", "Host Timings (in ms)", 51);
        std::vector<std::string> distinctNames;
        for(auto& interval : intervals) {
            if (std::find(distinctNames.begin(), distinctNames.end(), interval->name) == distinctNames.end())
                distinctNames.push_back(interval->name);
        }
        for(auto& name : distinctNames) {
            fmt::print("│{1: ^{0}}|{2: ^{0}}│\n", 25, name, sum(name));
        }
        fmt::print("└{1:─^{0}}┘\n", 51, "");
        fmt::print("│{1: ^{0}}|{2: ^{0}}│\n", 25, "Total", total());
        fmt::print("└{1:─^{0}}┘\n", 51, "");
    }
    ~HostTimer() {
        auto it = intervals.begin();
        for(; it != intervals.end(); ++it) delete *it;
    }
};


#endif // HOST_TIMER_HPP
