#ifndef _SYRAH_CYCLE_TIMER_H_
#define _SYRAH_CYCLE_TIMER_H_

#if defined(__APPLE__)
#if defined(__x86_64__)
    #include <sys/sysctl.h>
#else
    #include <mach/mach.h>
    #include <mach/mach_time.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#elif _WIN32
    #include <windows.h>
    #include <time.h>
#else
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    #include <sys/time.h>
#endif

/**
 * @brief CPU cycle timer.
 * 
 * This uses the cycle counter of the processor. Different
 * processors in the system will have different values for this. If
 * your process moves across processors, then the delta time you
 * measure will likely be incorrect. This is mostly for fine
 * grained measurements where the process is likely to be on the
 * same processor. For more global things you should use the
 * Time interface.
 * 
 * Also note that if your processors' speeds change (i.e. processors
 * scaling) or if you are in a heterogenous environment, you will
 * likely get spurious results.
 */
class cycle_timer {
public:
    typedef unsigned long long sys_clock;

    /**
     * @brief Gets current CPU time.
     * @details Time zero is at some arbitrary point in the past.
     * @return the current CPU time, in terms of clock ticks
     */
    static sys_clock current_ticks() {
#if defined(__APPLE__) && !defined(__x86_64__)
        return mach_absolute_time();
#elif defined(_WIN32)
        LARGE_INTEGER qwTime;
        QueryPerformanceCounter(&qwTime);
        return qwTime.QuadPart;
#elif defined(__x86_64__)
        unsigned int a, d;
        asm volatile("rdtsc" : "=a" (a), "=d" (d));
        return static_cast<unsigned long long>(a) |
            (static_cast<unsigned long long>(d) << 32);
#elif defined(__ARM_NEON__) && 0 // mrc requires superuser
        unsigned int val;
        asm volatile("mrc p15, 0, %0, c9, c13, 0" : "=r"(val));
        return val;
#else
        timespec spec;
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &spec);
        return cycle_timer::sys_clock(static_cast<float>(spec.tv_sec) * 1e9 + static_cast<float>(spec.tv_nsec));
#endif
    }

    /**
     * @brief Gets current CPU second.
     * @details This is slower than current_ticks(). Time zero is at some arbitrary point in the past.
     * @return the current CPU time, in terms of seconds
     */
    static double current_seconds() {
        return current_ticks() * seconds_per_tick();
    }

    /**
     * @brief Gets the conversion from seconds to ticks.
     * @details
     * @return ticks per second
     */
    static double ticks_per_second() {
        return 1.0 / seconds_per_tick();
    }

    /**
     * @brief Gets time tick units.
     * @details
     * @return tick units
     */
    static const char *tick_units() {
#if defined(__APPLE__) && !defined(__x86_64__)
        return "ns";
#elif defined(__WIN32__) || defined(__x86_64__)
        return "cycles";
#else
        return "ns"; // clock_gettime
#endif
    }

    /**
     * @brief Gets the conversion from ticks to seconds.
     * @details
     * @return seconds per tick
     */
    static double seconds_per_tick() {
        static bool initialized = false;
        static double seconds_per_tick_val;

        if (initialized)
            return seconds_per_tick_val;

#if defined(__APPLE__)
#ifdef __x86_64__
        int args[] = { CTL_HW, HW_CPU_FREQ };
        unsigned int Hz;
        size_t len = sizeof(Hz);
        if (sysctl(args, 2, &Hz, &len, NULL, 0) != 0) {
            fprintf(stderr, "failed to initialize seconds_per_tick_val\n");
            exit(-1);
        }
        seconds_per_tick_val = 1.0 / (double) Hz;
#else
        mach_timebase_info_data_t time_info;
        mach_timebase_info(&time_info);

        // scales to nanoseconds without 1e-9f
        seconds_per_tick_val = (1e-9 * static_cast<double>(time_info.numer)) /
            static_cast<double>(time_info.denom);
#endif
#elif defined(_WIN32)
        LARGE_INTEGER qwTicksPerSec;
        QueryPerformanceFrequency(&qwTicksPerSec);
        seconds_per_tick_val = 1.0 / static_cast<double>(qwTicksPerSec.QuadPart);
#else
        FILE *fp = fopen("/proc/cpuinfo", "r");
        char input[1024];
        if (!fp) {
            fprintf(stderr, "cycle_timer::seconds_per_tick failed: cannot find /proc/cpuinfo\n");
            exit(-1);
        }
        // in case we do not find it, e.g. on the N900
        seconds_per_tick_val = 1e-9;
        while (!feof(fp) && fgets(input, 1024, fp)) {
            // NOTE(boulos): because reading cpuinfo depends on dynamic
            // frequency scaling it is better to read the @ sign first
            float GHz, MHz;
            if (strstr(input, "model name")) {
                char *at_sign = strstr(input, "@");
                if (at_sign) {
                    char *after_at = at_sign + 1;
                    char *GHz_str = strstr(after_at, "GHz");
                    char *MHz_str = strstr(after_at, "MHz");
                    if (GHz_str) {
                        *GHz_str = '\0';
                        if (1 == sscanf(after_at, "%f", &GHz)) {
                            seconds_per_tick_val = 1e-9f / GHz;
                            break;
                        }
                    } else if (MHz_str) {
                        *MHz_str = '\0';
                        if (1 == sscanf(after_at, "%f", &MHz)) {
                            seconds_per_tick_val = 1e-6f / MHz;
                            break;
                        }
                    }
                }
            } else if (1 == sscanf(input, "cpu MHz : %f", &MHz)) {
                seconds_per_tick_val = 1e-6f / MHz;
                break;
            }
        }
        fclose(fp);
#endif
        initialized = true;
        return seconds_per_tick_val;
    }

    /**
     * @brief Gets the conversion from ticks to milliseconds
     * @details
     * @return milliseconds per tick
     */
    static double milliseconds_per_tick() {
        return seconds_per_tick() * 1000.0;
    }

private:
    cycle_timer();
};

#endif
