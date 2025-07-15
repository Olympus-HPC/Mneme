#include <string>
#include <vector>
#include <random>
#include <fmt/core.h>
#include <cxxopts.hpp>
#include <boost/dynamic_bitset.hpp>
#include <zipfian_int_distribution.h>
#include <ProgressBar.hpp>
#include "io.hpp"

boost::dynamic_bitset<> generateBitset(unsigned int universe,
                                       unsigned int iterations,
                                       std::string& distribution,
                                       double param,
                                       zipfian_int_distribution<unsigned int>::param_type p);
std::vector<unsigned int> bitsetToVector(boost::dynamic_bitset<>& bitset);

// taken from zipfian_int_distribution.h (it's private)
double zeta(unsigned long __n, double __theta);

int main(int argc, char** argv)
{
    try {
        std::string outputPath;
        std::string distribution = "uniform";
        unsigned int iterations = 10000000;
        double bernoulliP = 0.5; // bernoulli probability
        double stddev = 2.0; // standard deviation
        unsigned int k = 100; // dataset cardinality
        double skew = 0.9;

        cxxopts::Options options(argv[0], "Generate dataset");

        options.add_options()
                ("distribution", "Set element distribution {uniform|normal|bernoulli|zipf} (default: uniform)", cxxopts::value<std::string>(distribution))
                ("output", "Output prefix name for the generated dataset", cxxopts::value<std::string>(outputPath))
                ("prob", "Bernoulli probability", cxxopts::value<double>(bernoulliP))
                ("stdev", "Standard Deviation for normal distribution (default: 2)", cxxopts::value<double>(stddev))
                ("skew", "Theta (i.e. skew factor for zipf distribution)", cxxopts::value<double>(skew))
                ("iterations", "Number of iterations (i.e. approximate set size, default: 10000000)", cxxopts::value<unsigned int>(iterations))
                ("universe", "Universe cardinality", cxxopts::value<unsigned long>())
                ("k", "Dataset size (i.e. number of sets)", cxxopts::value<unsigned int>(k))
                ("help", "Print help");

        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            fmt::print("{}\n", options.help());
            return 0;
        }

        if (!result.count("universe")) {
            fmt::print("{}\n", "No universe given! Exiting...");
            return 1;
        }

        unsigned long universe = result["universe"].as<unsigned long>();

        if (!result.count("output")) {
            outputPath = "out";
        }

        // construct output name
        std::string tmp = distribution + "_" + std::to_string(k) + "_" + std::to_string(universe);

        double param = 0;

        if (distribution == "uniform") {
            tmp += "_" + std::to_string(iterations);
        } else if (distribution == "normal") {
            tmp += "_" + std::to_string(iterations) + "_" + std::to_string(stddev);
            param = stddev;
        } else if (distribution == "bernoulli") {
            tmp += "_" + std::to_string(bernoulliP);
            param = bernoulliP;
        } else { // zipf
            tmp += "_" + std::to_string(iterations) + "_" + std::to_string(skew);
            param = skew;
        }

        tmp.append(".bin");

        // construct dataset
        std::vector<std::vector<unsigned int>> dataset;
        dataset.reserve(k);

        progresscpp::ProgressBar progressBar(k, 70, '#', '-');

        unsigned long totalElements = 0;


        zipfian_int_distribution<unsigned int>::param_type p;

        if (distribution == "zipf") {
            // calculate zeta just once
            p = zipfian_int_distribution<unsigned int>::param_type(1, universe, param, zeta(universe, param));
        }

        for (unsigned int i = 0; i < k; ++i) {
            boost::dynamic_bitset<> bitset = generateBitset(universe, iterations, distribution, param, p);
            std::vector<unsigned int> set = bitsetToVector(bitset);
            dataset.push_back(set);
            totalElements += set.size();
            ++progressBar;
            progressBar.display();
        }

        progressBar.done();

        fmt::print("Sorting sets in ascending order\n");

        std::sort(dataset.begin(), dataset.end(), [](const std::vector<unsigned int>& a, const std::vector<unsigned int>& b) {
            return a.size() < b.size();
        });

        std::string outname = outputPath + "_asc_" + tmp;
        fmt::print("Writing dataset to {}\n", outname);
        writeDataset(k, universe, totalElements, dataset, outname);

        fmt::print("Sorting sets in descending order\n");

        std::sort(dataset.begin(), dataset.end(), [](const std::vector<unsigned int>& a, const std::vector<unsigned int>& b) {
            return a.size() > b.size();
        });

        outname = outputPath + "_desc_" + tmp;
        fmt::print("Writing dataset to {}\n", outname);
        writeDataset(k, universe, totalElements, dataset, outname);


        fmt::print("Finished!\n");

    } catch (const cxxopts::OptionException& e) {
        fmt::print("{}\n", e.what());
        return 1;
    }
    return 0;
}

boost::dynamic_bitset<> generateBitset(unsigned int universe,
                                       unsigned int iterations,
                                       std::string& distribution,
                                       double param,
                                       zipfian_int_distribution<unsigned int>::param_type p)
{
    boost::dynamic_bitset<> bitset(universe);

    std::mt19937_64 gen(19937);

    if (distribution == "uniform") {
        std::uniform_int_distribution<> ud(1, universe);

        for (unsigned int i = 0; i < iterations; ++i) {
            bitset.set(std::round(ud(gen)));
        }
    } else if (distribution == "normal") {
        std::normal_distribution<> nd(universe / 2, param);

        for (unsigned int i = 0; i < iterations; ++i) {
            bitset.set(std::round(nd(gen)));
        }
    } else if (distribution == "bernoulli") {
        std::bernoulli_distribution bd(param);
        for (unsigned int i = 0; i < universe; ++i) {
            if (bd(gen)) {
                bitset.set(i);
            }
        }
    } else { // zipf distribution
        zipfian_int_distribution<unsigned int> zipf(p);

        for (unsigned int i = 0; i < iterations; ++i) {
            bitset.set(zipf(gen));
        }
    }

    return bitset;
}

double zeta(unsigned long __n, double __theta)
{
    double ans = 0.0;
    for(unsigned long i = 1; i <= __n; ++i)
        ans += std::pow(1.0 / i, __theta);
    return ans;
}

std::vector<unsigned int> bitsetToVector(boost::dynamic_bitset<>& bitset)
{
    std::vector<unsigned int> set;
    set.reserve(bitset.count());
    for (unsigned int i = 0; i < bitset.size(); ++i) {
        if (bitset[i]) {
            set.push_back(i + 1);
        }
    }
    return set;
}

