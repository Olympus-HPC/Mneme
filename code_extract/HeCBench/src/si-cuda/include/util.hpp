#ifndef UTIL_HPP
#define UTIL_HPP

#pragma once

#include <string>
#include <sstream>
#include <iomanip>
#include <vector>

struct tile {
    unsigned int id;
    unsigned int start;
    unsigned int end;
    unsigned int length;

    tile(unsigned int id, unsigned int start, unsigned int end) : id(id), start(start), end(end) {
        length = end - start;
    }
};

typedef std::pair<tile, tile> tile_pair;

std::vector<tile> splitToTiles(unsigned int cardinality, unsigned int partition)
{
    std::vector<tile> tiles;

    unsigned int numOfTiles = std::ceil((double) cardinality / (double) partition);

    for (unsigned int i = 0; i < numOfTiles; ++i) {
        unsigned int start = i * partition;
        unsigned int end = start + partition;
        if (end >= cardinality) end -= (end - cardinality);
        tiles.push_back(tile(i, start, end));
    }

    return tiles;
}

std::vector<tile_pair> findTilePairs(unsigned int cardinality, unsigned int partition)
{
    std::vector<tile_pair> pairs;
    std::vector<tile> tiles = splitToTiles(cardinality, partition);

    for (unsigned int i = 0; i < tiles.size(); ++i) {
        for (unsigned int j = i; j < tiles.size(); ++j) {
            pairs.push_back(std::make_pair(tiles[i], tiles[j]));
        }
    }

    return pairs;
}

unsigned long long combination(unsigned int n, unsigned int k)
{
    if (k > n) return 0;
    if (k * 2 > n) k = n-k;
    if (k == 0) return 1;

    unsigned long long result = n;
    for( unsigned long long i = 2; i <= k; ++i ) {
        result *= (n-i+1);
        result /= i;
    }
    return result;
}

unsigned long long triangular_index(unsigned long long n, unsigned long long i, unsigned long long j) {
    return (n * (n - 1) / 2) - (n - i) * ((n - i) - 1) / 2 + j - i - 1;
}

unsigned long long quadratic_index(unsigned long long n, unsigned long long i, unsigned long long j) {
    return i * n + j;
}

std::string formatBytes(size_t bytes)
{
    size_t gb = 1073741824;
    size_t mb = 1048576;
    size_t kb = 1024;

    std::stringstream stream;
    stream << std::fixed << std::setprecision(3);

    if (bytes >= gb) {
        stream << ((double) bytes / (double) gb) << " GB";
    } else if (bytes >= mb) {
        stream << ((double) bytes / (double) mb) << " MB";
    } else if (bytes >= kb) {
        stream << ((double) bytes / (double) kb) << " KB";
    } else {
        stream << bytes << " bytes";
    }

    return stream.str();
}

#endif //UTIL_HPP
