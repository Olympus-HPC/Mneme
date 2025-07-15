#ifndef UTILS_H
#define UTILS_H

#include <fstream>
#include <iostream>

namespace io
{
    size_t FileSize(const std::string filename)
    {
        std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
        return static_cast<size_t>(in.tellg());
    }

    template <typename T>
    T *read_binary_to_new_array(const std::string &fname)
    {
        std::ifstream ifs(fname.c_str(), std::ios::binary | std::ios::in);
        if (not ifs.is_open())
        {
            std::cerr << "fail to open " << fname << std::endl;
            exit(1);
        }
        size_t dtype_len = FileSize(fname) / sizeof(T);
        auto _a = new T[dtype_len]();
        ifs.read(reinterpret_cast<char *>(_a), std::streamsize(dtype_len * sizeof(T)));
        ifs.close();
        return _a;
    }

    template <typename T>
    void read_binary_to_array(const std::string &fname, T *_a, size_t dtype_len)
    {
        std::ifstream ifs(fname.c_str(), std::ios::binary | std::ios::in);
        if (not ifs.is_open())
        {
            std::cerr << "fail to open " << fname << std::endl;
            exit(1);
        }
        ifs.read(reinterpret_cast<char *>(_a), std::streamsize(dtype_len * sizeof(T)));
        ifs.close();
    }

    template <typename T>
    void write_array_to_binary(const std::string &fname, T *const _a, size_t const dtype_len)
    {
        std::ofstream ofs(fname.c_str(), std::ios::binary | std::ios::out);
        if (not ofs.is_open())
            return;
        ofs.write(reinterpret_cast<const char *>(_a), std::streamsize(dtype_len * sizeof(T)));
        ofs.close();
    }

}

#endif // UTILS_H