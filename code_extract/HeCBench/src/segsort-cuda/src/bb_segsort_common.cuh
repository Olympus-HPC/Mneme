/*
* (c) 2015-2019 Virginia Polytechnic Institute & State University (Virginia Tech)
*          2020 Robin Kobus (kobus@uni-mainz.de)
*
*   This program is free software: you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*   the Free Software Foundation, version 2.1
*
*   This program is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU General Public License, version 2.1, for more details.
*
*   You should have received a copy of the GNU General Public License
*
*/

#ifndef _H_BB_SEGSORT_COMMON
#define _H_BB_SEGSORT_COMMON

template<class T>
void show_d(T *arr_d, int n, std::string prompt)
{
    std::vector<T> arr_h(n);
    cudaMemcpy(&arr_h[0], arr_d, sizeof(T)*n, cudaMemcpyDeviceToHost);
    std::cout << prompt;
    for(auto v: arr_h) std::cout << v << ", ";
    std::cout << std::endl;
}

template<class T>
void show_h(T *arr_h, int n, std::string prompt)
{
    std::cout << prompt;
    for(int i = 0; i < n; ++i) std::cout << arr_h[i] << ", ";
    std::cout << std::endl;
}

#endif
