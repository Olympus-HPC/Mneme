//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <iomanip>
#include <iostream>
#include <cuda.h>
#include "utils.hpp"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

using namespace std;

__global__ 
void gamma_correction(ImgPixel* pixel) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	// Lambda to process image with gamma = 2
	const float v = (0.3f * pixel[i].r + 0.59f * pixel[i].g + 0.11f * pixel[i].b) / 255.0;

	std::uint8_t gamma_pixel = static_cast<std::uint8_t>(255 * v * v);
	if (gamma_pixel > 255) gamma_pixel = 255;
	pixel[i].set(gamma_pixel, gamma_pixel, gamma_pixel, gamma_pixel);
}


int main() {
	// Image size is width x height
	int width = 2560;
	int height = 1600;

	Img<ImgFormat::BMP> image{width, height};
	ImgFractal fractal{width, height};

	// Lambda to process image with gamma = 2
	auto gamma_f = [](ImgPixel& pixel) {
		float v = (0.3f * pixel.r + 0.59f * pixel.g + 0.11f * pixel.b) / 255.0;

		std::uint8_t gamma_pixel = static_cast<std::uint8_t>(255 * v * v);
		if (gamma_pixel > 255) gamma_pixel = 255;
		pixel.set(gamma_pixel, gamma_pixel, gamma_pixel, gamma_pixel);
	};

	// fill image with created fractal
	int index = 0;
	image.fill([&index, width, &fractal](ImgPixel& pixel) {
			int x = index % width;
			int y = index / width;

			auto fractal_pixel = fractal(x, y);
			if (fractal_pixel < 0) fractal_pixel = 0;
			if (fractal_pixel > 255) fractal_pixel = 255;
			pixel.set(fractal_pixel, fractal_pixel, fractal_pixel, fractal_pixel);

			++index;
			});

	Img<ImgFormat::BMP> image2 = image;
#ifdef DEBUG
	image.write("fractal_original.bmp");
#endif

	// call standard serial function for correctness check
	image.fill(gamma_f);
#ifdef DEBUG
	image.write("fractal_gamma_serial.bmp");
#endif

	ImgPixel* pixel;
	cudaMalloc((void**)&pixel, sizeof(ImgPixel) * image2.width() * image2.height());
	cudaMemcpy(pixel, image2.data(), sizeof(ImgPixel) * image2.width() * image2.height(), cudaMemcpyHostToDevice);
	gamma_correction <<< dim3(width * height / BLOCK_SIZE), dim3(BLOCK_SIZE) >>> (pixel); 
	cudaMemcpy(image2.data(), pixel, sizeof(ImgPixel) * image2.width() * image2.height(), cudaMemcpyDeviceToHost);

	// check correctness
	if (check(image.begin(), image.end(), image2.begin())) {
		std::cout << "Pass";
	} else {
		std::cout << "Fail";
	}

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);

	std::cout << ". Run on " << props.name << std::endl;

#ifdef DEBUG
	image.write("fractal_gamma_parallel.bmp");
#endif
	cudaFree(pixel); 

	return 0;
}
