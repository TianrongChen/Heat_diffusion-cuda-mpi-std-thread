//
// Created by brian on 11/20/18.
//
#include "complex.cuh"
#include <fstream>
#include <sstream>
#include <iostream>
#pragma once


class InputImage {
public:

    InputImage(const char* filename);
    int get_width() const;
    int get_height() const;

    //returns a pointer to the image data.  Note the return is a 1D
    //array which represents a 2D image.  The data for row 1 is
    //immediately following the data for row 0 in the 1D array
    Complex_cuda* get_image_data() const;

    //use this to save output from forward DFT
    void save_image_data(const char* filename, Complex_cuda* d, int w, int h);
    //use this to save output from reverse DFT
    void save_image_data_real(const char* filename, Complex_cuda* d, int w, int h);

private:
    int w;
    int h;
    Complex_cuda* data;
};


InputImage::InputImage(const char* filename) {
    std::ifstream ifs(filename);
    if(!ifs) {
        std::cout << "Can't open image file " << filename << std::endl;
        exit(1);
    }

    ifs >> w >> h;
    data = new Complex_cuda[w * h];
    for(int r = 0; r < h; ++r) {
        for(int c = 0; c < w; ++c) {
            float real;
            ifs >> real;
            data[r * w + c] = Complex_cuda(real);
        }

    }
}

int InputImage::get_width() const {
    return w;
}

int InputImage::get_height() const {
    return h;
}

Complex_cuda* InputImage::get_image_data() const {
    return data;
}

void InputImage::save_image_data(const char *filename, Complex_cuda *d, int w, int h) {
    std::ofstream ofs(filename);
    if(!ofs) {
        std::cout << "Can't create output image " << filename << std::endl;
        return;
    }

    ofs << w << " " << h << std::endl;

    for(int r = 0; r < h; ++r) {
        for(int c = 0; c < w; ++c) {
            ofs << d[r * w + c] << " ";
        }
        ofs << std::endl;
    }
}

void InputImage::save_image_data_real(const char* filename, Complex_cuda* d, int w, int h) {
    std::ofstream ofs(filename);
    if(!ofs) {
        std::cout << "Can't create output image " << filename << std::endl;
        return;
    }

    ofs << w << " " << h << std::endl;

    for (int r = 0; r < h; ++r) {
        for (int c = 0; c < w; ++c) {
            ofs << d[r * w + c].real << " ";
        }
        ofs << std::endl;
    }
}