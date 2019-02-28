#pragma once

#include <iostream>
#include <cuda.h>
#include <cmath>

const float PI = 3.14159265358979f;

class Complex_cuda{
public:
    __device__ __host__ Complex_cuda();
    __device__ __host__ Complex_cuda(float r, float i);
    __device__ __host__ Complex_cuda(float r);
    __device__ __host__ Complex_cuda operator+(const Complex_cuda& b) const;
    __device__ __host__ Complex_cuda operator-(const Complex_cuda& b) const;
    __device__ __host__ Complex_cuda operator*(const Complex_cuda& b) const;

    __device__ __host__ Complex_cuda mag() const;
    __device__ __host__ Complex_cuda angle() const;
    __device__ __host__ Complex_cuda conj() const;

    float real;
    float imag;
};

//std::ostream& operator<<(std::ostream& os, const Complex_cuda& rhs);

__device__ __host__ Complex_cuda::Complex_cuda() : real(0.0f), imag(0.0f) {}

__device__ __host__ Complex_cuda::Complex_cuda(float r) : real(r), imag(0.0f) {}

__device__ __host__ Complex_cuda::Complex_cuda(float r, float i) : real(r), imag(i) {}

__device__ __host__ Complex_cuda Complex_cuda::operator+(const Complex_cuda &b) const {
	return Complex_cuda(this->real + b.real, this->imag + b.imag);
}

__device__ __host__ Complex_cuda Complex_cuda::operator-(const Complex_cuda &b) const {
	return Complex_cuda(this->real - b.real, this->imag - b.imag);
}

__device__ __host__ Complex_cuda Complex_cuda::operator*(const Complex_cuda &b) const {
	return Complex_cuda(this->real * b.real - this->imag * b.imag, this->real * b.imag + this->imag * b.real);
}

__device__ __host__ Complex_cuda Complex_cuda::mag() const {
	return Complex_cuda(sqrt(pow(this->real, 2.0) + pow(this->imag, 2.0)));
}

__device__ __host__ Complex_cuda Complex_cuda::angle() const {
	if (this->imag > 0) return Complex_cuda(acos(this->real / this->mag().real));
	if (this->imag < 0) return Complex_cuda(-acos(this->real / this->mag().real));
	if (this->real > 0 && this->imag == 0) return Complex_cuda(0);
	return Complex_cuda(PI);
}

__device__ __host__ Complex_cuda Complex_cuda::conj() const {
	return Complex_cuda(this->real, -this->imag);
}

std::ostream& operator<< (std::ostream& os, const Complex_cuda& rhs) {
    Complex_cuda c(rhs);
    if(fabsf(rhs.imag) < 1e-5) c.imag = 0.0f;
    if(fabsf(rhs.real) < 1e-5) c.real = 0.0f;

    if(c.imag == 0) {
        os << c.real;
    }
    else {
        os << "(" << c.real << "," << c.imag << ")";
    }
    return os;
}
