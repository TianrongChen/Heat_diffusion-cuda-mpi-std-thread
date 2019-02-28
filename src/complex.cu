#include "complex.cuh"
#include <cmath>


const float PI = 3.14159265358979f;

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
    if(fabsf(rhs.imag) < 1e-10) c.imag = 0.0f;
    if(fabsf(rhs.real) < 1e-10) c.real = 0.0f;

    if(c.imag == 0) {
        os << c.real;
    }
    else {
        os << "(" << c.real << "," << c.imag << ")";
    }
    return os;
}