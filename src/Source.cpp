#include <iostream> 
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <thread>
#include <fstream>
#include <string>
#include <chrono>
#include <ctime>
#include <cmath>

#include "complex.h"
#include "input_image.h"

#define M_PI 3.14159265358979323846 // Pi constant with double precision
#define NUM_THREADS 8

using namespace std;

void fft2(Complex *X, int N);
void separate(Complex *a, int n);
void idft(Complex *I, Complex *X, int N);

// separate even/odd elements to lower/upper halves of array respectively.
// Due to Butterfly combinations, this turns out to be the simplest way 
// to get the job done without clobbering the wrong elements.
void separate(Complex *a, int n) {
	Complex* b = new Complex[n / 2];  // get temp heap storage
	for (int i = 0; i < n / 2; i++)    // copy all odd elements to heap storage
		b[i] = a[i * 2 + 1];
	for (int i = 0; i < n / 2; i++)    // copy all even elements to lower-half of a[]
		a[i] = a[i * 2];
	for (int i = 0; i < n / 2; i++)    // copy all odd (from heap) to upper-half of a[]
		a[i + n / 2] = b[i];
	delete[] b;                 // delete heap storage
}

// N must be a power-of-2, or bad things will happen.
// Currently no check for this condition.
//
// N input samples in X[] are FFT'd and results left in X[].
// Because of Nyquist theorem, N samples means 
// only first N/2 FFT results in X[] are the answer.
// (upper half of X[] is a reflection with no new information).
void fft2(Complex *X, int N) {
	if (N < 2) {
		// bottom of recursion.
		// Do nothing here, because already X[0] = x[0]
	}
	else {
		separate(X, N);      // all evens to lower half, all odds to upper half
		fft2(X, N / 2);   // recurse even items
		fft2(X + N / 2, N / 2);   // recurse odd  items
		
		// combine results of two half recursions
		for (int k = 0; k < N / 2; k++) {
			Complex e = X[k];   // even
			Complex o = X[k + N / 2];   // odd
			
			// w is the "twiddle-factor"
			Complex w((float)cos(2 * M_PI * k / N), (float)-sin(2 * M_PI * k / N));
			X[k] = e + w * o;
			X[k + N / 2] = e - w * o;
		}
	}
}

void fft_thread_row(Complex *x, Complex *X, int N, int head, int tail) {
	while (head < tail){
		for (int k = 0; k < N; ++k) {
			for (int n = 0; n < N; ++n) {
				Complex w((float)cos(2 * M_PI * k * n / N), (float)-sin(2 * M_PI * k * n / N));
				X[head * N + k] = X[head * N + k] + x[head * N + n] * w;
			}
		}
		head++;
	}
}

void fft_thread_col(Complex *x, Complex *X, int N, int width, int head, int tail) {
	while (head < tail) {
		for (int k = 0; k < N; ++k) {
			for (int n = 0; n < N; ++n) {
				Complex w((float)cos(2 * M_PI * k * n / N), (float)-sin(2 * M_PI * k * n / N));
				X[k * width + head] = X[k * width + head] + x[n * width + head] * w;
			}
		}
		head++;
	}
}

void ifft_thread_row(Complex *X, Complex *x, int N, int head, int tail) {
	while (head < tail) {
		for (int n = 0; n < N; ++n) {
			for (int k = 0; k < N; ++k) {
				Complex w((float)cos(2 * M_PI * k * n / N), (float)sin(2 * M_PI * k * n / N));
				x[head * N + n] = x[head * N + n] + X[head * N + k] * w;
			}
			x[head * N + n].real /= N;
			x[head * N + n].imag /= N;
		}
		head++;
	}
}

void ifft_thread_col(Complex *X, Complex *x, int N, int width, int head, int tail) {
	while (head < tail) {
		for (int n = 0; n < N; ++n) {
			for (int k = 0; k < N; ++k) {
				Complex w((float)cos(2 * M_PI * k * n / N), (float)sin(2 * M_PI * k * n / N));
				x[n * width + head] = x[n * width + head] + X[k * width + head] * w;
			}
			x[n * width + head].real /= N;
			x[n * width + head].imag /= N;
		}
		head++;
	}
}

/*int main() {
	char* inputFile = "Tower256.txt";
	InputImage inImage(inputFile);
	int width = inImage.get_width();
	int height = inImage.get_height();
	cout << "width = " << width << ", height = " << height << '\n';

	Complex *h = inImage.get_image_data();
	Complex *H = inImage.get_image_data();

	// sample for first row
	Complex *x = (Complex *)malloc(width * sizeof(Complex));
	Complex *X = (Complex *)malloc(width * sizeof(Complex));
	for (int i = 0; i < width; ++i) {
		x[i] = h[i];
		X[i] = h[i];
	}
	fft2(X, width);

	// inverse FT
	Complex *I = (Complex *)malloc(width * sizeof(Complex));
	for (int i = 0; i < width; ++i) {
		I[i] = 0;
	}
	idft(I, X, width);

	cout << "n\tx[]\tX[]\t\t\tI[]\n";	// header line
	for (int i = 0; i < width; i++) { // loop to print values
		cout << i << "\t" << x[i] << "\t" << X[i] << "\t\t\t" << I[i] << '\n';
	}

	free(x); free(X);
	return 0;
}*/

int main(int argc, char* argv[]) {
	int dir = (strcmp(argv[1], "forward") == 0) ? 1 : -1;
	char *inputFile = argv[2];
	char *outputFile = argv[3];
	InputImage inImage(inputFile);
	int width = inImage.get_width();
	int height = inImage.get_height();
	cout << "width = " << width << ", height = " << height << '\n';

	if (dir == 1) { // forward
		Complex *h = inImage.get_image_data();
		Complex *X = (Complex *)malloc(width * height * sizeof(Complex));
		Complex *H = (Complex *)malloc(width * height * sizeof(Complex));
		for (int i = 0; i < width * height; ++i) {
			X[i] = 0;
			H[i] = 0;
		}
		cout << "Start clokcing...\n";
		chrono::steady_clock::time_point tStart = chrono::steady_clock::now();

		vector<thread> thread_pool;
		thread_pool.reserve(NUM_THREADS);
		//rows
		int row_per_thread = height / NUM_THREADS;
		for (int i = 0; i < NUM_THREADS; ++i) {
			int head = i * row_per_thread;
			int tail = (i == NUM_THREADS - 1) ? height : (i + 1) * row_per_thread;
			thread_pool.push_back(thread(fft_thread_row, h, X, width, head, tail));
		}
		for (int i = 0; i < NUM_THREADS; ++i)
			thread_pool[i].join();
		thread_pool.clear();
		//columns
		int col_per_thread = width / NUM_THREADS;
		for (int i = 0; i < NUM_THREADS; ++i) {
			int head = i * col_per_thread;
			int tail = (i == NUM_THREADS - 1) ? width : (i + 1) * col_per_thread;
			thread_pool.push_back(thread(fft_thread_col, X, H, height, width, head, tail));
		}
		for (int i = 0; i < NUM_THREADS; ++i)
			thread_pool[i].join();
		thread_pool.clear();

		chrono::steady_clock::time_point tEnd = chrono::steady_clock::now();
		chrono::duration<double> time_span = chrono::duration_cast<chrono::duration<double>>(tEnd - tStart);
		cout << "Time ellipsed: " << time_span.count() << " seconds... \n";
		cout << "Printing results...\n";
		ofstream myfile(outputFile);
		if (myfile.is_open()) {
			for (int i = 0; i < height; ++i) {
				for (int j = 0; j < width - 1; ++j) {
					myfile << X[i * width + j] << ',';
				}
				myfile << X[i * width + width - 1] << '\n';
			}
		}
		myfile.close();
		free(X); free(H);
	}
	else { // reverse DFT
		Complex *H = inImage.get_image_data();
		Complex *X = (Complex *)malloc(width * height * sizeof(Complex));
		Complex *h = (Complex *)malloc(width * height * sizeof(Complex));
		for (int i = 0; i < width; ++i) {
			X[i] = 0;
			h[i] = 0;
		}
		cout << "Start clokcing...\n";
		chrono::steady_clock::time_point tStart = chrono::steady_clock::now();

		vector<thread> thread_pool;
		thread_pool.reserve(NUM_THREADS);
		//columns
		int col_per_thread = width / NUM_THREADS;
		for (int i = 0; i < width * height; ++i) {
			h[i] = 0;
			X[i] = 0;
		}
		for (int i = 0; i < NUM_THREADS; ++i) {
			int head = i * col_per_thread;
			int tail = (i == NUM_THREADS - 1) ? width : (i + 1) * col_per_thread;
			thread_pool.push_back(thread(ifft_thread_col, H, X, height, width, head, tail));
		}
		for (int i = 0; i < NUM_THREADS; ++i)
			thread_pool[i].join();
		thread_pool.clear();
		//rows
		int row_per_thread = height / NUM_THREADS;
		for (int i = 0; i < NUM_THREADS; ++i) {
			int head = i * row_per_thread;
			int tail = (i == NUM_THREADS - 1) ? height : (i + 1) * row_per_thread;
			thread_pool.push_back(thread(ifft_thread_row, X, h, width, head, tail));
		}
		for (int i = 0; i < NUM_THREADS; ++i)
			thread_pool[i].join();
		thread_pool.clear();

		chrono::steady_clock::time_point tEnd = chrono::steady_clock::now();
		chrono::duration<double> time_span = chrono::duration_cast<chrono::duration<double>>(tEnd - tStart);
		cout << "Time ellipsed: " << time_span.count() << " seconds... \n";
		cout << "Printing results...\n";
		ofstream myfile(outputFile);
		if (myfile.is_open()) {
			for (int i = 0; i < height; ++i) {
				for (int j = 0; j < width - 1; ++j) {
					myfile << h[i * width + j].real << ',';
				}
				myfile << h[i * width + width - 1].real << '\n';
			}
		}
		myfile.close();
		free(X); free(H);
	}

	cout << "finished...\n";
	return 0;
}