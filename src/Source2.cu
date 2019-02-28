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
#include <cuda.h>

#include "input_image.cuh"

#define M_PI 3.14159265358979323846 // Pi constant with double precision

using namespace std;


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) getchar();
    }
}


int Divup(int a, int b)
{

    if (a % b)
    {
        return a / b + 1; /* add in additional block */
    }
    else
    {
        return a / b; /* divides cleanly */
    }
}


__global__ void fft(Complex_cuda *new_a,Complex_cuda *a, Complex_cuda *omega, int width);
__global__ void ifft(Complex_cuda *a, int num_col);
__device__ void transpose(Complex_cuda *Matrix, int num_col);

__global__ void fft(Complex_cuda *new_a,Complex_cuda *a, Complex_cuda *omega, int width)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int height=width;

    int start=x/width*width;
    int end=(x/width+1)*width;
    printf("gridsize=%d,blockIdx.x=%d, blockDim.x=%d,x=%d,start=%d, end=%d\n",gridDim.x,blockIdx.x, blockDim.x, x,start,end);

        for(int i=start; i<end; ++i)
        {
            if(i==0)
            {
                new_a[x]=new_a[x]+a[i+start];
            }
            else
            {
                new_a[x]=new_a[x]+ Complex_cuda(cos(2*M_PI/(float)width*i), -sin(2*M_PI/(float)width*i))*a[i+start];
            }
        }
    __syncthreads();
    
    if((x%width)>x/width && x<width*height)
    {
        Complex_cuda tmp = new_a[x%width+x/width*width];        
        new_a[x%width+x/width*width]=new_a[x];        
        new_a[x]=tmp;
    }     
    __syncthreads();  
    //printf("1: x=%d\n", x);
    while(start<end)
    {
        for(int i=0; i<width; ++i)
        {
            if(i==0)
            {
                a[x]=a[x]+new_a[i+start];
            }
            else
            {
                a[x]=a[x]+ Complex_cuda(cos(2*M_PI/(float)width*i), -sin(2*M_PI/(float)width*i))*new_a[i+start];
            }
        }
        start++;
    }
    __syncthreads();
    if((x%width)>x/width && x<width*height)
    {
        Complex_cuda tmp = a[x%width+x/width*width];        
        a[x%width+x/width*width]=a[x];        
        a[x]=tmp;
    }         
    __syncthreads();  
}
__global__ void ifft(Complex_cuda *a, Complex_cuda *omega_inverse, int num_col)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i=0; i<num_col; ++ i)  {
        omega_inverse[i] = Complex_cuda(cos(2*M_PI/num_col*i), -sin(2*M_PI/num_col*i)).conj();
    }

    for(int i=0, j=0; i<num_col ; ++i)  
    {
        if (i>j)  
        {
            Complex_cuda tmp = a[x*num_col+i];
            a[x*num_col+i] = a[x*num_col+j];
            a[x*num_col+j] = tmp;
        }
		for(int l=num_col>>1; (j^=l)<l; l>>=1);
	}

    for(int l=2; l<=num_col; l<<=1)  
    {
        int m = l/2;
        for(Complex_cuda *p=a; p!=a+num_col; p=p+l)  
        {
            for(int i=0; i<m; ++i)  
            {
                Complex_cuda t = omega_inverse[num_col/l*i] * p[x*num_col+m+i];
                p[x*num_col+m+i] = p[x*num_col+i] - t ;
                p[x*num_col+i] = p[x*num_col+i] + t ;
            }
        }
    }

    __syncthreads();
    transpose(a, num_col); /* get transpose result and do ifft again == do ifft on another axis*/
    __syncthreads();

    for(int i=0, j=0; i<num_col ; ++i)  
    {
        if (i>j)  
        {
            Complex_cuda tmp = a[x*num_col+i];
            a[x*num_col+i] = a[x*num_col+j];
            a[x*num_col+j] = tmp;
        }
		for(int l=num_col>>1; (j^=l)<l; l>>=1);
	}

    for(int l=2; l<=num_col; l<<=1)  
    {
        int m = l/2;
        for(Complex_cuda *p=a; p!=a+num_col; p=p+l)  
        {
            for(int i=0; i<m; ++i)  
            {
                Complex_cuda t = omega_inverse[num_col/l*i] * p[x*num_col+m+i];
                p[x*num_col+m+i] = p[x*num_col+i] - t ;
                p[x*num_col+i] = p[x*num_col+i] + t ;
            }
        }
    }
    transpose(a, num_col);
    __syncthreads();

}

__device__ void transpose(Complex_cuda *Matrix, int num_col)
{    
    const int x = blockIdx.x * blockDim.x + threadIdx.x ;  
    for(int i=x+1; i<num_col; i++)
    {
        Complex_cuda tmp = Matrix[x*num_col+i];        
        Matrix[x*num_col+i]=Matrix[i*num_col+x];        
        Matrix[i*num_col+x]=tmp;
    }   
}


int main(int argc, char* argv[]) {
	int dir = (strcmp(argv[1], "forward") == 0) ? 1 : -1;
	char *inputFile = argv[2];
	char *outputFile = argv[3];
	InputImage inImage(inputFile);
	int width = inImage.get_width();
    int height = inImage.get_height();
    
	cout << "width = " << width << ", height = " << height << '\n';

	if (dir == 1) { // forward
        Complex_cuda *h = inImage.get_image_data();
		Complex_cuda *X = (Complex_cuda*)malloc(width * height * sizeof(Complex_cuda));
        Complex_cuda *h_cuda;
        Complex_cuda *h_new_cuda;
        cudaMalloc((void **)&h_cuda, width * height * sizeof(Complex_cuda));
        Complex_cuda* omega = (Complex_cuda *)malloc(width*sizeof(Complex_cuda));
        Complex_cuda* omega_cuda;
        cudaMalloc((void **)&omega_cuda, width * sizeof(Complex_cuda));
        cudaMalloc((void **)&h_new_cuda, width*height * sizeof(Complex_cuda));
		for (int i = 0; i < width * height; ++i) {
			X[i] = 0;
		}
		cout << "Start clokcing...\n";
		chrono::steady_clock::time_point tStart = chrono::steady_clock::now();

        int block_size = (height>512)?(512):(height);
        dim3 grid(Divup(height,block_size),1);    
        dim3 block(block_size,1);

        cudaMemcpy(h_cuda, h, width*height*sizeof(Complex_cuda), cudaMemcpyHostToDevice);
        //cudaMemcpy(omega_cuda, omega, width*sizeof(Complex_cuda), cudaMemcpyHostToDevice);
        cudaMemcpy(h_new_cuda, X, width*height*sizeof(Complex_cuda), cudaMemcpyHostToDevice);
        fft <<<width*height/256, 256>>>(h_new_cuda,h_cuda, omega_cuda,height);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        cudaMemcpy(X, h_cuda, width*height*sizeof(Complex_cuda), cudaMemcpyDeviceToHost);


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
        free(h); 
        free(X);
        cudaFree(h_cuda);
	}
	else { // reverse DFT
		Complex_cuda *H = inImage.get_image_data();
        Complex_cuda *h = (Complex_cuda *)malloc(width*height*sizeof(Complex_cuda));
        Complex_cuda *H_cuda;
        Complex_cuda* omega_inverse = (Complex_cuda *)malloc(width*sizeof(Complex_cuda));
        Complex_cuda* omega_inverse_cuda;
        cudaMalloc((void **)&H_cuda, width * height * sizeof(Complex_cuda));
        cudaMalloc((void **)&omega_inverse_cuda, width * sizeof(Complex_cuda));
        
		cout << "Start clokcing...\n";
        chrono::steady_clock::time_point tStart = chrono::steady_clock::now();
        
        int block_size = (height>512)?(512):(height);
        dim3 grid(Divup(height,block_size),1);    
        dim3 block(block_size,1);
        cudaMemcpy(H_cuda, H, width*height*sizeof(Complex_cuda), cudaMemcpyHostToDevice);
        cudaMemcpy(omega_inverse_cuda, omega_inverse, width*sizeof(Complex_cuda), cudaMemcpyHostToDevice);

        ifft<<<grid, block>>>(H_cuda, omega_inverse_cuda, width);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
		cudaMemcpy(h, H_cuda, width*height*sizeof(Complex_cuda), cudaMemcpyDeviceToHost);

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
        free(H); 
        free(h);
        cudaFree(H_cuda); 
	}

	cout << "finished...\n";
	return 0;
}