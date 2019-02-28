#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include "complex.cuh"
#include "input_image.cuh"
#include <cuda.h>
#include <sstream>
#include <chrono>
#include <ctime>

using namespace std;

#define threadPerBlk 256
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


template<typename T>
std::string toString(const T& value)
{
    std::ostringstream oss;
    oss << value;
    return oss.str();
}

__global__ void bitreverse(unsigned int N, Complex_cuda* h, Complex_cuda* h_new){
    

    int log2N = log2f(N * 1.0f);
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int x = index / N; // if N = 2048, each blc has 256 threads (blockDim.x) then 8 blks make a row and if blockIdx.x is the 19th blk then it should be at 3rd row.
    int y = index % N;

    unsigned int reversed = y;
    reversed = ((reversed >> 1) & 0x55555555) | ((reversed << 1) & 0xaaaaaaaa);
    reversed = ((reversed >> 2) & 0x33333333) | ((reversed << 2) & 0xcccccccc);
    reversed = ((reversed >> 4) & 0x0f0f0f0f) | ((reversed << 4) & 0xf0f0f0f0);
    reversed = ((reversed >> 8) & 0x00ff00ff) | ((reversed << 8) & 0xff00ff00);
    reversed = ((reversed >> 16) & 0x0000ffff) | ((reversed << 16) & 0xffff0000);
    reversed >>= (32 - log2N);

    h_new[x * N + reversed] = h[x * N + y];
}

__global__ void calculateOmega(unsigned int N, Complex_cuda* omega){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    omega[index] = Complex_cuda(cos(2.0f * M_PI * index / N), sin(-2.0f * M_PI * index / N));
}


__global__ void cudaFFT1D(unsigned int N, Complex_cuda* h, Complex_cuda* h_new, Complex_cuda* omega){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int x = index / N;
    int y = index % N;
    for(int l = 2; l <= N; l <<= 1){
        //create a shared mem from h?
        int m = l >> 1;
        int startPtr = y / l * l; // for 39th, if l = 4 then subarray starts at 36th.
        int subIndex = y % l; // 39 th is 3rd element of the subarray from 36 to 39.
        int pairIndexY = (subIndex + m) % l + startPtr; // 39 pair with 37
        int pairIndex = x * N + pairIndexY;
        Complex_cuda temp = omega[N / l * subIndex];
        if(subIndex >= m){//odd
            h_new[index] = h[pairIndex] + temp * h[index];
        }else{//even
            h_new[index] = h[index] + temp * h[pairIndex];
        }
        __syncthreads();
        Complex_cuda* swapTemp = h_new;
        h_new = h;
        h = swapTemp;
        __syncthreads();
    }
}


__global__ void mergeByLength(unsigned int N, Complex_cuda* h, Complex_cuda* h_new, Complex_cuda* omega, int l){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int x = index / N;
    int y = index % N;
    int m = l / 2;
    int startPtr = y / l * l; // for 39th, if l = 4 then subarray starts at 36th.
    int subIndex = y % l; // 39 th is 3rd element of the subarray from 36 to 39.
    int pairIndexY = (subIndex + m) % l + startPtr; // 39 pair with 37
    int pairIndex = x * N + pairIndexY;
    Complex_cuda temp = omega[N / l * subIndex];
    if(subIndex >= m){//odd
        h_new[index] = h[pairIndex] + temp * h[index];
    }else{//even
        h_new[index] = h[index] + temp * h[pairIndex];
    }
    __syncthreads();
    Complex_cuda* swapTemp = h_new;
    h_new = h;
    h = swapTemp;
    __syncthreads();
}

__global__ void mergeInBlk(unsigned int N, Complex_cuda* h, Complex_cuda* h_new, Complex_cuda* omega){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    for(int l = 2; l <= blockDim.x; l <<= 1){
        //create a shared mem from h?
        int m = l >> 1;
        int startPtr = threadIdx.x / l * l; // for 39th, if l = 4 then subarray starts at 36th.
        int subIndex = threadIdx.x % l; // 39 th is 3rd element of the subarray from 36 to 39.
        int pairIndex = (subIndex + m) % l + startPtr + blockDim.x * blockIdx.x; // 39 pair with 37
        Complex_cuda temp = omega[N / l * subIndex];
        if(subIndex >= m){//odd
            h_new[index] = h[pairIndex] + temp * h[index];
        }else{//even
            h_new[index] = h[index] + temp * h[pairIndex];
        }
        __syncthreads();
        Complex_cuda* swapTemp = h_new;
        h_new = h;
        h = swapTemp;
        __syncthreads();
    }
}

__global__ void mergeBetweenBlk(unsigned int N, Complex_cuda* h, Complex_cuda* h_new, Complex_cuda* omega){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int x = index / N;
    int blkPerRow = N / blockDim.x;
    int blockIdxInRow = blockIdx.x % blkPerRow;
    for(int gap = 2; gap <= blkPerRow; gap <<= 1){
        int l = gap * blockDim.x;
        int m = l >> 1;
        int startPtr = blockIdxInRow / gap * gap;
        int subBlkIndex = blockIdxInRow % gap;
        int subIndex = subBlkIndex * blockDim.x + threadIdx.x;
        int pairBlkIndex = (subBlkIndex + gap / 2) % gap + startPtr;
        int pairIndex = pairBlkIndex * blockDim.x + threadIdx.x + x * N;
        Complex_cuda temp = omega[N / l * subIndex];
        // if(index == 1){
        //     printf("l is %d\n", l);
        //     printf("m is %d\n", m);
        //     printf("startPtr is %d\n", startPtr);
        //     printf("subIndex is %d\n", subIndex);
        //     printf("subBlkIndex is %d\n", subBlkIndex);
        //     printf("pairBlkIndex is %d\n", pairBlkIndex);
        //     printf("pairIndex is %d\n", pairIndex);

        //     printf("before h[%d] is %12f %f\n", pairIndex, h[pairIndex].real, h[pairIndex].imag);
        //     printf("omega index is %d\n", N / l * subIndex);
        //     printf("omega.real is %f\n", temp.real);
        //     printf("omega.imag is %f\n", temp.imag);
        //     printf("before h[%d] is %12f %f\n", index, h[index].real, h[index].imag);
        // }
        if(subIndex >= m){
            h_new[index] = h[pairIndex] + temp * h[index];
        }else{
            h_new[index] = h[index] + temp * h[pairIndex];
        }
        __syncthreads();
        Complex_cuda* swapTemp = h_new;
        h_new = h;
        h = swapTemp;
        __syncthreads();
        // if(index == 1){
        //     printf("after h[%d] is %12f %f\n", pairIndex, h[pairIndex].real, h[pairIndex].imag);
        //     printf("after h[%d] is %12f %f\n", index, h[index].real, h[index].imag);
        // }
    }
}
     
void FFT1D(unsigned int N, Complex_cuda* h, int test){
    Complex_cuda* omega = new Complex_cuda[N];

    for(int i = 0; i < N; i++){
        omega[i] = Complex_cuda(cos(2.0f * M_PI * i / N), sin(-2.0f * M_PI * i / N));
    }

    int log2N = std::   log2(N * 1.0f);
    //sort array by reversed binary bits
    for(unsigned int i = 1; i < N; i++){
        unsigned int reversed = i;
        reversed = ((reversed >> 1) & 0x55555555) | ((reversed << 1) & 0xaaaaaaaa);
        reversed = ((reversed >> 2) & 0x33333333) | ((reversed << 2) & 0xcccccccc);
        reversed = ((reversed >> 4) & 0x0f0f0f0f) | ((reversed << 4) & 0xf0f0f0f0);
        reversed = ((reversed >> 8) & 0x00ff00ff) | ((reversed << 8) & 0xff00ff00);
        reversed = ((reversed >> 16) & 0x0000ffff) | ((reversed << 16) & 0xffff0000);

        reversed >>= (32 - log2N);
        if(i > reversed)  std::swap (h[i], h[reversed]);
    }
    for(int l = 2; l <= test; l <<= 1){
        int m = l / 2;
        for(int i = 0; i < N; i += l){
            for(int j = 0; j < m; j++){
                Complex_cuda temp = omega[N / l * j] * h[i + m + j];
                // if(i + j == 1 && m == 256){
                //     printf("temp is %f, %f\n", temp.real, temp.imag);
                //     printf("omega is %f, %f\n", omega[N / l * j].real, omega[N / l * j].imag);
                //     printf("h[%d] is %f, %f\n",i + j, h[i + j].real, h[i + j].imag);
                //     printf("h[%d] is %f, %f\n",i + m + j, h[i + m + j].real, h[i + m + j].imag);
                // }
                h[i + m + j] = h[i + j] - temp;
                h[i + j] = h[i + j] + temp;
                // if(i + j == 1 && m == 256){
                //     printf("h[%d] is %f, %f\n",i + j, h[i + j].real, h[i + j].imag);
                //     printf("h[%d] is %f, %f\n",i + m + j, h[i + m + j].real, h[i + m + j].imag);
                // }
            }
        }
    }
     
}
__global__ void mergeInBlkforTest(unsigned int N, Complex_cuda* h, Complex_cuda* h_new, Complex_cuda* omega){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    for(int l = 2; l <= blockDim.x; l <<= 1){
        //create a shared mem from h?
        int m = l >> 1;
        int startPtr = threadIdx.x / l * l; // for 39th, if l = 4 then subarray starts at 36th.
        int subIndex = threadIdx.x % l; // 39 th is 3rd element of the subarray from 36 to 39.
        int pairIndex = (subIndex + m) % l + startPtr + blockDim.x * blockIdx.x; // 39 pair with 37
        Complex_cuda temp = omega[N / l * subIndex];
        //Complex_cuda temp = Complex_cuda(cos(2.0f * M_PI * subIndex / l), sin(-2.0f * M_PI * subIndex / l));
        // if(index == 1){
        //     // printf("before h[%d] is %12f %f\n", pairIndex, h[pairIndex].real, h[pairIndex].imag);
        //     // printf("omega index is %d\n", N / l * subIndex);
        //     // printf("omega.real is %x\n", *(unsigned int*)&temp.real);
        //     // printf("omega.imag is %x\n", *(unsigned int*)&temp.imag);
        //     // printf("before h[%d] is %12f %f\n", index, h[index].real, h[index].imag);
        //     // printf("omega index is %d\n", N / l * subIndex);
        //     // printf("omega.real is %f\n", temp.real);
        //     // printf("omega.imag is %f\n", temp.imag);
        //     // printf("omega.real * h.real is %12f\n", (temp.real * h[index].real));
        //     // printf("omega.imag * h.imag is %12f\n", (temp.imag * h[index].imag));
        //     // printf("omega.real * h.imag is %12f\n", (temp.real * h[index].imag));
        //     // printf("omega.imag * h.real is %12f\n", (temp.imag * h[index].real));

        //     // printf("omega * h.real is %12f\n", (temp * h[index]).real);
        //     // printf("omega * h.imag is %12f\n", (temp * h[index]).imag);
        //     printf("pairIndex is %d\n", pairIndex);

        // }
        if(subIndex >= m){//odd
            h_new[index] = h[pairIndex] + temp * h[index];
        }else{//even
            h_new[index] = h[index] + temp * h[pairIndex];
        }
        __syncthreads();
        Complex_cuda* swapTemp = h_new;
        h_new = h;
        h = swapTemp;
        // if(index == 1){
        //     //printf("after h[%d] is %12f %f\n", pairIndex, h[pairIndex].real, h[pairIndex].imag);
        //     // printf("omega index is %d\n", N / l * subIndex);
        //     // printf("omega.real is %x\n", *(unsigned int*)&temp.real);
        //     // printf("omega.imag is %x\n", *(unsigned int*)&temp.imag);
        //     // printf("after h[%d] is %12f %f\n", index, h[index].real, h[index].imag);
        // }
    }
}

__global__ void simple(unsigned int N, Complex_cuda* h, Complex_cuda* h_new, Complex_cuda* omega){

    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int x = index / N;
    int y = index % N;

    Complex_cuda* startPtr = &h[x * N];

    //shared mem
    //extern __shared__ Complex_cuda s[];
    // int readPerThread = N / blockDim.x;
    // for(int i = 0; i < readPerThread; i++){
    //     s[threadIdx.x * readPerThread + i] = startPtr[threadIdx.x * readPerThread + i];
    // }
    // __syncthreads();
    // Complex_cuda res = Complex_cuda();
    // for(int i = 0; i < N; i++){
    //     res = res + omega[(y * i) % N] * s[i];
    // }

    //global mem
    Complex_cuda res = Complex_cuda();
    for(int i = 0; i < N; i++){
        res = res + omega[(y * i) % N] * startPtr[i];
    }

    h_new[y * N + x] = res;
}


__global__ void transpose(unsigned int N, Complex_cuda* h, Complex_cuda* h_new){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int x = index / N;
    int y = index % N;
    h_new[y * N + x] = h[index];
}





int main(int argc, char* argv[]) {

    if(argc < 4){
        printf("not enough arguments!");
        return -1;
    }

    int forward = 1;
    printf("%s\n", argv[1]);
    char* temp = argv[1];
    if(strcmp(argv[1], "forward") == 0){
        forward = 1;
    }else if(strcmp(argv[1], "reverse")){
        forward = 0;
    }else{
        printf("wrong parameter provided!\n");
        return -1;
    }
    char* inputfile = argv[2];
    char* outputfile = argv[3];

    cudaEvent_t start,stop;
    float ms;

    InputImage input(inputfile);
    InputImage input2(inputfile);
    Complex_cuda* h = input.get_image_data();


    int N = input.get_width();
    int height = input.get_height();
    int size = N * height;
    Complex_cuda* h2 = new Complex_cuda[size];

    //Complex_cuda* omega = new Complex_cuda[N];

    Complex_cuda* d_h_1;
    Complex_cuda* d_h_2;
    Complex_cuda* d_omega;

    gpuErrchk(cudaMalloc(&d_h_1, sizeof(Complex_cuda) * size));
    gpuErrchk(cudaMalloc(&d_h_2, sizeof(Complex_cuda) * size));
    gpuErrchk(cudaMalloc(&d_omega, sizeof(Complex_cuda) * N));

    gpuErrchk(cudaMemcpy(d_h_1, h, sizeof(Complex_cuda) * size, cudaMemcpyHostToDevice));
    

    printf("Start!\n");
    chrono::steady_clock::time_point tStart = chrono::steady_clock::now();
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start, 0));
    
    calculateOmega<<<N / 128, 128>>>(N, d_omega);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // FT regular method
    // simple<<<N * height / threadPerBlk, threadPerBlk, N * sizeof(Complex_cuda)>>>(N, d_h_1, d_h_2, d_omega);//1D
    // simple<<<N * height / threadPerBlk, threadPerBlk, N * sizeof(Complex_cuda)>>>(N, d_h_2, d_h_1, d_omega);//2D
    // cudaMemcpy(h, d_h_1, sizeof(Complex_cuda) * size, cudaMemcpyDeviceToHost);

    //FFT method
    bitreverse<<<N * height / threadPerBlk, threadPerBlk>>>(N, d_h_1, d_h_2);
    
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    int step = log2(N * 1.0);
    printf("%f\n", step);
    int length = 1;
    for(int i = 0; i < step / 2; i += 1){
        length *= 2;
        mergeByLength<<<N * height / threadPerBlk, threadPerBlk>>>(N, d_h_2, d_h_1, d_omega, length);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        length *= 2;
        mergeByLength<<<N * height / threadPerBlk, threadPerBlk>>>(N, d_h_1, d_h_2, d_omega, length);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }
    if(step % 2 == 1){
        length *= 2;
        mergeByLength<<<N * height / threadPerBlk, threadPerBlk>>>(N, d_h_2, d_h_1, d_omega, length);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        std::swap (d_h_1, d_h_2);
    }
    gpuErrchk(cudaMemcpy(h, d_h_2, sizeof(Complex_cuda) * size, cudaMemcpyDeviceToHost));

    transpose<<<N * height / threadPerBlk, threadPerBlk>>>(N, d_h_2, d_h_1);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    bitreverse<<<N * height / threadPerBlk, threadPerBlk>>>(N, d_h_1, d_h_2);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());


    length = 1;
    for(int i = 0; i < step / 2; i += 1){
        length *= 2;
        mergeByLength<<<N * height / threadPerBlk, threadPerBlk>>>(N, d_h_2, d_h_1, d_omega, length);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        length *= 2;
        mergeByLength<<<N * height / threadPerBlk, threadPerBlk>>>(N, d_h_1, d_h_2, d_omega, length);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }
    if(step % 2 == 1){
        length *= 2;
        mergeByLength<<<N * height / threadPerBlk, threadPerBlk>>>(N, d_h_2, d_h_1, d_omega, length);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        std::swap (d_h_1, d_h_2);
    }

    transpose<<<N * height / threadPerBlk, threadPerBlk>>>(N, d_h_2, d_h_1);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(h2, d_h_1, sizeof(Complex_cuda) * size, cudaMemcpyDeviceToHost));


    // cudaFFT1D<<<N * height / threadPerBlk, threadPerBlk>>>(N, d_h_2, d_h_1, d_omega);
    // gpuErrchk(cudaPeekAtLastError());
    // gpuErrchk(cudaDeviceSynchronize());


    // mergeInBlkforTest<<<N * height / threadPerBlk, threadPerBlk>>>(N, d_h_2, d_h_1, d_omega);
    // gpuErrchk(cudaPeekAtLastError());
    // gpuErrchk(cudaDeviceSynchronize());

    // mergeBetweenBlk<<<N * height / threadPerBlk, threadPerBlk>>>(N, d_h_2, d_h_1, d_omega);
    // gpuErrchk(cudaPeekAtLastError());
    // gpuErrchk(cudaDeviceSynchronize());

    // transpose<<<N * height / threadPerBlk, threadPerBlk>>>(N, d_h_2, d_h_1);
    // bitreverse<<<N * height / threadPerBlk, threadPerBlk>>>(N, d_h_1, d_h_2);
    // mergeInBlk<<<N * height / threadPerBlk, threadPerBlk>>>(N, d_h_2, d_h_1, d_omega);
    // transpose<<<N * height / threadPerBlk, threadPerBlk>>>(N, d_h_2, d_h_1);


    gpuErrchk(cudaEventRecord(stop, 0));
    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&ms, start, stop));
    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));
    printf("test%d GPU: %f ms \n",N, ms);
    gpuErrchk(cudaFree(d_h_1));
    gpuErrchk(cudaFree(d_h_2));
    gpuErrchk(cudaFree(d_omega));
    chrono::steady_clock::time_point tEnd = chrono::steady_clock::now();
    chrono::duration<double> time_span = chrono::duration_cast<chrono::duration<double>>(tEnd - tStart);

    // string filesuffix = toString(N);
    // string outputfilename ="result" +  filesuffix + ".txt";
    input.save_image_data(outputfile, h2, N, height);


    free(h);
    free(h2);
}