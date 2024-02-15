/**
 * This is a simple example use of the nvjpeg library to help understand the API
 */
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <nvjpeg.h>
#include <stdio.h>
#include <sys/types.h>
#include <vector>
#include <assert.h>
#include <iostream>
#include <sys/mman.h>

#define CHECK(call)                                                            \
{                                                                              \
    const auto error = call;                                                   \
    if (error != 0)                                                            \
    {                                                                          \
        fprintf(stderr, "Error: %i,%s:%d, ", error, __FILE__, __LINE__);       \
        exit(1);                                                               \
    }                                                                          \
}

/*
// this function is used only to wam up gpu before benchmarking
__global__ void warm_up_gpu(){
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;
  ib += ia + tid; 
}
*/ 

void error(const char* s){
    fprintf(stderr, "%s\n", s);
    exit(1);
}

int main(int argc, char** argv){
    // Usage: jpegEncoder: path/to/img1 path/to/img2...
    if (argc == 1)
        error("Usage: jpegEncoder path/to/img1 path/to/img2 ... path/to/imgn");

    // init cuda context
    cudaFree(0);
    // warm_up_gpu<<<1024, 512>>>();
    // arrray[n]

    // resoures needed to load imgs in RAM
    size_t n_imgs = argc - 1;
    cudaStream_t* stream = (cudaStream_t*)malloc(sizeof(cudaEvent_t) * n_imgs);
    nvjpegImage_t* img = (nvjpegImage_t*)malloc(sizeof(nvjpegImage_t) * n_imgs);
    size_t* length = (size_t*)malloc(sizeof(length) * n_imgs);
    uint8_t** data = (uint8_t**)malloc(sizeof(uint8_t*) * n_imgs);
    uint8_t** jpeg = (uint8_t**)malloc(sizeof(uint8_t*) * n_imgs);
    int* w = (int*)malloc(sizeof(int) * n_imgs);
    int* h = (int*)malloc(sizeof(int) * n_imgs);
    for (int i = 0; i < n_imgs; ++i)
        cudaStreamCreate(stream + i);

    // loading imgs in RAM
    uint8_t header[54]; // 54 is the usual size of a bmp header
    for (int i = 1; i < argc; ++i){
        FILE* is = fopen(argv[i], "rb");
        if (!is)
            error("Error: cannot open one of source files\n");
        fread(header, 54, 1, is);
        w[i - 1] = *(int*)&header[18];
        h[i - 1] = *(int*)&header[22];
        size_t sz = w[i - 1] * h[i - 1] * 3;
        cudaMallocHost(&data[i - 1], sz);
        for (int j = h[i - 1] - 1; j >= 0; --j)
            fread(data[i - 1] + w[i - 1] * 3 * j, w[i - 1] * 3, 1, is); // care about the order of rows
        fclose(is);
    }

    // encoder paramters setting 
    nvjpegHandle_t nv_handle;
    nvjpegEncoderState_t nv_enc_state;
    nvjpegEncoderParams_t nv_enc_params;
    
    nvjpegCreateSimple(&nv_handle);
    nvjpegEncoderStateCreate(nv_handle, &nv_enc_state, 0);
    nvjpegEncoderParamsCreate(nv_handle, &nv_enc_params, 0);
    nvjpegEncoderParamsSetQuality(nv_enc_params, 90, 0);
    nvjpegEncoderParamsSetSamplingFactors(nv_enc_params, NVJPEG_CSS_420, 0); // NEEDED! There's no default value
    cudaStreamSynchronize(0); // wait for paramters
    // start measuring time and imgs encoding
    // encoding steps: load img on the device -> encode the img -> retrive encoded img on the host
    for (int i = 0; i < n_imgs; ++i){
        size_t sz = w[i] * h[i] * 3;
        img[i].pitch[0] = w[i] * 3;
        cudaMallocAsync(img[i].channel, sz, stream[i]);
        cudaMemcpyAsync(img[i].channel[0], data[i], sz, cudaMemcpyHostToDevice, stream[i]);
        nvjpegEncodeImage(nv_handle, nv_enc_state, nv_enc_params, &img[i], NVJPEG_INPUT_BGRI, w[i], h[i], stream[i]);
    }
    for (int i = 0; i < n_imgs; ++i){
        nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, NULL, length + i, stream[i]);
        cudaMallocHost(jpeg + i, length[i]);
        nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, jpeg[i], length + i, stream[i]);
    }
    // wait for data
    cudaDeviceSynchronize();

// define __NO_OUT if you do not care about the output image being saved on disk
#ifndef __NO_OUT
    char out_name[] = "outA.jpg";
    for (int i = 0; i < n_imgs; ++i){
        out_name[3] = i + 'A'; // this is hacky
        FILE* out = fopen(out_name, "w+b");
        fwrite(jpeg[i], length[i], 1, out);
        fclose(out);
    }
#endif
}

