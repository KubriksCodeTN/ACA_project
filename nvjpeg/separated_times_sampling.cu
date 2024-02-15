/**
 * This file benchmarks the compression of one photo and return three separate values for respectively:
 * - the time to copy data from CPU to GPU
 * - compression time
 * - the time to copy data from GPU to CPU 
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

__global__ void warm_up_gpu(){
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;
  ib += ia + tid;
}

int main(){
    FILE* is = fopen("1.bmp", "rb");
    if (!is){
        std::cerr << "Error: failed to open input file\n";
        exit(1);
    }

    cudaFree(0);
    warm_up_gpu<<<1024, 512>>>();

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    nvjpegImage_t img;

    cudaEvent_t start[3], end[3];
    for(int i=0; i<3; ++i){
        cudaEventCreate(start+i);
        cudaEventCreate(end+i);
    }

    uint8_t header[54];
    fread(header, 54, 1, is);
    int w = *(int*)&header[18];
    int h = *(int*)&header[22];

    size_t sz = w * h * 3;
    uint8_t* ptr;
    uint8_t* ptr_dev;
    cudaMallocHost(&ptr, sz);
    for(int i = h - 1; i>=0; --i)
        fread(ptr+i*w*3, w*3, 1, is);


    cudaEventRecord(start[0], stream);
    cudaMallocAsync(&ptr_dev, sz, stream);
    cudaMemcpyAsync(ptr_dev, ptr, sz, cudaMemcpyHostToDevice, stream);
    cudaEventRecord(end[0], stream);
    cudaEventSynchronize(end[0]);

    img.channel[0] = ptr_dev;
    img.pitch[0] = w * 3;

    nvjpegHandle_t nv_handle;
    nvjpegEncoderState_t nv_enc_state;
    nvjpegEncoderParams_t nv_enc_params;

    nvjpegCreateSimple(&nv_handle);
    nvjpegEncoderStateCreate(nv_handle, &nv_enc_state, stream);
    nvjpegEncoderParamsCreate(nv_handle, &nv_enc_params, stream);
    nvjpegEncoderParamsSetSamplingFactors(nv_enc_params, NVJPEG_CSS_420, stream);

    cudaEventRecord(start[1], stream);
    nvjpegEncodeImage(nv_handle, nv_enc_state, nv_enc_params, &img, NVJPEG_INPUT_BGRI, w, h, stream);
    cudaEventRecord(end[1], stream);
    cudaEventSynchronize(end[1]);

    size_t length;

    cudaEventRecord(start[2], stream);
    nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, NULL, &length, stream);
    //cudaStreamSynchronize(stream);

    uint8_t* jpeg;
    cudaMallocHost(&jpeg, length);
    nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, jpeg, &length, 0);
    cudaEventRecord(end[2], stream);
    cudaEventSynchronize(end[2]);

    float time[3];
    cudaEventElapsedTime(time, start[0], end[0]);
    cudaEventElapsedTime(time + 1, start[1], end[1]);
    cudaEventElapsedTime(time + 2, start[2], end[2]);
    printf("Elapsed time host2device: %f ms\n", time[0]);
    printf("Elapsed time encoding: %f ms\n", time[1]);
    printf("Elapsed time device2host: %f ms\n", time[2]);
    printf("Elapsed time total: %f ms\n", time[0] + time[1] + time[2]);
#ifndef __NO_OUT
    std::cerr << length << '\n';
    FILE* out = fopen("out.jpg", "w+b");
    fwrite(jpeg, length, 1, out);
#endif
}

