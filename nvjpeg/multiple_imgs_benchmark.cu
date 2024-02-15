/**
 * This file benchmarks the compression time of a batch of images and outputs a csv file with the necessary data
 */
#include <charconv>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <functional>
#include <nvjpeg.h>
#include <stdio.h>
#include <string>
#include <sys/types.h>
#include <vector>
#include <assert.h>
#include <iostream>
#include <sys/mman.h>
#include <chrono>

/*
#define __PINNED_MEM
#define __DEBUG
#define __OUT
*/

// in case you want to measure time using std::chrono
// time is very similar to the results from cuda events
using namespace std::chrono;

// in case the compiler tries to remove something from the code to optimize
int main(int, char**) __attribute__((optimize(0)));

// encoding methods for the jpeg imgs
static const nvjpegJpegEncoding_t encoding[] = {
    NVJPEG_ENCODING_BASELINE_DCT, 
    // NVJPEG_ENCODING_EXTENDED_SEQUENTIAL_DCT_HUFFMAN, // this should have little difference compared to BASELINE_DCT
    // NVJPEG_ENCODING_LOSSLESS_HUFFMAN,
    // NVJPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN // this is not used as it doesn't fit the use case
};

// quality for the encoder range is from 1 to 100 we choose some sample values to try it out 
static const int quality[] = {100, 90, 80, 70};

// flag for optimized huffman encoding, if used the encoding is slower but size is also smaller
static const int optimized[] = {0, 1};

// various sampling factors that can be used for encoding
// grayscale is not used 
static const nvjpegChromaSubsampling_t sampling[] = {
    // NVJPEG_CSS_410V,
    NVJPEG_CSS_410,
    NVJPEG_CSS_411,
    NVJPEG_CSS_420,
    NVJPEG_CSS_422,
    NVJPEG_CSS_440,
    NVJPEG_CSS_444
    // NVJPEG_CSS_GRAY
};

// look up tables for paramter to string conversion (used to generate output file names)
static const char* str_encoding[] = {
    "baseline",
    // "extended",
    // "lossless"
};
static const char* str_optimized[] = {
    "Huffman",
    "optimizedHuffman"
};
static const char* str_sampling[] = {
    // "410V",
    "410",
    "411",
    "420",
    "422",
    "440",
    "444"
};

// macro for memory type selection
#if  defined(__PINNED_MEM)
static const char* mem_type = "pinned";
#elif defined(__MAPPED_MEM) /* zero copy */
static const char* mem_type = "mapped";
#else
#error "Please specify the type of memory to use: -D (__PINNED_MEM|__MAPPED_MEM");
#endif

// call CHECK only in debug mode, it might slow down program and impact the benchmark otherwise
#ifdef __DEBUG
// multi purpose function to check if an error code is != 0
#define CHECK(call)                                                                   \
{                                                                                     \
    const auto error = call;                                                          \
    if (error)                                                                        \
    {                                                                                 \
        fprintf(stderr, "Error: %i on file %s line %d, ", error, __FILE__, __LINE__); \
        exit(1);                                                                      \
    }                                                                                 \
}
// #define CHECK(call) assert(!call)

#else 
#define CHECK(call) (call)
#endif

__global__ void warm_up_gpu(){
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    ib += ia + tid; 
}

inline void error(const char* s){
    fprintf(stderr, "%s\n", s);
    exit(1);
}


int main(int argc, char** argv){
    // Usage: jpegEncoder: path/to/img1 path/to/img2...
    if (argc == 1)
        error("Usage: jpegEncoder path/to/img1 path/to/img2 ... path/to/imgn");

    // init cuda context and wam up the gpu
    cudaFree(0);
    warm_up_gpu<<<1024, 512>>>();

#if defined(__PINNED_MEM)
    const char* csv_file = "results_pinned_mem.csv";
#elif defined(__MAPPED_MEM)
    const char* csv_file = "results_mapped_mem.csv";
#endif
    // output is in csv format
    FILE* result = fopen(csv_file, "w+");
    if (!result)
         error("Cannot create benchmark output file");
    fprintf(result, "#imgs,mem_type,encoding,quality,optimizedHuffman,subsampling,time,img_size\n");

    /* 
     * NOTE: it might be good in case of multiple imgs to set a limit for how many imgs can be encoded in 
     * parallel, for example 8 imgs at a time. This strictly depends on the GPU in use and the amount of
     * memory that the system can use so it's not implemented
     */
    // resoures needed to load imgs in RAM
    size_t n_imgs = argc - 1;
    cudaStream_t* stream = (cudaStream_t*)malloc(sizeof(cudaEvent_t) * n_imgs);
    nvjpegImage_t* img = (nvjpegImage_t*)malloc(sizeof(nvjpegImage_t) * n_imgs);
    nvjpegEncoderState_t* nv_enc_state = (nvjpegEncoderState_t*)malloc(sizeof(nvjpegEncoderState_t) * n_imgs);
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
            error("Cannot open input file");
        fread(header, 54, 1, is);
        w[i - 1] = *(int*)&header[18];
        h[i - 1] = *(int*)&header[22];
        size_t sz = w[i - 1] * h[i - 1] * 3;
        CHECK(cudaMallocHost(&data[i - 1], sz));
        for (int j = h[i - 1] - 1; j >= 0; --j)
            fread(data[i - 1] + w[i - 1] * 3 * j, w[i - 1] * 3, 1, is);
        fclose(is);
    }

    for (int e = 0; e < sizeof(encoding) / sizeof(nvjpegJpegEncoding_t); ++e){
    for (int q = 0; q < sizeof(quality) / sizeof(int); ++q){
    for (int o = 0; o < sizeof(optimized) / sizeof(int); ++o){
    for (int s = 0; s < sizeof(sampling) / sizeof(nvjpegChromaSubsampling_t); ++s){
        // cuda events to measure time (ms)
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);

        // encoder parameters creation and setting (negligible amount of time so not measured)
        nvjpegHandle_t nv_handle;
        nvjpegEncoderParams_t nv_enc_params;
        
        nvjpegCreateSimple(&nv_handle);
        for (int i = 0; i < n_imgs; ++i)
            CHECK(nvjpegEncoderStateCreate(nv_handle, nv_enc_state + i, 0));
        CHECK(nvjpegEncoderParamsCreate(nv_handle, &nv_enc_params, 0));
        CHECK(nvjpegEncoderParamsSetEncoding(nv_enc_params, encoding[e], 0));
        CHECK(nvjpegEncoderParamsSetQuality(nv_enc_params, quality[q], 0));
        CHECK(nvjpegEncoderParamsSetOptimizedHuffman(nv_enc_params, optimized[o], 0));
        CHECK(nvjpegEncoderParamsSetSamplingFactors(nv_enc_params, sampling[s], 0));

        cudaStreamSynchronize(0); // wait for paramters

        // start measuring time and imgs encoding
        // encoding steps: load img on the device -> encode the img -> retrive encoded img on the host
        cudaEventRecord(start);
        for (int i = 0; i < n_imgs; ++i){
            size_t sz = w[i] * h[i] * 3;
            img[i].pitch[0] = w[i] * 3;
        #if defined(__PINNED_MEM)
            CHECK(cudaMallocAsync(img[i].channel, sz, stream[i]));
            CHECK(cudaMemcpyAsync(img[i].channel[0], data[i], sz, cudaMemcpyHostToDevice, stream[i]));
            CHECK(nvjpegEncodeImage(nv_handle, nv_enc_state[i], nv_enc_params, &img[i], NVJPEG_INPUT_BGRI, w[i], h[i], stream[i]));
        #elif defined(__MAPPED_MEM)
            img[i].channel[0] = data[i];
            CHECK(nvjpegEncodeImage(nv_handle, nv_enc_state[i], nv_enc_params, &img[i], NVJPEG_INPUT_BGRI, w[i], h[i], stream[i]));
        #endif
            CHECK(nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state[i], NULL, length + i, stream[i]));
        }
        // it is more efficient to call cudaMallocHost after starting the streams since cudaMallocHost can't be async
        // and also you need to finish the encoding to know the size needed to save the img in host RAM
        for (int i = 0; i < n_imgs; ++i){
            cudaStreamSynchronize(stream[i]);
            CHECK(cudaMallocHost(jpeg + i, length[i]));
            CHECK(nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state[i], jpeg[i], length + i, stream[i]));
        }
        cudaDeviceSynchronize(); // wait for the streams to complete
        cudaEventRecord(end);
        cudaEventSynchronize(end);

        float time;
        std::string size = std::to_string(w[0]) + " x " + std::to_string(h[0]);
        cudaEventElapsedTime(&time, start, end);
        fprintf(result, "%i,%s,%s,%i,%i,%s,%f,%s\n", argc - 1, mem_type, str_encoding[e], quality[q], optimized[o], str_sampling[s], time, size.c_str());

    #ifdef __OUT
        // if you want to save the output of the encoder define the macro __OUT
        /*
         * the output file name is outIdx_encoding_quality_optimized_subsampling_memory
         * where:
         *  - Idx is the idx of the image going from A to Z (if more it goes on after the Z in ASCII order)
         *  - Encoding is the encoding method used for the jpeg
         *  - Quality is the % of quality of the jpeg (the better the quality the bigger the size of the file)
         *  - Optimized is a flag that states if optimized Huffman encoding was used 
         *  - Subsampling is the chroma subsampling used for the jpeg 
         *  - Memory indicated if pinned memory or mapped memory (zero copy) was used
         */
        char* out_name;
        asprintf(&out_name, "outA_%s_%i_%s_%s_%s.jpg", str_encoding[e], quality[q], str_optimized[o], str_sampling[s], mem_type);
        for (int i = 0; i < n_imgs; ++i){
            out_name[3] = i + 0x41;
            FILE* out = fopen(out_name, "w+b");
            fwrite(jpeg[i], length[i], 1, out);
            fclose(out);
        }
        free(out_name);
    #endif
        
        // free resources
        cudaEventDestroy(start);
        cudaEventDestroy(end);
        for (int i = 0; i < n_imgs; ++i){
        #ifdef __PINNED_MEM
            cudaFree(img[i].channel[0]);
        #endif
            cudaFreeHost(jpeg[i]);
            nvjpegEncoderStateDestroy(nv_enc_state[i]);
        }
        nvjpegEncoderParamsDestroy(nv_enc_params);
        nvjpegDestroy(nv_handle);
    }
    }
    }
    }
    fclose(result);
}
