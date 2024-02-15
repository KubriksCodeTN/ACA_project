/**
 * This file is a simplified version of multiple_images_benchmark as it benchmarks time for only one image
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
#error "Please specify the type of memory to use";
#endif

// call CHECK only in debug mode, it might slow down program and impact the benchmark otherwise
#ifdef __DEBUG
// multi purpose function to check if an error code is != 0
/*
#define CHECK(call)                                                                   \
{                                                                                     \
    const auto error = call;                                                          \
    if (error)                                                                        \
    {                                                                                 \
        fprintf(stderr, "Error: %i on file %s line %d, ", error, __FILE__, __LINE__); \
        exit(1);                                                                      \
    }                                                                                 \
}
*/
#define CHECK(call) assert(!call)

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
    // Usage: jpegEncoder: path/to/img1
    if (argc != 2)
        error("Usage: jpegEncoder path/to/img");

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
    fprintf(result, "mem_type,encoding,quality,optimizedHuffman,subsampling,original_size(MiB), compressed_size(MiB)\n");

    /* 
     * NOTE: it might be good in case of multiple imgs to set a limit for how many imgs can be encoded in 
     * parallel, for example 8 imgs at a time. This strictly depends on the GPU in use and the amount of
     * memory that the system can use so it's not implemented
     */
    // resoures needed to load imgs in RAM
    cudaStream_t stream;
    nvjpegImage_t img;
    nvjpegEncoderState_t nv_enc_state;
    size_t length;
    uint8_t* data;
    uint8_t* jpeg;
    int w; 
    int h;
    cudaStreamCreate(&stream);

    // loading imgs in RAM
    uint8_t header[54]; // 54 is the usual size of a bmp header
    FILE* is = fopen(argv[1], "rb");
    if (!is)
        error("Cannot open input file");
    fread(header, 54, 1, is);
    w = *(int*)&header[18];
    h = *(int*)&header[22];
    size_t sz = w * h * 3;
    CHECK(cudaMallocHost(&data, sz));
    for (int j = h - 1; j >= 0; --j)
        fread(data + w * 3 * j, w * 3, 1, is);
    fclose(is);

    for (int e = 0; e < sizeof(encoding) / sizeof(nvjpegJpegEncoding_t); ++e){
    for (int q = 0; q < sizeof(quality) / sizeof(int); ++q){
    for (int o = 0; o < sizeof(optimized) / sizeof(int); ++o){
    for (int s = 0; s < sizeof(sampling) / sizeof(nvjpegChromaSubsampling_t); ++s){
        // cuda events to measure time (ms)
        cudaEvent_t start[3], end[3];
        float time[3];
        for(int i = 0; i < 3; ++i){
            cudaEventCreate(start + i);
            cudaEventCreate(end + i);
        }

        // encoder parameters creation and setting (negligible amount of time so not measured)
        nvjpegHandle_t nv_handle;
        nvjpegEncoderParams_t nv_enc_params;
        
        nvjpegCreateSimple(&nv_handle);
        CHECK(nvjpegEncoderStateCreate(nv_handle, &nv_enc_state, 0));
        CHECK(nvjpegEncoderParamsCreate(nv_handle, &nv_enc_params, 0));
        CHECK(nvjpegEncoderParamsSetEncoding(nv_enc_params, encoding[e], 0));
        CHECK(nvjpegEncoderParamsSetQuality(nv_enc_params, quality[q], 0));
        CHECK(nvjpegEncoderParamsSetOptimizedHuffman(nv_enc_params, optimized[0], 0));
        CHECK(nvjpegEncoderParamsSetSamplingFactors(nv_enc_params, sampling[s], 0));

        cudaStreamSynchronize(0); // wait for paramters

        // start measuring time and imgs encoding
        // encoding steps: load img on the device -> encode the img -> retrive encoded img on the host
        size_t sz = w * h * 3;
        img.pitch[0] = w * 3;
    #if defined(__PINNED_MEM)
        cudaEventRecord(start[0], stream);
        CHECK(cudaMallocAsync(img.channel, sz, stream));
        CHECK(cudaMemcpyAsync(img.channel[0], data, sz, cudaMemcpyHostToDevice, stream));
        cudaEventRecord(end[0], stream);
        cudaEventSynchronize(end[0]);

        cudaEventRecord(start[1], stream);
        CHECK(nvjpegEncodeImage(nv_handle, nv_enc_state, nv_enc_params, &img, NVJPEG_INPUT_BGRI, w, h, stream));
        cudaEventRecord(end[1], stream);
        cudaEventSynchronize(end[1]);
    #elif defined(__MAPPED_MEM)
        // cudaMallocHostManaged
        cudaEventRecord(start[0], stream);
        img.channel[0] = data;
        cudaEventRecord(end[0], stream);
        cudaEventSynchronize(end[0]);
        cudaEventRecord(start[1], stream);
        CHECK(nvjpegEncodeImage(nv_handle, nv_enc_state, nv_enc_params, &img, NVJPEG_INPUT_BGRI, w, h, stream));
        cudaEventRecord(end[1], stream);
        cudaEventSynchronize(end[1]);
    #endif
        cudaEventRecord(start[2], stream);
        CHECK(nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, NULL, &length, stream));
        // it is more efficient to call cudaMallocHost after starting the streams since cudaMallocHost can't be async
        // and also you need to finish the encoding to know the size needed to save the img in host RAM
        cudaStreamSynchronize(stream);
        CHECK(cudaMallocHost(&jpeg, length));
        CHECK(nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, jpeg, &length, stream));
        cudaEventRecord(end[2], stream);
        cudaEventSynchronize(end[2]);

        cudaEventElapsedTime(time, start[0], end[0]);
        cudaEventElapsedTime(time+1, start[1], end[1]);
        cudaEventElapsedTime(time+2, start[2], end[2]);
        fprintf(result, "%s,%i,%s,%.2f,%.2f\n", 
            str_encoding[e], quality[q], str_sampling[s], sz / 1024. / 1024., length / 1024. / 1024.);

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
        asprintf(&out_name, "out_%s_%i_%s_%s_%s.jpg", str_encoding[e], quality[q], str_optimized[0], str_sampling[s], mem_type);
        FILE* out = fopen(out_name, "w+b");
        fwrite(jpeg, length, 1, out);
        fclose(out);
        free(out_name);
    #endif
        
        // free resources
        for (int i = 0; i < 3; ++i){
            cudaEventDestroy(start[i]);
            cudaEventDestroy(end[i]);
        }
    #ifdef __PINNED_MEM
        cudaFree(img.channel[0]);
    #endif
        cudaFreeHost(jpeg);
        nvjpegEncoderStateDestroy(nv_enc_state);
        nvjpegEncoderParamsDestroy(nv_enc_params);
        nvjpegDestroy(nv_handle);
    }
    }
    }
    }
    fclose(result);
}
