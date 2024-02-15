/*
* Copyright (c) 2018-2023 NVIDIA Corporation
*
* Permission is hereby granted, free of charge, to any person
* obtaining a copy of this software and associated documentation
* files (the "Software"), to deal in the Software without
* restriction, including without limitation the rights to use,
* copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the software, and to permit persons to whom the
* software is furnished to do so, subject to the following
* conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
* OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
* NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
* HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
* WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
* OTHER DEALINGS IN THE SOFTWARE.
*/


#include <fstream>
#include <iostream>
#include <memory>
#include <cuda.h>
#include <sstream>
#include <iterator>
#include "NvOFCuda.h"
#include "NvOFDataLoader.h"
#include "NvOFUtils.h"
#include "NvOFCmdParser.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// #define __DEBUG__

#ifdef __DEBUG__
#define LOG(format, ...) fprintf (stderr, format, __VA_ARGS__)
#else
#define LOG(format, ...) ;
#endif

// assuming that img contains an image in format r8g8b8a8 (read by stb_image), 
// it converts it into a8b8g8r8 (ABGR) used by OF
uint8_t* rgba_to_abgr(uint8_t* img, int width, int height){
    for(int i = 0; i < width * height; i += 4){
        std::swap(img[i], img[i+3]);
        std::swap(img[i+1], img[i+2]);
    }

    return img;
}


void estimate_flow(
    std::vector<std::string> input_file_names, 
    std::string output_file_name, 
    int grid_size,
    bool visual_flow,
    NV_OF_PERF_LEVEL perfPreset,
    CUcontext cu_context
){
    // var to handle images with stbi library
    int width;
    int height;
    int channels; // number of channels in the image (gray:1 - gray + alpha:2 - rgb:3 - rgba:4)
    uint8_t* frame[2]{0}; // ptr to the pair of frames currently used to calculate OF
    
    frame[0] = stbi_load(input_file_names[0].c_str(), &width, &height, &channels, 4); // 4 asks the library to always return rgba matrix
    if(!frame[0]){
        std::cerr << "Error with: " << input_file_names[0] << "\n";
        exit(1);
    }

    LOG("width: %d, height: %d\n", width, height);
    
    // from stbi library description:
    // An output image with N components has the following components interleaved
    // in this order in each pixel:
    //
    //     N=#comp     components
    //        ... unused ...
    //       4           red, green, blue, alpha

    // I need to swap components becouse OF needs ABGR
    rgba_to_abgr(frame[0], width, height);

    NV_OF_CUDA_BUFFER_TYPE inputBufferTypeEnum = NV_OF_CUDA_BUFFER_TYPE_CUARRAY;
    NV_OF_CUDA_BUFFER_TYPE outputBufferTypeEnum = NV_OF_CUDA_BUFFER_TYPE_CUARRAY;
    NV_OF_BUFFER_FORMAT bufferFormat = NV_OF_BUFFER_FORMAT_ABGR8;

    // create OF handler
    NvOFObj nvOpticalFlow = NvOFCuda::Create(
        cu_context, 
        width, 
        height, 
        bufferFormat,
        inputBufferTypeEnum, 
        outputBufferTypeEnum, 
        NV_OF_MODE_OPTICALFLOW, 
        perfPreset
    );


    // in Turing architecture hwGridSize can only be 4, 
    // so if I want a smaller grid, I need to upsample the result
    uint32_t hwGridSize = grid_size;
    uint32_t nScaleFactor = 1;
    if (!nvOpticalFlow->CheckGridSize(grid_size)){

        if (!nvOpticalFlow->GetNextMinGridSize(grid_size, hwGridSize))
            throw std::runtime_error("Invalid parameter");
        
        else
            nScaleFactor = hwGridSize / grid_size;
    }    

    nvOpticalFlow->Init(hwGridSize);

    // buffers to move img data from/to GPU, used by OF execute() function
    std::vector<NvOFBufferObj> inputBuffers = nvOpticalFlow->CreateBuffers(NV_OF_BUFFER_USAGE_INPUT, 2);
    std::vector<NvOFBufferObj> outputBuffers = nvOpticalFlow->CreateBuffers(NV_OF_BUFFER_USAGE_OUTPUT, 1);
    
    // used to upsample the result to the image resolution if needed
    std::unique_ptr<NvOFUtils> nvOFUtils(new NvOFUtilsCuda(NV_OF_MODE_OPTICALFLOW));
    std::vector<NvOFBufferObj> upsampleBuffers;
    // array to store the outpt of OF
    std::unique_ptr<NV_OF_FLOW_VECTOR[]> pOut(nullptr);
    
    int nOutWidth = (width + grid_size - 1) / grid_size;
    int nOutHeight = (height + grid_size - 1) / grid_size;
    int nOutSize = nOutWidth * nOutHeight;

    upsampleBuffers = nvOpticalFlow->CreateBuffers(nOutWidth, nOutHeight, NV_OF_BUFFER_USAGE_OUTPUT, 1);
    pOut.reset(new NV_OF_FLOW_VECTOR[nOutSize]);
    if (pOut == nullptr)    {
        std::ostringstream err;
        err << "Failed to allocate output host memory of size " << nOutSize * sizeof(NV_OF_FLOW_VECTOR) << " bytes" << std::endl;
        throw std::bad_alloc();
    }

    // utility to write .flo file
    std::unique_ptr<NvOFFileWriter> flowFileWriter = NvOFFileWriter::Create(
        nOutWidth,
        nOutHeight,
        NV_OF_MODE_OPTICALFLOW
    );
    
    LOG("outputbuf dim: %d\n", nOutSize);

    // send frame_0 to GPU and start to iterate through the frames
    inputBuffers[0]->UploadData(frame[0]); 
    LOG("loading image: %s\n", input_file_names[0]);
    
    for (int i = 1; i < input_file_names.size(); i++){

        LOG("loading image: %s\n", input_file_names[i]);
    
        frame[1] = stbi_load(input_file_names[i].c_str(), &width, &height, &channels, 4);
        if(!frame[1]){
            std::cerr << "Error with: " << input_file_names[1] << "\n";
            exit(1);
        }
        rgba_to_abgr(frame[1], width, height);

        // send second frame to GPU
        inputBuffers[1]->UploadData(frame[1]);
        
        // calulate OF between frame[0] and frame[1]
        nvOpticalFlow->Execute(inputBuffers[0].get(),
            inputBuffers[1].get(),
            outputBuffers[0].get()
        );

        // do upsample if needed
        if (nScaleFactor > 1){
            nvOFUtils->Upsample(outputBuffers[0].get(), upsampleBuffers[0].get(), nScaleFactor);
            upsampleBuffers[0]->DownloadData(pOut.get());
        }
        else
            outputBuffers[0]->DownloadData(pOut.get());
        
        // save .flo file
        flowFileWriter->SaveOutput((void*)pOut.get(), output_file_name, i, visual_flow);    

        // swap frames and free the first one
        inputBuffers[1].swap(inputBuffers[0]);
        std::swap(frame[0], frame[1]);
        stbi_image_free(frame[1]);
        frame[1] = nullptr;     
    }

    stbi_image_free(frame[0]);
}

int main(int argc, char **argv){

    if(argc < 6){
        std::cerr << "usage: ./AppOFCuda_minimal frame_0 frame_1 [... frame_n] flo_filename grid(1 - 2 - 4) preset(slow - medium - fast)\n";
        exit(1);
    }


    std::vector<std::string> input_file_names;
    for(int i = 1; i < argc - 3; ++i)
        input_file_names.push_back(argv[i]);

    std::string output_file_name = argv[argc - 3];
    int grid_size = std::atoi(argv[argc - 2]); // 1, 2 or 4
    bool visual_flow = false;
    int gpuId = 0;
    std::string preset = argv[argc - 1];
    NV_OF_PERF_LEVEL preset_enum = (preset == "fast" ? NV_OF_PERF_LEVEL_FAST : (preset == "medium") ? NV_OF_PERF_LEVEL_MEDIUM : NV_OF_PERF_LEVEL_SLOW);
    CUcontext cu_context = nullptr;
    CUdevice cuDevice = 0;

    // init cuda context, required by OF API
    CUDA_DRVAPI_CALL(cuInit(0));
    CUDA_DRVAPI_CALL(cuDeviceGet(&cuDevice, gpuId));
    CUDA_DRVAPI_CALL(cuCtxCreate(&cu_context, 0, cuDevice));       
#ifdef __DEBUG__
    char szDeviceName[80];
    CUDA_DRVAPI_CALL(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
    std::cout << "GPU in use: " << szDeviceName << std::endl;
#endif
    

    estimate_flow(input_file_names, output_file_name, grid_size, visual_flow, preset_enum, cu_context);


    CUDA_DRVAPI_CALL(cuCtxDestroy(cu_context));
 
    return 0;
}
