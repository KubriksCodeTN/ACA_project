# High-Performance Image Processing with NVJPEG and NVOF

This repository contains the analysis and benchmarking of NVIDIA's hardware-accelerated libraries, developed in collaboration with **CovisionLab (Bolzano, Italy)**. The project focuses on optimizing image compression and optical flow estimation using GPU-based pipelines.  A complete review and analysis of the results can be seen in the file report.pdf

## Project Overview

The study evaluates the performance and trade-offs of two key NVIDIA SDKs:
1. **NVJPEG**: For high-throughput image encoding.
2. **NVOF (NVIDIA Optical Flow)**: For hardware-accelerated motion estimation.

## NVJPEG Analysis & Benchmarking

The goal was to build an efficient compression pipeline, analyzing how different parameters affect execution time and image quality.

* **Parallel Processing**: Evaluated batch sizes (1, 4, 8, 16) to maximize GPU occupancy.
* **Compression Tuning**: Systematic testing of **Chroma Subsampling** (4:4:4 to 4:1:0) and **Quality factors** (70–100).
* **Quality Assessment**: Used **SSIM (Structural Similarity Index Measure)** to quantify information loss.
* **Key Insight**: Identified the "sweet spot" at Quality 90 (4:4:4) for optimal fidelity-to-performance ratio.

## NVOF (Optical Flow) Exploration

Analysis of the hardware-based Optical Flow engine on the **NVIDIA Turing architecture**.

* **Preset Comparison**: Benchmarked `Slow`, `Medium`, and `Fast` presets to evaluate accuracy vs. latency.
* **Performance Metrics**: Measured upload/download latencies and execution times on a **Quadro RTX 6000**.
* **Observations**: Analyzed how the "Slow" preset significantly reduces artifacts in monochromatic or low-texture areas compared to faster configurations.

