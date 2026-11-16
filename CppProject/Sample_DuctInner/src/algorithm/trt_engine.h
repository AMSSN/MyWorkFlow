#pragma once
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <io.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <functional>

using namespace nvinfer1;
using namespace nvonnxparser;

#define DEVICE 0

#define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char *op, const char *file, int line);

template <typename _T>
std::shared_ptr<_T> make_nvshared(_T *ptr)
{
    return std::shared_ptr<_T>(ptr, [](_T *p) { p->destroy(); });
}

inline const char *severity_string(nvinfer1::ILogger::Severity t)
{
    switch (t)
    {
    case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
        return "internal_error";
    case nvinfer1::ILogger::Severity::kERROR:
        return "error";
    case nvinfer1::ILogger::Severity::kWARNING:
        return "warning";
    case nvinfer1::ILogger::Severity::kINFO:
        return "info";
    case nvinfer1::ILogger::Severity::kVERBOSE:
        return "verbose";
    default:
        return "unknow";
    }
}

class TRTLogger : public nvinfer1::ILogger
{
public:
    virtual void log(Severity severity, const char *msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
        {
            if (severity == Severity::kWARNING)
            {
                printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else if (severity <= Severity::kERROR)
            {
                printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else
            {
                printf("%s: %s\n", severity_string(severity), msg);
            }
        }
    }
};

int onnx2engine(std::string onnx_filename, std::string enginefilePath, int batchsize, int imagesize, int type);

int generate_engine_model(std::string &engine_file, int batch_size, int infer_size, int mode);
