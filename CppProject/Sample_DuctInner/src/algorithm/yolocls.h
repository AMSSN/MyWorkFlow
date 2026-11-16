#pragma once
#include "trt_engine.h"
#include "../base/utils.h"

struct YOLOClsTRTContext
{
    float *input_data_host;
    float *input_data_device;
    float *output_data_device;
    float *output_data_host;

    int input_batch;
    int input_width;
    int input_height;

    int input_numel;
    int output_numel;

    cudaStream_t stream;
    nvinfer1::ICudaEngine *engine;
    nvinfer1::IExecutionContext *context;
};

class YOLOClassify
{
public:
    explicit YOLOClassify() {};

    int model_init(const std::string model_path, const int gpu_id);

    void prepare_data_cpu(
        std::vector<cv::Mat> images, float *input_data_host, int input_width, int input_height, bool is_rgb, std::vector<float> means, std::vector<float> stds);

    void trt_engine_infer(float *input_data_device, float *output_data_device, cudaStream_t stream, nvinfer1::IExecutionContext *execution_context);

    // 单标签分类-每个图片返回一个类别
    int classify_V1(std::vector<cv::Mat> &images, std::vector<float> &scores, std::vector<int> &cls_ids, ConfigModel &config_model);
    int postprocess_V1(cv::Mat& feat, std::vector<float>& scores, std::vector<int>& cls_ids);
    
    // 多标签分类-每个图片返回零个或多个类别
    int classify_V2(std::vector<cv::Mat>& images, std::vector<float>& scores, std::vector<int>& cls_ids, ConfigModel& config_model);
    int postprocess_V2(cv::Mat& feat, std::vector<float>& scores, std::vector<int>& cls_ids, float conf);

    void cuda_free();

private:
    std::vector<int> inputTensorShape;

    std::vector<int> outputTensorShape;

    YOLOClsTRTContext *trt = nullptr;
};
