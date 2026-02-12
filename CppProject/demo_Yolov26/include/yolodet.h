#pragma once
#include "trt_engine.h"
#include "enutils.h"

struct YOLODetTRTContext
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

class YOLODetector
{
public:
    explicit YOLODetector() {};

    int model_init(const std::string model_path, const int gpu_id);

    void prepare_data_cpu(
        std::vector<cv::Mat> images, float *input_data_host, int input_width, int input_height, bool is_rgb, std::vector<float> means, std::vector<float> stds);

    void trt_engine_infer(float *input_data_device, float *output_data_device, cudaStream_t stream, nvinfer1::IExecutionContext *execution_context);

    int postprocess(cv::Mat &feat, std::vector<std::vector<Bbox>> &vec_boxes, float conf_thres, float nms_thres, int postproc_type);

    int postprocess_v5(cv::Mat &feat, std::vector<std::vector<Bbox>> &vec_boxes, float conf_thres, float nms_thres);
    int postprocess_v8(cv::Mat &feat, std::vector<std::vector<Bbox>> &vec_boxes, float conf_thres, float nms_thres);
    int postprocess_v10(cv::Mat &feat, std::vector<std::vector<Bbox>> &vec_boxes, float conf_thres);

    int detect1(std::vector<cv::Mat> &images, std::vector<std::vector<Bbox>> &vec_boxes, ConfigModel &config_model);

    int detect2(cv::Mat &image, std::vector<Bbox> &final_boxes, ConfigModel &config_model);

    void cuda_free();

private:
    std::vector<int> inputTensorShape;
    std::vector<int> outputTensorShape;

    YOLODetTRTContext *trt = nullptr;
};
