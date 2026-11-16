#pragma once
#include "trt_engine.h"
#include "../base/utils.h"

struct YOLOSegTRTContext
{
    float *input_data_host;
    float *input_data_device;
    float *output0_data_device;
    float *output0_data_host;
    float *output1_data_device;
    float *output1_data_host;

    int input_batch;
    int input_width;
    int input_height;

    int input_numel;
    int output0_numel;
    int output1_numel;

    cudaStream_t stream;
    nvinfer1::ICudaEngine *engine;
    nvinfer1::IExecutionContext *context;
};

class YOLOSegmentor
{
public:
    explicit YOLOSegmentor() {};

    int model_init(const std::string model_path, const int gpu_id);

    void prepare_data_cpu(
        std::vector<cv::Mat> images, float *input_data_host, int input_width, int input_height, bool is_rgb, std::vector<float> means, std::vector<float> stds);

    void trt_engine_infer(
        float *input_data_device, float *output0_data_device, float *output1_data_device, cudaStream_t stream, nvinfer1::IExecutionContext *execution_context);

    //int postprocess(std::vector<cv::Mat> &out_feats, std::vector<std::vector<YoloMask>> &vec_segs, float conf_thres, float nms_thres);
    int postprocess(std::vector<cv::Mat>& out_feats, std::vector<std::vector<YoloMask>>& vec_segs, float conf_thres, float nms_thres, int postproc_type);

    int postprocess_v5(std::vector<cv::Mat>& out_feats, std::vector<std::vector<YoloMask>>& vec_segs, float conf_thres, float nms_thres);
    int postprocess_v8(std::vector<cv::Mat>& out_feats, std::vector<std::vector<YoloMask>>& vec_segs, float conf_thres, float nms_thres);
    int postprocess_v10(std::vector<cv::Mat>& out_feats, std::vector<std::vector<YoloMask>>& vec_segs, float conf_thres);

    void get_yolo_mask(const cv::Mat &maskProposals, const cv::Mat &mask_protos, YoloMask&output);

    int detect1(std::vector<cv::Mat> &images, std::vector<std::vector<YoloMask>> &vec_outsegs, ConfigModel &config_model);

    int detect2(cv::Mat &image, std::vector<YoloMask> &final_outsegs, ConfigModel &config_model);

    void cuda_free();

private:
    std::vector<int> input_tensor_shapes;
    std::vector<int> output0_tensor_shapes;
    std::vector<int> output1_tensor_shapes;

    bool output_tensor_order = true;

    int _classNum = -1;
    int _segChannels = -1;
    int _segHeight = -1;
    int _segWidth = -1;

    YOLOSegTRTContext *trt = nullptr;
};
