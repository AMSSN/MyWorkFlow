//#ifndef ONNXTOTRT_ONNX_TO_TRT_H
//#define ONNXTOTRT_ONNX_TO_TRT_H
//class onnx_to_trt {};
//#endif //ONNXTOTRT_ONNX_TO_TRT_H


#pragma once

#include "NvInferRuntimeBase.h"

#ifdef _WIN32
#define ONNX_TO_TRT_API __declspec(dllexport)
#else
#define ONNX_TO_TRT_API __attribute__((visibility("default")))
#endif

namespace onnx_to_trt {

    enum class PrecisionType {
        FP32,
        FP16,
        INT8
    };

    struct Dim4 {
        int n, c, h, w;

        // 辅助转换为 TensorRT 的 Dims
        nvinfer1::Dims toDims() const {
            nvinfer1::Dims dims;
            dims.nbDims = 4;
            dims.d[0] = n;
            dims.d[1] = c;
            dims.d[2] = h;
            dims.d[3] = w;
            return dims;
        }
    };

    struct ConversionConfig {
        const char *onnxPath{nullptr};           ///< 输入ONNX模型路径
        const char *enginePath{nullptr};        ///< 输出TensorRT引擎路径
        int inferWidth{-1};                    ///< 推理输入宽度 (静态或opt)
        int inferHeight{-1};                   ///< 推理输入高度 (静态或opt)
        PrecisionType precision{PrecisionType::FP32}; ///< 目标推理精度
        int gpuIndex{0};                        ///< 使用的GPU设备索引
        const char *inputName{nullptr};        ///< 输入名称
        // 可扩展字段...
        int batchSize{-1};                      ///< 显式Batch大小 (-1表示动态batch)
        bool enableFP16{false};                 ///< 是否启用FP16（即使precision!=FP16也可手动开）
        bool enableInt8{false};                 ///< 是否启用INT8校准（需提供校准数据）
        Dim4 minShape{1, 3, 224, 224};              ///< 最小尺寸 (min)
        Dim4 optShape{1, 3, 640, 640};              ///< 最优尺寸 (opt)
        Dim4 maxShape{8, 3, 1280, 1280};            ///< 最大尺寸 (max)
    };

    /**
     * @brief 将ONNX模型转换为TensorRT引擎
     * @param config 转换配置参数结构体
     * @return 是否成功
     */
    ONNX_TO_TRT_API bool convertOnnxToTrtEngine(const ConversionConfig &config);

    /**
     * @brief 打印TensorRT引擎信息
     * @param enginePath 引擎文件路径
     * @return 是否成功
     */
    ONNX_TO_TRT_API bool printEngineInfo(const char *enginePath);

} // namespace onnx_to_trt