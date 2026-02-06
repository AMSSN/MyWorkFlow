//#ifndef ONNXTOTRT_ONNX_TO_TRT_H
//#define ONNXTOTRT_ONNX_TO_TRT_H
//class onnx_to_trt {};
//#endif //ONNXTOTRT_ONNX_TO_TRT_H


#pragma once

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

/**
 * @brief 将ONNX模型转换为TensorRT引擎
 * @param onnxPath ONNX模型文件路径
 * @param enginePath 输出引擎文件路径
 * @param precision 精度类型 (FP32/FP16)
 * @return 是否成功
 */
    ONNX_TO_TRT_API bool convertOnnxToTrtEngine(
            const char *onnxPath,
            const char *enginePath,
            int infer_width,
            int infer_height,
            PrecisionType precision = PrecisionType::FP16,
            int gpuIndex = 0
    );
    /**
     * @brief 打印TensorRT引擎信息
     * @param enginePath 引擎文件路径
     * @return 是否成功
     */
    ONNX_TO_TRT_API bool printEngineInfo(const char *enginePath);

} // namespace onnx_to_trt