//
// Created by M on 2025/12/18.
//
#include "include/OnnxToTrt.h"
#include "iostream"
int main() {

    std::cout<<"start convert onnx to engine."<<std::endl;
    onnx_to_trt::ConversionConfig config;
    config.onnxPath = "yolo26_P2.onnx";
    config.enginePath = "yolo26_P2.engine";
//    config.inferWidth = 1280;
//    config.inferHeight = 1280;
    config.precision = onnx_to_trt::PrecisionType::FP16;
    config.gpuIndex = 0;
    config.inputName = "images";
    // 转换ONNX模型到TensorRT引擎 (FP16精度)
    bool success = onnx_to_trt::convertOnnxToTrtEngine(config);

    if (success) {
        std::cout << "Engine created successfully!" << std::endl;
    } else {
        std::cerr << "Conversion failed!" << std::endl;
    }

    onnx_to_trt::printEngineInfo("yolo26_P2.engine");
    return 0;


}

#include "NvInferRuntime.h"


