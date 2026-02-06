//
// Created by M on 2025/12/18.
//
#include "include/OnnxToTrt.h"
#include "iostream"
int main() {

    std::cout<<"hello world"<<std::endl;
//    // 转换ONNX模型到TensorRT引擎 (FP16精度)
//    bool success = onnx_to_trt::convertOnnxToTrtEngine(
//            "seg_p1_20251105.onnx",
//            "seg_p1_20251105.engine",
//            1280,1280,
//            onnx_to_trt::PrecisionType::FP16
//
//    );
//
//    if (success) {
//        std::cout << "Engine created successfully!" << std::endl;
//    } else {
//        std::cerr << "Conversion failed!" << std::endl;
//    }

    onnx_to_trt::printEngineInfo("seg_p1_20251105.engine");
    return 0;


}

#include "NvInferRuntime.h"


