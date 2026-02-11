# CppProject

### 文件目录

3rdparty：第三方库，包括cuda、halcon、opencv、OpenSSL、spdlog、tensorrt :white_check_mark:
3rdparty(other version)：其他版本的第三方库 :white_check_mark:
demo_OnnxToTrt：一个onnx转engine的库 :white_check_mark:
demo_Yolo：trt推理yolo:x:



### TODO

trt推理yolo



### 项目介绍

#### demo_OnnxToTrt 一个onnx转engine的库

设置trt相关的参数，比如输入输出路径，形状等等，使用convertOnnxToTrtEngine转换，使用printEngineInfo显示engine信息。

使用方法：

```
#include "include/OnnxToTrt.h"
#include "iostream"
int main() {
    std::cout<<"start convert onnx to engine."<<std::endl;
    onnx_to_trt::ConversionConfig config;
    config.onnxPath = "yolo26_P2.onnx";
    config.enginePath = "yolo26_P2.engine";
    config.inferWidth = 1280;
    config.inferHeight = 1280;
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

```

### 更新日志

#### 2026.2.10

优化了demo_OnnxToTrt的参数输入方法，能够设置min、opt、max参数，虽然导出的engine是[-1,3,-1,-1]不过应该不影响使用，只要在推理时设置Bindings就行了。

