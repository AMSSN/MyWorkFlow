#include "../include/OnnxToTrt.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <iostream>
#include <fstream>
#include <vector>

using namespace nvinfer1;
using namespace nvonnxparser;

// 定义 Logger 类，继承自 nvinfer1::ILogger
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept override {
        // 根据需要过滤日志级别
        if (severity <= Severity::kINFO) {
            printf("%s\n", msg);
        }
    }
};

// 全局 Logger 实例
static Logger gLogger;

namespace onnx_to_trt {

    bool convertOnnxToTrtEngine(const char *onnxPath, const char *enginePath, int infer_width, int infer_height,
                                PrecisionType precision, int gpuIndex) {
        // 设置要使用的 GPU 设备
        cudaError_t cudaStatus = cudaSetDevice(gpuIndex);
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaSetDevice failed: " << cudaGetErrorString(cudaStatus) << std::endl;
            return false;
        }

        // 1. 创建TensorRT builder
        auto builder = createInferBuilder(gLogger);
        if (!builder) {
            std::cerr << "Failed to create builder!" << std::endl;
            return false;
        }

        // 2. 创建网络
        // 创建网络定义时添加 EXPLICIT_BATCH 标志
        auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        nvinfer1::INetworkDefinition *network = builder->createNetworkV2(explicitBatch);
        if (!network) {
            std::cerr << "Failed to create network!" << std::endl;
            builder->destroy();
            return false;
        }

        // 3. 创建ONNX解析器
        auto parser = createParser(*network, gLogger);
        if (!parser) {
            std::cerr << "Failed to create parser!" << std::endl;
            network->destroy();
            builder->destroy();
            return false;
        }

        // 4. 解析ONNX模型
        if (!parser->parseFromFile(onnxPath, static_cast<int>(ILogger::Severity::kINFO))) {
            std::cerr << "Failed to parse ONNX file: " << onnxPath << std::endl;
            parser->destroy();
            network->destroy();
            builder->destroy();
            return false;
        }

        // 5. 配置构建参数
        auto config = builder->createBuilderConfig();
        if (!config) {
            std::cerr << "Failed to create builder config!" << std::endl;
            parser->destroy();
            network->destroy();
            builder->destroy();
            return false;
        }
        // 构建配置并设置 profile
        IOptimizationProfile *profile = builder->createOptimizationProfile();
        // 设置最小、最优、最大尺寸（三者可相同表示静态）
        profile->setDimensions("input_name", OptProfileSelector::kMIN, Dims4(1, 3, 640, 640));
        profile->setDimensions("input_name", OptProfileSelector::kOPT, Dims4(1, 3, 224, 224));
        profile->setDimensions("input_name", OptProfileSelector::kMAX, Dims4(8, 3, 1280, 1280));
        config->addOptimizationProfile(profile);

        // 6. 设置精度
        if (precision == PrecisionType::FP16) {
            config->setFlag(BuilderFlag::kFP16);
        }

        // 7-1. 构建engine引擎
        // (序列化的 engine 字节流)
        IHostMemory *engineData = builder->buildSerializedNetwork(*network, *config);
//        // 7-2. 构建engine引擎
//        // (可执行的运行时引擎对象)
//        auto engine = builder->buildEngineWithConfig(*network, *config);
//        if (!engine) {
//            std::cerr << "Failed to build engine!" << std::endl;
//            config->destroy();
//            parser->destroy();
//            network->destroy();
//            builder->destroy();
//            return false;
//        }
//        // 序列化引擎为 IHostMemory 对象
//        nvinfer1::IHostMemory *engineData = engine->serialize();

        // 检查引擎序列化成功
        if (!engineData) {
            std::cerr << "Failed to serialize engine." << std::endl;
            return false;
        }

        // 8. 保存到文件
        std::ofstream file(enginePath, std::ios::binary);
        if (!file) {
            std::cerr << "Failed to open file: " << enginePath << std::endl;
            engineData->destroy();
            config->destroy();
            parser->destroy();
            network->destroy();
            builder->destroy();
            return false;
        }
        file.write(reinterpret_cast<const char *>(engineData->data()),
                   static_cast<std::streamsize>(engineData->size()));
        file.close();
        std::cout << "Serialized engine saved to: " << enginePath << " (size: " << engineData->size() << " bytes)"
                  << std::endl;

        // 9. 清理资源
        engineData->destroy();
        config->destroy();
        parser->destroy();
        network->destroy();
        builder->destroy();

        return true;
    }

    // 检查engine文件
    bool printEngineInfo(const char* enginePath) {
        // 读取engine文件到内存
        std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            std::cerr << "Failed to open engine file: " << enginePath << std::endl;
            return false;
        }
        
        std::streamsize fileSize = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::vector<char> engineBuffer(fileSize);
        if (!file.read(engineBuffer.data(), fileSize)) {
            std::cerr << "Failed to read engine file: " << enginePath << std::endl;
            file.close();
            return false;
        }
        file.close();
        
        // 创建运行时对象（注意：需自己管理 logger）
        nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
        if (!runtime) {
            std::cerr << "Failed to create TensorRT runtime." << std::endl;
            return false;
        }

        // 反序列化得到 engine
        nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engineBuffer.data(), engineBuffer.size());
        if (!engine) {
            std::cerr << "Failed to deserialize engine." << std::endl;
            runtime->destroy();
            return false;
        }

        // 打印基本信息
        std::cout << "=== TensorRT Engine Info ===" << std::endl;
        std::cout << "Number of bindings: " << engine->getNbBindings() << std::endl;

        // 遍历所有 binding（输入/输出）
        for (int i = 0; i < engine->getNbBindings(); ++i) {
            auto name = engine->getBindingName(i);
            auto dims = engine->getBindingDimensions(i);
            auto type = engine->getBindingDataType(i);
            bool isInput = engine->bindingIsInput(i);

            std::cout << "Binding[" << i << "] "
                      << (isInput ? "Input" : "Output") << ", "
                      << "Name: " << name << ", "
                      << "Type: " << static_cast<int>(type) << ", "
                      << "Dims: ";

            for (int j = 0; j < dims.nbDims; ++j) {
                std::cout << (j ? "x" : "") << dims.d[j];
            }
            std::cout << std::endl;
        }

        // 清理资源
        engine->destroy();
        runtime->destroy();
        return true;
    }

} // namespace onnx_to_trt