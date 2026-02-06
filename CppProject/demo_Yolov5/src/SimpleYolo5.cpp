#include "SimpleYolo5.h"
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstring>

// 简单的TensorRT日志记录器
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char *msg) noexcept override {
        // 只输出错误和警告信息，避免过多日志
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << msg << std::endl;
        }
    }
};

static Logger gLogger;

namespace SimpleYolo5 {

    ///////////////////////////////////////////////////////////////////////////
    // TRTEngine类实现
    ///////////////////////////////////////////////////////////////////////////

    TRTEngine::TRTEngine() {
    }

    TRTEngine::~TRTEngine() {
        freeMemory();

        if (m_context) {
            m_context->destroy();
            m_context = nullptr;
        }

        if (m_engine) {
            m_engine->destroy();
            m_engine = nullptr;
        }

        if (m_runtime) {
            m_runtime->destroy();
            m_runtime = nullptr;
        }
    }

    bool TRTEngine::load(const std::string &enginePath) {
        // 读取engine文件
        std::ifstream file(enginePath, std::ios::binary);
        if (!file.good()) {
            std::cerr << "Failed to open engine file: " << enginePath << std::endl;
            return false;
        }

        // 获取文件大小
        file.seekg(0, file.end);
        size_t size = file.tellg();
        file.seekg(0, file.beg);

        // 读取文件内容
        std::vector<char> buffer(size);
        if (!file.read(buffer.data(), size)) {
            std::cerr << "Failed to read engine file" << std::endl;
            return false;
        }
        file.close();

        // 创建TensorRT运行时环境
        m_runtime = nvinfer1::createInferRuntime(gLogger);
        if (!m_runtime) {
            std::cerr << "Failed to create TensorRT runtime" << std::endl;
            return false;
        }

        // 创建TensorRT引擎
        m_engine = m_runtime->deserializeCudaEngine(buffer.data(), size);
        if (!m_engine) {
            std::cerr << "Failed to deserialize CUDA engine" << std::endl;
            return false;
        }

        // 创建TensorRT上下文
        m_context = m_engine->createExecutionContext();
        if (!m_context) {
            std::cerr << "Failed to create execution context" << std::endl;
            return false;
        }

        // 检查绑定数量
        int binding_num = m_engine->getNbBindings();
        if (binding_num < 2) {
            std::cerr << "Invalid number of bindings: " << binding_num << std::endl;
            return false;
        }

        // 获取输入维度信息
        auto inputDims = m_engine->getBindingDimensions(0);
        // TensorRT 维度顺序 永远是 NCHW
        m_input_batch = inputDims.d[0];
        m_input_height = inputDims.d[2];
        m_input_width = inputDims.d[3];

        // 计算输入元素数量
        m_input_numel = 1;
        m_inputTensorShape.clear();
        for (int i = 0; i < inputDims.nbDims; ++i) {
            m_input_numel *= inputDims.d[i];
            m_inputTensorShape.push_back(inputDims.d[i]);
        }

        // 获取输出维度信息
        auto outputDims = m_engine->getBindingDimensions(1);
        m_output_numel = 1;
        m_outputTensorShape.clear();
        for (int i = 0; i < outputDims.nbDims; ++i) {
            m_output_numel *= outputDims.d[i];
            m_outputTensorShape.push_back(outputDims.d[i]);
        }

        // 创建CUDA流
        if (cudaStreamCreate(&m_stream) != cudaSuccess) {
            std::cerr << "Failed to create CUDA stream" << std::endl;
            return false;
        }

        // 分配内存
        if (!allocateMemory()) {
            std::cerr << "Failed to allocate memory" << std::endl;
            return false;
        }

        std::cout << "Engine loaded successfully. Input: " << m_input_width << "x" << m_input_height << std::endl;
        return true;
    }

    bool TRTEngine::executeInference() {
        if (!isLoaded()) {
            std::cerr << "Engine not loaded" << std::endl;
            return false;
        }

        // 执行推理
        void *bindings[] = {m_input_data_device, m_output_data_device};
        bool success = m_context->executeV2(bindings);
        if (!success) {
            std::cerr << "Inference execution failed" << std::endl;
            return false;
        }

        return true;
    }

    bool TRTEngine::allocateMemory() {
        // 释放之前的内存（如果存在）
        freeMemory();

        // 分配主机内存
        try {
            m_input_data_host = new float[m_input_numel];
            m_output_data_host = new float[m_output_numel];
        } catch (const std::bad_alloc &e) {
            std::cerr << "Failed to allocate host memory: " << e.what() << std::endl;
            return false;
        }

        // 分配设备内存
        if (cudaMalloc(reinterpret_cast<void **>(&m_input_data_device), m_input_numel * sizeof(float)) != cudaSuccess) {
            std::cerr << "Failed to allocate input device memory" << std::endl;
            return false;
        }

        if (cudaMalloc(reinterpret_cast<void **>(&m_output_data_device), m_output_numel * sizeof(float)) !=
            cudaSuccess) {
            std::cerr << "Failed to allocate output device memory" << std::endl;
            return false;
        }

        return true;
    }

    void TRTEngine::freeMemory() {
        // 释放CUDA流
        if (m_stream) {
            cudaStreamDestroy(m_stream);
            m_stream = nullptr;
        }

        // 释放设备内存
        if (m_input_data_device) {
            cudaFree(m_input_data_device);
            m_input_data_device = nullptr;
        }

        if (m_output_data_device) {
            cudaFree(m_output_data_device);
            m_output_data_device = nullptr;
        }

        // 释放主机内存
        if (m_input_data_host) {
            delete[] m_input_data_host;
            m_input_data_host = nullptr;
        }

        if (m_output_data_host) {
            delete[] m_output_data_host;
            m_output_data_host = nullptr;
        }
    }

    bool TRTEngine::copyHostToDevice() {
        if (!m_input_data_host || !m_input_data_device) {
            std::cerr << "Invalid input buffers" << std::endl;
            return false;
        }

        if (cudaMemcpyAsync(m_input_data_device, m_input_data_host,
                            m_input_numel * sizeof(float),
                            cudaMemcpyHostToDevice, m_stream) != cudaSuccess) {
            std::cerr << "Failed to copy data from host to device" << std::endl;
            return false;
        }

        return true;
    }

    bool TRTEngine::copyDeviceToHost() {
        if (!m_output_data_host || !m_output_data_device) {
            std::cerr << "Invalid output buffers" << std::endl;
            return false;
        }

        if (cudaMemcpyAsync(m_output_data_host, m_output_data_device,
                            m_output_numel * sizeof(float),
                            cudaMemcpyDeviceToHost, m_stream) != cudaSuccess) {
            std::cerr << "Failed to copy data from device to host" << std::endl;
            return false;
        }

        // 等待数据传输完成
        if (cudaStreamSynchronize(m_stream) != cudaSuccess) {
            std::cerr << "Failed to synchronize CUDA stream" << std::endl;
            return false;
        }

        return true;
    }

    int TRTEngine::getOutputDimensions(int index) const {
        if (index >= 0 && index < m_outputTensorShape.size()) {
            return m_outputTensorShape[index];
        }
        return 0;
    }

    int TRTEngine::detectNumClasses() const {
        // 对于YOLOv5模型，输出格式通常是 [batch_size, num_boxes, 5 + num_classes]
        // 其中5是 [x_center, y_center, width, height, confidence]
        if (!m_engine || m_outputTensorShape.size() < 2) {
            return 1; // 默认类别数
        }

        // 打印输出维度信息用于调试
        std::cout << "Output dimensions: ";
        for (size_t i = 0; i < m_outputTensorShape.size(); ++i) {
            std::cout << m_outputTensorShape[i] << " ";
        }
        std::cout << std::endl;

        // 标准YOLOv5输出格式应该是3维或4维
        if (m_outputTensorShape.size() == 3) {
            // 对于 [batch, num_boxes, 5 + num_classes] 格式
            int lastDim = m_outputTensorShape[2];
            if (lastDim > 5) {  // 确保至少有5个基础属性
                return lastDim - 5;
            }
        } else if (m_outputTensorShape.size() == 2) {
            // 对于某些简化格式 [num_boxes, 5 + num_classes]
            int lastDim = m_outputTensorShape[1];
            if (lastDim > 5) {
                return lastDim - 5;
            }
        }

        // 如果无法自动检测，返回默认值1
        std::cout << "Warning: Could not auto-detect number of classes, using default (1)" << std::endl;
        return 1;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Detector类实现
    ///////////////////////////////////////////////////////////////////////////

    Detector::Detector(const std::string &enginePath,
                       float confThreshold,
                       float nmsThreshold) :
            m_confThreshold(confThreshold),
            m_nmsThreshold(nmsThreshold),
            m_trtEngine(std::make_unique<TRTEngine>()) {

        loadModel(enginePath);
    }

    Detector::~Detector() {
        // TRTEngine会自动释放资源
    }

    bool Detector::loadModel(const std::string &enginePath) {
        if (!m_trtEngine->load(enginePath)) {
            return false;
        }

        // 自动检测类别数量
        m_numClasses = m_trtEngine->detectNumClasses();
        std::cout << "Auto-detected number of classes: " << m_numClasses << std::endl;

        return true;
    }

    std::vector<Detection> Detector::infer(const unsigned char *imageData, int width, int height) {
        if (!isLoaded()) {
            std::cerr << "Model not loaded properly" << std::endl;
            return {};
        }

        // 保存图像尺寸
        m_InputImageWidth = width;
        m_InputImageHeight = height;

        // 执行预处理
        preprocess(imageData, width, height);

        // 将数据从主机复制到设备
        if (!m_trtEngine->copyHostToDevice()) {
            return {};
        }

        // 执行推理
        if (!m_trtEngine->executeInference()) {
            return {};
        }

        // 将结果从设备复制到主机
        if (!m_trtEngine->copyDeviceToHost()) {
            return {};
        }

        // 后处理并返回结果
        return postprocess();
    }

    void Detector::preprocess(const unsigned char *imageData, int width, int height) {
        // 输入合法性检查
        if (width <= 0 || height <= 0 || !imageData) {
            std::cerr << "Invalid image data or dimensions" << std::endl;
            return;
        }

        // 获取输入缓冲区
        float *inputBuffer = m_trtEngine->getInputHostBuffer();
        if (!inputBuffer) {
            std::cerr << "Invalid input buffer" << std::endl;
            return;
        }

        // 获取模型输入尺寸
        int inputWidth = m_trtEngine->getInputWidth();
        int inputHeight = m_trtEngine->getInputHeight();

        // 执行图像缩放和填充
        resizeAndPadImage(imageData, width, height, inputBuffer, inputWidth, inputHeight);
    }

    void Detector::resizeAndPadImage(const unsigned char *srcData, int srcWidth, int srcHeight,
                                     float *dstData, int dstWidth, int dstHeight) {
        // 计算缩放比例（保持宽高比）
        float scale = std::min(
                static_cast<float>(dstWidth) / static_cast<float>(srcWidth),
                static_cast<float>(dstHeight) / static_cast<float>(srcHeight)
        );

        int newW = static_cast<int>(float(srcWidth) * scale);
        int newH = static_cast<int>(float(srcHeight) * scale);
        int padW = (dstWidth - newW) / 2;
        int padH = (dstHeight - newH) / 2;

        // 初始化目标缓冲区为0（黑色填充）
        std::memset(dstData, 0, dstWidth * dstHeight * 3 * sizeof(float));

        // 使用双线性插值进行图像缩放和通道转换（BGR -> RGB）
        for (int y = 0; y < newH; ++y) {
            // 计算源图像的y坐标（双线性插值）
            float srcY_f = float(y) / scale;
            int y0 = static_cast<int>(srcY_f);
            int y1 = std::min(y0 + 1, srcHeight - 1);
            float yRatio = srcY_f - y0;

            for (int x = 0; x < newW; ++x) {
                // 计算源图像的x坐标（双线性插值）
                float srcX_f = float(x) / scale;
                int x0 = static_cast<int>(srcX_f);
                int x1 = std::min(x0 + 1, srcWidth - 1);
                float xRatio = srcX_f - x0;

                // 目标图像中的位置
                int dstX = padW + x;
                int dstY = padH + y;
                int dstIdx = dstY * dstWidth + dstX;

                // 确保目标索引有效
                if (dstX < 0 || dstX >= dstWidth || dstY < 0 || dstY >= dstHeight) {
                    continue;
                }

                // 双线性插值计算四个角的像素值
                for (int c = 0; c < 3; ++c) { // BGR通道
                    // 获取四个角的像素值
                    int idx00 = (y0 * srcWidth + x0) * 3 + c;
                    int idx01 = (y0 * srcWidth + x1) * 3 + c;
                    int idx10 = (y1 * srcWidth + x0) * 3 + c;
                    int idx11 = (y1 * srcWidth + x1) * 3 + c;

                    // 双线性插值
                    float val = (1.0f - xRatio) * (1.0f - yRatio) * srcData[idx00] +
                                xRatio * (1.0f - yRatio) * srcData[idx01] +
                                (1.0f - xRatio) * yRatio * srcData[idx10] +
                                xRatio * yRatio * srcData[idx11];

                    // BGR -> RGB 转换，并应用归一化
                    int rgbIdx = 2 - c; // BGR -> RGB
                    dstData[dstIdx * 3 + rgbIdx] = (val - m_mean[c]) / m_std[c];
                }
            }
        }
    }

    std::vector<Detection> Detector::postprocess() {
        std::vector<Detection> detections;

        // 获取输出缓冲区
        float *outputBuffer = m_trtEngine->getOutputHostBuffer();
        if (!outputBuffer) {
            std::cerr << "Invalid output buffer" << std::endl;
            return detections;
        }

        // 确保类别数至少为1
        int numClasses = std::max(1, m_numClasses);
        int numAttributes = 5 + numClasses; // x, y, w, h, conf + class scores

        // 获取输出元素数量
        size_t outputNumel = m_trtEngine->getOutputSize() / sizeof(float);

        // 解析YOLOv5输出
        for (size_t i = 0; i + numAttributes <= outputNumel; i += numAttributes) {
            // 获取置信度
            float confidence = outputBuffer[i + 4];

            // 过滤低置信度检测
            if (confidence < m_confThreshold) {
                continue;
            }

            // 寻找最高概率的类别
            int maxClassIdx = 0;
            float maxClassProb = outputBuffer[i + 5];
            for (int j = 1; j < numClasses; ++j) {
                if (outputBuffer[i + 5 + j] > maxClassProb) {
                    maxClassProb = outputBuffer[i + 5 + j];
                    maxClassIdx = j;
                }
            }

            // 计算最终置信度
            float finalConfidence = confidence * maxClassProb;
            if (finalConfidence < m_confThreshold) {
                continue;
            }

            // 创建检测结果
            Detection det{};
            det.x = outputBuffer[i + 0];
            det.y = outputBuffer[i + 1];
            det.w = outputBuffer[i + 2];
            det.h = outputBuffer[i + 3];
            det.class_id = maxClassIdx;
            det.confidence = finalConfidence;

            // 缩放坐标到原始图像尺寸
            scaleCoords(det.x, det.y, det.w, det.h);

            detections.push_back(det);
        }

        // 应用NMS
        return nms(detections);
    }

    std::vector<Detection> Detector::nms(const std::vector<Detection> &detections) const {
        if (detections.empty()) return {};

        // 按置信度降序排序
        std::vector<Detection> sortedDetections = detections;
        std::sort(sortedDetections.begin(), sortedDetections.end(),
                  [](const Detection &a, const Detection &b) {
                      return a.confidence > b.confidence;
                  });

        std::vector<bool> suppressed(sortedDetections.size(), false);
        std::vector<Detection> result;

        // 计算IoU并执行NMS
        for (size_t i = 0; i < sortedDetections.size(); ++i) {
            if (suppressed[i]) continue;

            result.push_back(sortedDetections[i]);

            for (size_t j = i + 1; j < sortedDetections.size(); ++j) {
                if (suppressed[j]) continue;

                // 计算IoU
                float x1 = std::max(sortedDetections[i].x - sortedDetections[i].w / 2,
                                    sortedDetections[j].x - sortedDetections[j].w / 2);
                float y1 = std::max(sortedDetections[i].y - sortedDetections[i].h / 2,
                                    sortedDetections[j].y - sortedDetections[j].h / 2);
                float x2 = std::min(sortedDetections[i].x + sortedDetections[i].w / 2,
                                    sortedDetections[j].x + sortedDetections[j].w / 2);
                float y2 = std::min(sortedDetections[i].y + sortedDetections[i].h / 2,
                                    sortedDetections[j].y + sortedDetections[j].h / 2);

                float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
                float unionArea = sortedDetections[i].w * sortedDetections[i].h +
                                  sortedDetections[j].w * sortedDetections[j].h -
                                  intersection;

                float iou = intersection / unionArea;

                // 如果IoU超过阈值，抑制第二个检测
                if (iou >= m_nmsThreshold) {
                    suppressed[j] = true;
                }
            }
        }

        return result;
    }

    void Detector::scaleCoords(float &x, float &y, float &w, float &h) const {
        // 计算缩放比例和填充
        float scale = std::min(
                static_cast<float>(m_trtEngine->getInputWidth()) / float(m_InputImageWidth),
                static_cast<float>(m_trtEngine->getInputHeight()) / float(m_InputImageHeight)
        );

        int padW = static_cast<int>((m_trtEngine->getInputWidth() - m_InputImageWidth * scale) / 2);
        int padH = static_cast<int>((m_trtEngine->getInputHeight() - m_InputImageHeight * scale) / 2);

        // 从网络输出坐标转换回原始图像坐标
        x = (x - float(padW)) / scale;
        y = (y - float(padH)) / scale;
        w = w / scale;
        h = h / scale;

        // 确保坐标在图像范围内
        x = std::max(0.0f, std::min(static_cast<float>(m_InputImageWidth), x));
        y = std::max(0.0f, std::min(static_cast<float>(m_InputImageHeight), y));
    }

} // namespace SimpleYolo5