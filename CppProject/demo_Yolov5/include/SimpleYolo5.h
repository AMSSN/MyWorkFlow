#pragma once

#include <string>
#include <vector>
#include <memory>
#include <NvInferRuntime.h>

// 定义平台相关的导出宏
#ifdef _WIN32
#define SIMPLE_YOLO5_API __declspec(dllexport)
#else
#define SIMPLE_YOLO5_API __attribute__((visibility("default")))
#endif

namespace SimpleYolo5 {

    // 检测结果结构体
    struct Detection {
        float x, y, w, h;      // 中心点坐标 + 宽高
        int class_id;          // 类别ID
        float confidence;      // 置信度分数
    };

    // TRT引擎封装类 - 负责TensorRT资源管理和推理操作
    class SIMPLE_YOLO5_API TRTEngine {
    public:
        TRTEngine();
        ~TRTEngine();

        // 加载TensorRT引擎
        bool load(const std::string &enginePath);
        
        // 执行推理
        bool executeInference();
        
        // 内存管理
        bool allocateMemory();
        void freeMemory();
        
        // 获取内存指针
        float* getInputHostBuffer() { return m_input_data_host; }
        float* getOutputHostBuffer() { return m_output_data_host; }
        float* getInputDeviceBuffer() { return m_input_data_device; }
        float* getOutputDeviceBuffer() { return m_output_data_device; }
        
        // 数据传输
        bool copyHostToDevice();
        bool copyDeviceToHost();
        
        // 获取模型信息
        int getInputWidth() const { return m_input_width; }
        int getInputHeight() const { return m_input_height; }
        int getInputBatch() const { return m_input_batch; }
        size_t getInputSize() const { return m_input_numel * sizeof(float); }
        size_t getOutputSize() const { return m_output_numel * sizeof(float); }
        int getOutputDimensions(int index) const; // 获取输出维度
        
        // 检查引擎状态
        bool isLoaded() const { return m_engine != nullptr && m_context != nullptr; }
        
        // 获取类别数量
        int detectNumClasses() const;
        
        // 禁用拷贝操作
        TRTEngine(const TRTEngine &) = delete;
        TRTEngine &operator=(const TRTEngine &) = delete;
        
    private:
        // TensorRT核心组件
        nvinfer1::IRuntime* m_runtime{nullptr};
        nvinfer1::ICudaEngine* m_engine{nullptr};
        nvinfer1::IExecutionContext* m_context{nullptr};
        cudaStream_t m_stream{nullptr};
        
        // 内存缓冲区
        float* m_input_data_host{nullptr};
        float* m_input_data_device{nullptr};
        float* m_output_data_host{nullptr};
        float* m_output_data_device{nullptr};
        
        // 尺寸信息
        int m_input_batch{1};
        int m_input_width{640};
        int m_input_height{640};
        size_t m_input_numel{0};
        size_t m_output_numel{0};
        
        // 张量维度信息
        std::vector<int> m_inputTensorShape;
        std::vector<int> m_outputTensorShape;
    };

    // YOLOv5检测器类 - 负责算法逻辑，使用TRTEngine进行推理
    class SIMPLE_YOLO5_API Detector {
    public:
        /**
         * @brief 构造函数，自动加载模型
         * @param enginePath TensorRT engine文件路径
         * @param confThreshold 置信度阈值（默认0.25）
         * @param nmsThreshold NMS IoU阈值（默认0.45）
         */
        explicit Detector(const std::string &enginePath,
                          float confThreshold = 0.25f,
                          float nmsThreshold = 0.45f);

        /**
         * @brief 析构函数，自动释放资源
         */
        ~Detector();

        /**
         * @brief 推理单张图像
         * @param imageData BGR图像数据指针 (HWC格式, uint8_t*)
         * @param width 图像宽度
         * @param height 图像高度
         * @return 检测结果列表
         */
        std::vector<Detection> infer(const unsigned char *imageData,
                                     int width,
                                     int height);

        /**
         * @brief 获取模型输入尺寸
         */
        int getInputWidth() const { return m_trtEngine->getInputWidth(); }
        int getInputHeight() const { return m_trtEngine->getInputHeight(); }

        /**
         * @brief 检查模型是否成功加载
         */
        bool isLoaded() const { return m_trtEngine && m_trtEngine->isLoaded(); }

        // 禁用拷贝操作
        Detector(const Detector &) = delete;
        Detector &operator=(const Detector &) = delete;

    private:
        // 内部功能函数
        bool loadModel(const std::string &enginePath);
        void preprocess(const unsigned char *imageData, int width, int height);
        std::vector<Detection> postprocess();
        std::vector<Detection> nms(const std::vector<Detection> &detections) const;
        void scaleCoords(float &x, float &y, float &w, float &h) const;
        
        // 不依赖OpenCV的图像缩放和填充函数
        void resizeAndPadImage(const unsigned char *srcData, int srcWidth, int srcHeight,
                              float *dstData, int dstWidth, int dstHeight);

        // TRT引擎实例
        std::unique_ptr<TRTEngine> m_trtEngine;
        
        // 模型参数
        int m_numClasses{0};
        float m_confThreshold{0.25f};
        float m_nmsThreshold{0.45f};
        
        // 临时存储当前图像尺寸，用于坐标缩放
        int m_InputImageWidth{0};
        int m_InputImageHeight{0};
        
        // 归一化参数（固定值） (适合YOLOv5)
        const float m_mean[3] = {0.0f, 0.0f, 0.0f};
        const float m_std[3] = {255.0f, 255.0f, 255.0f};
    };
} // namespace SimpleYolo5
