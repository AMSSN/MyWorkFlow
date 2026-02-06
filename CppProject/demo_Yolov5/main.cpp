//#include <iostream>
//#include <opencv2/opencv.hpp>
//#include "SimpleYolo5.h"
//#include "OnnxToTrt.h"
//
//int main(int argc, char **argv) {
//    // 检查命令行参数
//    std::string enginePath = (argc > 1) ? argv[1] : "glass.engine";
//    std::string imagePath = (argc > 2) ? argv[2] : "test_image.png";
//    std::string outputPath = (argc > 3) ? argv[3] : "output_result.jpg";
//
//    std::cout << "Simple YOLOv5 Inference Demo" << std::endl;
//    std::cout << "- Engine: " << enginePath << std::endl;
//    std::cout << "- Image: " << imagePath << std::endl;
//
////    onnx_to_trt::convertOnnxToTrtEngine("glass.onnx", "glass.engine", 1280, 1280);
//    onnx_to_trt::printEngineInfo(enginePath.c_str());
//
//    try {
//        SimpleYolo5::Detector detector(enginePath);
//        std::cerr << detector.isLoaded() << std::endl;
//        std::cout << "Model loaded successfully. Input size: "
//                  << detector.getInputWidth() << "x" << detector.getInputHeight() << std::endl;
//        // 读取图像
//        cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
//        std::cout << "image-shape: " << image.rows << "x" << image.cols << std::endl;
//        std::cout << "channel: " << image.channels() << std::endl;
//        std::cout << "dims: " << image.dims << std::endl;
//        if (image.empty()) {
//            std::cerr << "Failed to load image: " << imagePath << std::endl;
//        }
//        // 执行推理
//        auto start = cv::getTickCount();
//        std::vector<SimpleYolo5::Detection> results = detector.infer(image.data, image.cols, image.rows);
//        double inferenceTime = (double(cv::getTickCount()) - double(start)) / cv::getTickFrequency() * 1000;
//        std::cout << "Inference completed in " << inferenceTime << " ms" << std::endl;
//        std::cout << "Detected " << results.size() << " objects" << std::endl;
//        // 显示检测结果并绘制边界框
//        for (size_t i = 0; i < results.size(); ++i) {
//            const auto &det = results[i];
//            // 计算边界框坐标
//            int x1 = static_cast<int>(det.x - det.w / 2);
//            int y1 = static_cast<int>(det.y - det.h / 2);
//            int x2 = static_cast<int>(det.x + det.w / 2);
//            int y2 = static_cast<int>(det.y + det.h / 2);
//            // 绘制边界框和标签
//            cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
//
//            std::string label = "Class: " + std::to_string(det.class_id) +
//                                " (" + std::to_string(static_cast<int>(det.confidence * 100)) + "%)";
//            cv::putText(image, label, cv::Point(x1, y1 - 10),
//                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
//            std::cout << "- Object " << i + 1 << ": Class=" << det.class_id
//                      << ", Confidence=" << det.confidence
//                      << ", BBox=[" << x1 << "," << y1 << "," << x2 << "," << y2 << "]" << std::endl;
//        }
//        // 保存结果图像
//        if (cv::imwrite(outputPath, image)) {
//            std::cout << "Results saved to " << outputPath << std::endl;
//        } else {
//            std::cerr << "Failed to save output image" << std::endl;
//        }
//    } catch (const std::exception &e) {
//        std::cerr << "Error: " << e.what() << std::endl;
//        return -1;
//    }
//    return 0;
//}




#include "SimpleYolo5.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {
    // 检查命令行参数
    std::string enginePath = (argc > 1) ? argv[1] : "glass.engine";
    std::string imagePath = (argc > 2) ? argv[2] : "test_image.png";
    std::string outputPath = (argc > 3) ? argv[3] : "output_result.jpg";

    std::cout << "Simple YOLOv5 Inference Demo" << std::endl;
    std::cout << "- Engine: " << enginePath << std::endl;
    std::cout << "- Image: " << imagePath << std::endl;

    try {
        // 初始化检测器
        SimpleYolo5::Detector detector(enginePath, 0.3f, 0.45f);

        if (!detector.isLoaded()) {
            std::cerr << "Failed to load detector" << std::endl;
            return 1;
        }

        std::cout << "Detector initialized successfully" << std::endl;

        // 读取图像
        cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
        std::cout << "image-shape: " << image.rows << "x" << image.cols << std::endl;
        std::cout << "channel: " << image.channels() << std::endl;
        std::cout << "dims: " << image.dims << std::endl;
        if (image.empty()) {
        std::cerr << "Failed to load image: " << imagePath << std::endl;
        }

        // 执行推理
        std::cout << "Running inference..." << std::endl;
        auto results = detector.infer(image.data, image.cols, image.rows);

        // 输出检测结果
        std::cout << "Detection results: " << results.size() << " objects detected" << std::endl;

        for (size_t i = 0; i < results.size(); ++i) {
            const auto &det = results[i];
            std::cout << "Object " << i + 1 << ": "
                      << "class=" << det.class_id
                      << ", conf=" << det.confidence
                      << ", x=" << det.x
                      << ", y=" << det.y
                      << ", w=" << det.w
                      << ", h=" << det.h
                      << std::endl;
        }

        std::cout << "Test completed successfully!" << std::endl;
        return 0;

    } catch (const std::exception &e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}
