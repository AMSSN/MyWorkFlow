#include "SimpleYolo5.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>

// 简单的BMP图像加载函数（不使用OpenCV）
bool loadBMPImage(const std::string &filePath, std::vector<unsigned char> &imageData, int &width, int &height) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open BMP file: " << filePath << std::endl;
        return false;
    }
    
    // BMP文件头结构
    struct BMPHeader {
        unsigned char signature[2];
        unsigned int fileSize;
        unsigned int reserved;
        unsigned int pixelOffset;
    } header;
    
    // 读取文件头
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    
    // 检查BMP签名
    if (header.signature[0] != 'B' || header.signature[1] != 'M') {
        std::cerr << "Not a valid BMP file" << std::endl;
        return false;
    }
    
    // 跳到信息头
    file.seekg(14); // 跳过文件头
    
    // 读取图像宽度和高度
    unsigned int infoSize;
    file.read(reinterpret_cast<char*>(&infoSize), 4);
    file.read(reinterpret_cast<char*>(&width), 4);
    file.read(reinterpret_cast<char*>(&height), 4);
    
    // 读取位深度
    short planes, bitDepth;
    file.read(reinterpret_cast<char*>(&planes), 2);
    file.read(reinterpret_cast<char*>(&bitDepth), 2);
    
    // 目前只支持24位BMP
    if (bitDepth != 24) {
        std::cerr << "Only 24-bit BMP files are supported" << std::endl;
        return false;
    }
    
    // 计算行大小（BMP每行是4字节对齐的）
    int rowSize = ((width * 24 + 31) / 32) * 4;
    int pixelSize = width * height * 3;
    
    // 读取像素数据
    file.seekg(header.pixelOffset);
    
    // 分配缓冲区（注意BMP是BGR格式）
    imageData.resize(pixelSize);
    
    // 读取数据（需要处理行填充）
    for (int y = 0; y < height; ++y) {
        file.read(reinterpret_cast<char*>(imageData.data() + y * width * 3), width * 3);
        // 跳过行填充
        file.seekg(rowSize - width * 3, std::ios::cur);
    }
    
    file.close();
    std::cout << "Loaded BMP image: " << width << "x" << height << std::endl;
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <engine_path> [image_path.bmp]" << std::endl;
        return 1;
    }
    
    std::string enginePath = argv[1];
    std::string imagePath = (argc > 2) ? argv[2] : "test_image.bmp";
    
    try {
        // 初始化检测器
        SimpleYolo5::Detector detector(enginePath, 0.3f, 0.45f);
        
        if (!detector.isLoaded()) {
            std::cerr << "Failed to load detector" << std::endl;
            return 1;
        }
        
        std::cout << "Detector initialized successfully" << std::endl;
        
        // 加载测试图像
        std::vector<unsigned char> imageData;
        int width, height;
        if (!loadBMPImage(imagePath, imageData, width, height)) {
            std::cerr << "Failed to load test image" << std::endl;
            return 1;
        }
        
        // 执行推理
        std::cout << "Running inference..." << std::endl;
        auto results = detector.infer(imageData.data(), width, height);
        
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
