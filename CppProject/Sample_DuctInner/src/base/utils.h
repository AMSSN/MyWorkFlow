#ifndef UTILS_H
#define UTILS_H
#include <iostream>
#include <QBuffer>
// log
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h> // For colored console logs
#include <spdlog/sinks/basic_file_sink.h>    // For file logging
#include <spdlog/sinks/qt_sinks.h>           // For QT widget logging

#include <time.h>
// ========== OpenCV 头文件 ==========
#include <opencv2/opencv.hpp>      // 主头文件（包含 cv::Mat, imencode, imdecode）
#include <opencv2/imgcodecs.hpp>   // imencode, imdecode 的显式头文件（可选，opencv.hpp 已包含）

// ========== Qt 头文件 ==========
#include <QImage>                  // QImage 类
#include <QPixmap>                 // QPixmap 类
#include <QByteArray>              // QByteArray 类
#include <QBuffer>                 // QBuffer 类（用于内存读写）
#include <QIODevice>               // QIODevice::WriteOnly 等枚举
#include <QJsonDocument>		   //json相关
#include <QJsonObject>
#include <QJsonArray>
#include <QFile>
#include <QJsonValue>

// 定义全局配置文件的结构体，对应config.json
struct ConfigGlobal
{
    std::string version = "1.0";
    std::string product_type = "";
    int frame_gap = 33; //30帧
    std::string model_path = "";
    float conf = 0.6;
    std::string last_update = "";
    double ng_num = 0;
    double ok_num = 0;
    double dpd_num = 0;
};
extern ConfigGlobal config_global;


void init_spdlog();
void load_config_file(std::string config_file_path);
QImage matToQimage(const cv::Mat& cvImage);
QPixmap matToPixmap(const cv::Mat& cvImage);
cv::Mat qimageToMat(const QImage& image);



#endif // UTILS_H
