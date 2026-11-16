#ifndef UTILS_H
#define UTILS_H
#include <iostream>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cmath>
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
    int infer_size = 640;
    int postprocess_type = 2;
    std::string last_update = "";
    double ng_num = 0;
    double ok_num = 0;
    double dpd_num = 0;
};
extern ConfigGlobal config_global;

struct ConfigModel
{
    std::string model_type;
    std::string model_path;
    int batch_size;
    int infer_size;
    bool is_rgb;
    std::vector<float> means;
    std::vector<float> stds;
    float conf_threshold;
    float nms_threshold;
    int postproc_type;
    int crop_size;
    int stride;

    ConfigModel()
    {
        model_type = "";
        model_path = "";
        batch_size = 1;
        infer_size = -1;
        is_rgb = false;
        means = { 0.f, 0.f, 0.f };
        stds = { 255.f, 255.f, 255.f };
        conf_threshold = -1;
        nms_threshold = -1;
        postproc_type = -1;
        crop_size = infer_size;
        stride = infer_size;
    }
};
typedef struct _yolo_mask
{
    _yolo_mask() {
        label = 0;
        score = 0.f;
    }
    int label;         // 结果类别id
    float score;       // 结果置信度
    cv::Rect box;      // 矩形框
    cv::Mat boxMask;   // 矩形框内mask，节省内存空间和加快速度
}YoloMask;
typedef struct _yolo_box
{
    _yolo_box()
    {
        x1 = 0;
        y1 = 0;
        x2 = 0;
        y2 = 0;
        score = 0.f;
        label = 0;
    }
    int x1;
    int y1;
    int x2;
    int y2;
    float score;
    int label;

    int w() const { return x2 - x1; }
    int h() const { return y2 - y1; }
    int area() const { return w() * h(); }
} YoloBox;


void init_spdlog();
void load_config_file(std::string config_file_path);



// 常用工具
namespace Enutils
{
    QImage matToQimage(const cv::Mat& cvImage);
    QPixmap matToPixmap(const cv::Mat& cvImage);
    cv::Mat qimageToMat(const QImage& image);
    //std::string convert_date_to_yyyymmdd();
    std::vector<uint8_t> load_file(const std::string& file);
    std::vector<unsigned char> load_engine_file(const std::string& file);
    int vectorProduct(const std::vector<int>& vector);
    int clip(int x, int min, int max);
    //void convert_outsegs_to_mask(std::vector<YoloMask>& output_segs, cv::Mat& mask);
    //void generate_colors(std::vector<cv::Scalar>& colors, int num);
    std::vector<YoloBox> NMS(std::vector<YoloBox>& rawbox, float threshold, std::string modelname = "Union");
    float IoU(cv::Rect& w1, cv::Rect& w2, std::string modelname = "Union");
    float IoU(const YoloBox& w1, const YoloBox& w2, std::string modelname = "Union");
    void mergeOverlapOutputs(std::vector<YoloBox>& boxes, std::vector<YoloBox>& boxes1, std::string iou_type, float iou_thr);
    //std::string rand_str(int len);

}
//// halcon相关的工具
//namespace HalconUtils {
//    HalconCpp::HObject __Mat1ToHObject__(const cv::Mat& image);
//    cv::Mat HImageToMat(const HalconCpp::HImage& hImg);
//    HalconCpp::HObject MatToHObject(const cv::Mat& image);
//}
//// json相关的工具
//namespace JsonConfig {
//
//    int loadConfigFile(const std::string file_name, ConfigGlobal& config_global);
//
//    int loadConfigModelParams(cv::FileStorage& fs, std::string pos_name, std::string model_name, ConfigModel& config_model);
//
//    void printConfig(ConfigGlobal config_global);
//}




#endif // UTILS_H
