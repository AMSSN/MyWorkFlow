#include "utils.h"
// 全局变量
ConfigGlobal config_global = {}; // 初始化为默认值

void init_spdlog()
{
    try {
        // 创建控制台（终端）接收器
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(spdlog::level::info);  // 设置控制台日志等级
        console_sink->set_pattern("%l[%Y-%m-%d %H:%M:%S][%s%#:%!] %v"); // 设置日志格式

        // 创建文件接收器
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("logs/DailyLog.log", true);
        file_sink->set_level(spdlog::level::debug);    // 设置文件日志等级
        file_sink->set_pattern("%l[%Y-%m-%d %H:%M:%S][%s%#:%!] %v");        // 设置日志格式

        // auto qt_sink = std::make_shared<spdlog::sinks::qt_sink_mt>(ui->textEdit_log, "append");
        // qt_sink->set_level(spdlog::level::debug); // 设置文件日志等级
        // qt_sink->set_pattern("%l[%H:%M:%S] %v"); // 设置日志格式

        // 创建一个多重接收器的 logger
        // std::vector<spdlog::sink_ptr> sinks {console_sink, file_sink, qt_sink};
        std::vector<spdlog::sink_ptr> sinks {console_sink, file_sink};
        auto logger = std::make_shared<spdlog::logger>("multi_sink", sinks.begin(), sinks.end());

        // 设置全局 logger
        spdlog::set_default_logger(logger);
        // 设置全局日志等级为 debug，所有日志都会输出
        spdlog::set_level(spdlog::level::debug);
        // 开启日志刷新
        spdlog::flush_on(spdlog::level::info);
        // 示例日志
        spdlog::info("log class has been initialed !");
        // 手动刷新日志，确保写入到文件
        //spdlog::flush_every(std::chrono::seconds(1));  // 每秒刷新日志

    } catch (const spdlog::spdlog_ex &ex) {
        std::cout << "日志初始化失败: " << ex.what() << std::endl;
    }

}

void load_config_file(std::string config_file_path) {
    QFile file(QString::fromStdString(config_file_path));
    if (!file.open(QIODevice::ReadOnly)) {
        spdlog::error("can't open file {}", config_file_path);
    }
    QByteArray data = file.readAll();
    file.close();

    QJsonDocument doc = QJsonDocument::fromJson(data);
    if (doc.isNull()) {
        spdlog::error("JSON parse error");
    }
    // 成功打开
    spdlog::info("Load config file successful. parsing.......");
    QJsonObject obj = doc.object();
    //// 方式1：直接访问（键不存在返回 QJsonValue::Undefined）
    //QString name = obj["name"].toString();           // "张三"
    //int age = obj["age"].toInt();                    // 25
    //double salary = obj["salary"].toDouble();       // 15000.5
    //bool isVip = obj["isVip"].toBool();              // true
    //// 方式2：contains 检查 + value 带默认值（推荐）
    //QString name2 = obj.contains("name") ?
    //    obj["name"].toString() : "未知";
    //int age2 = obj.contains("age") ?
    //    obj["age"].toInt(-1) : -1;  // toInt 可传默认值
    //// 方式3：使用 value() 方法（更简洁）
    //QString name3 = obj.value("name").toString("未知");
    //int age3 = obj.value("age").toInt(-1);
    //double salary3 = obj.value("salary").toDouble(0.0);
    //// 键不存在时的处理
    //if (obj["notExist"].isUndefined()) {
    //    qDebug() << "键不存在";
    //}
    try {
        config_global.version = obj["version"].toString().toStdString();
        config_global.product_type = obj["product_type"].toString().toStdString();
        config_global.frame_gap = obj["frame_gap"].toInt();
        QJsonObject model_obj = obj["model"].toObject();
        config_global.model_path = model_obj["model_path"].toString().toStdString();
        config_global.conf = model_obj["confidence"].toDouble();
        config_global.infer_size = model_obj["infer_size"].toInt();
        config_global.postprocess_type = model_obj["postprocess_type"].toInt();
        config_global.last_update = obj["last_update"].toString().toStdString();
        spdlog::info("-------ConfigGlobal File-------");
        spdlog::info("version={}", config_global.version);
        spdlog::info("product_type={}", config_global.product_type);
        spdlog::info("frame_gap={}", config_global.frame_gap);
        spdlog::info("model_path={}", config_global.model_path);
        spdlog::info("confidence={}", config_global.conf);
        spdlog::info("infer_size={}", config_global.infer_size);
        spdlog::info("postprocess_type={}", config_global.postprocess_type);
        spdlog::info("last_update={}", config_global.last_update);
        spdlog::info("-------------------------------");
    }
    catch (const std::exception& e)
    {
        spdlog::warn("Parse config file error!");
    }
}


QImage Enutils::matToQimage(const cv::Mat& cvImage) {
    std::vector<uchar> imgBuf;
    imencode(".bmp", cvImage, imgBuf);

    QByteArray baImg((char*)imgBuf.data(), static_cast<int>(imgBuf.size()));
    QImage qimg;
    qimg.loadFromData(baImg, "BMP");
    return qimg;
}

QPixmap Enutils::matToPixmap(const cv::Mat& cvImage)
{
    return QPixmap::fromImage(matToQimage(cvImage));
}

cv::Mat Enutils::qimageToMat(const QImage& image) {
    QByteArray ba;
    QBuffer buffer(&ba);
    buffer.open(QIODevice::WriteOnly);
    image.save(&buffer, "BMP");
    std::vector<uchar> imgBuf;
    return imdecode(std::vector<uchar>(ba.begin(), ba.end()), cv::IMREAD_COLOR);
}

std::vector<uint8_t> Enutils::load_file(const std::string& file)
{
    // 配合trt_engine读取onnx
    std::ifstream in(file, std::ios::in | std::ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, std::ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0)
    {
        in.seekg(0, std::ios::beg);
        data.resize(length);

        in.read((char*)&data[0], length);
    }
    in.close();

    return data;
}

std::vector<unsigned char> Enutils::load_engine_file(const std::string& file)
{
    std::ifstream in(file, std::ios::in | std::ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, std::ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0)
    {
        in.seekg(0, std::ios::beg);
        data.resize(length);

        in.read((char*)&data[0], length);
    }
    in.close();
    return data;
}

int Enutils::vectorProduct(const std::vector<int>& vector)
{
    if (vector.empty())
        return 0;

    size_t product = 1;
    for (const auto& element : vector)
        product *= element;

    return product;
}

int Enutils::clip(int x, int min, int max)
{
    if (x > max)
        return max;
    if (x < min)
        return min;
    return x;
}

std::vector<YoloBox> Enutils::NMS(std::vector<YoloBox>& rawbox, float threshold, std::string modelname)
{
    if (rawbox.size() == 0)
        return rawbox;
    std::sort(rawbox.begin(), rawbox.end(), [](YoloBox w1, const YoloBox w2) { return w1.score > w2.score; });
    std::vector<bool> flag(rawbox.size(), false);
    for (size_t i = 0; i < rawbox.size(); i++)
    {
        if (flag[i])
            continue;
        for (size_t j = i + 1; j < rawbox.size(); j++)
        {
            if (IoU(rawbox[i], rawbox[j], modelname) > threshold)
                flag[j] = true;
        }
    }

    std::vector<YoloBox> ret;
    for (size_t i = 0; i < rawbox.size(); i++)
    {
        if (!flag[i])
            ret.push_back(rawbox[i]);
    }
    rawbox.clear();
    rawbox.assign(ret.begin(), ret.end());
    return ret;
}

float Enutils::IoU(cv::Rect& w1, cv::Rect& w2, std::string modelname)
{
    int xOverlap = std::max(0, std::min(w1.x + w1.width, w2.x + w2.width) - std::max(w1.x, w2.x));
    int yOverlap = std::max(0, std::min(w1.y + w1.height, w2.y + w2.height) - std::max(w1.y, w2.y));
    int intersection = xOverlap * yOverlap;
    int unio = w1.width * w1.height + w2.width * w2.height - intersection;
    float iou = 0;

    if (!modelname.compare("Union"))
        iou = (float)intersection / (float)unio;
    else if (!modelname.compare("Min"))
    {
        iou = (float)intersection / float((w1.width * w1.height < w2.width * w2.height) ? w1.width * w1.height : w2.width * w2.height);
    }
    return iou;
}

float Enutils::IoU(const YoloBox& w1, const YoloBox& w2, std::string modelname)
{
    cv::Rect rect1 = { w1.x1, w1.y1, w1.w(), w1.h() };
    cv::Rect rect2 = { w2.x1, w2.y1, w2.w(), w2.h() };
    return IoU(rect1, rect2, modelname);
}

void Enutils::mergeOverlapOutputs(std::vector<YoloBox>& boxes, std::vector<YoloBox>& boxes1, std::string iou_type, float iou_thr)
{
    boxes1.clear();

    while (boxes1.size() < boxes.size())
    {
        if (boxes1.size() > 0)
        {
            boxes.clear();
            boxes.assign(boxes1.begin(), boxes1.end());
            boxes1.clear();
        }

        for (size_t i = 0; i < boxes.size(); i++)
        {
            YoloBox box = boxes[i];
            float iou_m = 0.f;
            int index = -1;
            for (size_t j = 0; j < boxes1.size(); j++)
            {
                YoloBox box1 = boxes1[j];
                float iou = Enutils::IoU(box, box1, iou_type);
                if (iou > iou_m && box.label == box1.label)
                {
                    iou_m = iou;
                    index = j;
                }
            }
            //printf("i:%d iou_m:%.2f\n", i, iou_m);
            if (iou_m > iou_thr)
            {
                YoloBox& box1 = boxes1[index];
                int x1 = std::min(box.x1, box1.x1);
                int y1 = std::min(box.y1, box1.y1);
                int x2 = std::max(box.x2, box1.x2);
                int y2 = std::max(box.y2, box1.y2);
                float score = std::max(box.score, box1.score);

                box1.x1 = x1;
                box1.y1 = y1;
                box1.x2 = x2;
                box1.y2 = y2;
                box1.score = score;
            }
            else
            {
                boxes1.emplace_back(box);
            }
        }
    }
}


// cv::Mat PixmapToMat(const QPixmap& image) {
//     return qimageToMat(image.toImage());
// }
