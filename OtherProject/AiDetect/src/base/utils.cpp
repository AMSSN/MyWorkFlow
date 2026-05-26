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
        config_global.model_path = obj["model_path"].toString().toStdString();
        config_global.conf = obj["confidence"].toDouble();
        config_global.last_update = obj["last_update"].toString().toStdString();
        spdlog::info("-------ConfigGlobal File-------");
        spdlog::info("version={}", config_global.version);
        spdlog::info("product_type={}", config_global.product_type);
        spdlog::info("frame_gap={}", config_global.frame_gap);
        spdlog::info("model_path={}", config_global.model_path);
        spdlog::info("confidence={}", config_global.conf);
        spdlog::info("last_update={}", config_global.last_update);
        spdlog::info("-------------------------------");
    }
    catch (const std::exception& e)
    {
        spdlog::warn("Parse config file error!");
    }
}


inline QImage matToQimage(const cv::Mat& cvImage) {
    std::vector<uchar> imgBuf;
    imencode(".bmp", cvImage, imgBuf);

    QByteArray baImg((char*)imgBuf.data(), static_cast<int>(imgBuf.size()));
    QImage qimg;
    qimg.loadFromData(baImg, "BMP");
    return qimg;
}

inline QPixmap matToPixmap(const cv::Mat& cvImage)
{
    return QPixmap::fromImage(matToQimage(cvImage));
}

inline cv::Mat qimageToMat(const QImage& image) {
    QByteArray ba;
    QBuffer buffer(&ba);
    buffer.open(QIODevice::WriteOnly);
    image.save(&buffer, "BMP");
    std::vector<uchar> imgBuf;
    return imdecode(std::vector<uchar>(ba.begin(), ba.end()), cv::IMREAD_COLOR);
}

// cv::Mat PixmapToMat(const QPixmap& image) {
//     return qimageToMat(image.toImage());
// }
