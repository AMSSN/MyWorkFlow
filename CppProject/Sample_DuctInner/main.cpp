#include <QtWidgets/QApplication>
#include "src/base/utils.h"
#include "mainController.h"
#include <opencv2/core/utils/logger.hpp>

//int main(int argc, char* argv[])
//{
//    // 本地推理
//    std::string lensType = "HJ_6174_101";
//    std::cout << "lensType=" << lensType << std::endl;
//    return 0;
//}



int main(int argc, char* argv[])
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);
    init_spdlog();
    spdlog::info("Hello, World!");
    QApplication a(argc, argv);
    MainController w;
    w.show();
    return a.exec();
}