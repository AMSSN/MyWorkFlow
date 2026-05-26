#ifndef MAIN_CONTROLLER_H
#define MAIN_CONTROLLER_H
#include <filesystem>
#include <vector>
#include <future>
#include <fstream>
#include <iostream>
#include "ui_demowindow.h"

#include <QMainWindow>
#include <QTimer>
#include <QRandomGenerator>
#include "src/base/utils.h"
#include "src/camera/CVCameraThread.h"
// extern std::string InputMode;
// extern int PictureNum;

class MainController : public QMainWindow
{
    Q_OBJECT
public:
    MainController(QMainWindow* parent = nullptr);
    ~MainController();
    double SampleNumber = 0;
    void beautifyUI();
    //void closeEvent(QCloseEvent* e);
    //void load_configfile(std::string file_path = "GlobalConfig.json");
    //void load_localimg();
    //int init_SDK_flag = -1;
    //QVector<double> chart_data; // 统计图数据
    //Sql sql;

    //YoloDetect* yoloDetector;
    OpencvThreadedCamera* camThread;
signals:
    void finished();


private:
    //void setQSS();
    Ui::MainWindow* ui;


private slots:
    void start_detect();
    void pause_detect();
    void stop_detect();
    void count_sample();
    void Image_process(const cv::Mat img);
    //void change_widget_to_detect();
    //void change_widget_to_data();
    //void change_widget_to_setting();
    //void change_algorithm_stage(bool stage);
    //void change_localimg_stage(bool stage);
    //void set_localimg_path();
    //// detect页面
    //void show_detect_result_in_listwidget(cv::Mat res, int index, int num);
    //// void updata_detect_processbar(int number);
    //// data页面
    //void select_data_from_sql();
    //void export_data_from_sql();
    //void Image_process(QImage img);
    //void temp1_soft_trigger();

};

#endif // MAIN_CONTROLLER_H
