#ifndef CVCAMERATHREAD_H
#define CVCAMERATHREAD_H
#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include <QAtomicInt>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include "../base/utils.h"
class OpencvThreadedCamera : public QThread
{
    Q_OBJECT
public:
    explicit OpencvThreadedCamera(QObject* parent = nullptr);//默认构造
    explicit OpencvThreadedCamera(int cameraIndex, QObject* parent = nullptr); // USB摄像头
    explicit OpencvThreadedCamera(const QString& source, QObject* parent = nullptr); //网络摄像头source是网址
    virtual ~OpencvThreadedCamera();

    // 控制接口
    bool stream(); //启动获取视频流
    void pause(); // 暂停/恢复暂停
    //void resumeCamera(); //恢复暂停
    void stop(); //停止

    // 获取图像
    cv::Mat captured() const; // 线程安全获取最新帧

    // 状态查询
    bool isRunningStatus() const;
    bool isPausedStatus() const;

    // 摄像头属性设置
    //bool setProperty(int propId, double value);
    //double getProperty(int propId) const;

signals:
    void frameReady(const cv::Mat& frame);  // 新帧就绪信号
    void frameIndex(const double index);  // 当前帧序号
    void errorOccurred(const QString& error);  // 错误信号

protected:
    virtual void run() override;

private:
    mutable QMutex m_frameMutex;
    mutable QMutex m_stateMutex;
    QWaitCondition m_pauseCondition;

    cv::VideoCapture m_cap;
    cv::Mat m_currentFrame;

    QAtomicInt m_isRunning;
    QAtomicInt m_isPaused;
    QAtomicInt m_shouldStop;

    QString m_source; // 网络摄像头地址
    int m_cameraIndex; // 摄像头设备
    double m_currentFrameIndex = 0; //当前帧序号
    bool m_isFileSource; // 区分网络摄像头还是物理设备
};

#endif