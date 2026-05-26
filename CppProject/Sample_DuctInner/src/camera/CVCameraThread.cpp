#include "CVCameraThread.h"

OpencvThreadedCamera::OpencvThreadedCamera(QObject* parent)
    : QThread(parent)
    , m_isRunning(0)
    , m_isPaused(0)
    , m_shouldStop(0)
    , m_cameraIndex(0)
    , m_isFileSource(false)
{
}

OpencvThreadedCamera::OpencvThreadedCamera(int cameraIndex, QObject* parent)
    : QThread(parent)
    , m_isRunning(0)
    , m_isPaused(0)
    , m_shouldStop(0)
    , m_cameraIndex(cameraIndex)
    , m_isFileSource(false)
{
}

OpencvThreadedCamera::OpencvThreadedCamera(const QString& source, QObject* parent)
    : QThread(parent)
    , m_isRunning(0)
    , m_isPaused(0)
    , m_shouldStop(0)
    , m_source(source)
    , m_cameraIndex(-1)
    , m_isFileSource(true)
{
}

OpencvThreadedCamera::~OpencvThreadedCamera()
{
    stop();
    wait();  // 等待线程结束
}

bool OpencvThreadedCamera::stream()
{
    QMutexLocker stateLocker(&m_stateMutex);

    if (m_isRunning.loadAcquire()) {
        return true;  // 已经在运行
    }

    // 打开摄像头
    if (m_isFileSource) {
        if (!m_cap.open(m_source.toStdString())) {
            emit errorOccurred(QString("Cannot open video source: %1").arg(m_source));
            return false;
        }
    }
    else {
        if (!m_cap.open(m_cameraIndex)) {
            emit errorOccurred(QString("Cannot open camera with index: %1").arg(m_cameraIndex));
            return false;
        }
    }

    // 设置摄像头参数
    m_cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    m_cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    m_cap.set(cv::CAP_PROP_FPS, 30);

    m_isRunning.storeRelease(1);
    m_isPaused.storeRelease(0);
    m_shouldStop.storeRelease(0);

    // 启动线程
    if (!isRunning()) {
        start(QThread::HighPriority);
    }
    else {
        // 如果线程已经存在但不在运行状态，唤醒它
        m_pauseCondition.wakeAll();
    }

    return true;
}

void OpencvThreadedCamera::pause()
{
    QMutexLocker stateLocker(&m_stateMutex);
    if (m_isRunning.loadAcquire()) {
        if (m_isPaused.loadAcquire()) {
            // 当前是暂停状态，执行恢复
            m_isPaused.storeRelease(0);
            m_pauseCondition.wakeAll();  // 唤醒等待的线程
        }
        else {
            // 当前是运行状态，执行暂停
            m_isPaused.storeRelease(1);
        }
    }
}

void OpencvThreadedCamera::stop()
{
    QMutexLocker stateLocker(&m_stateMutex);
    m_shouldStop.storeRelease(1);
    m_isRunning.storeRelease(0);
    m_isPaused.storeRelease(0);

    m_pauseCondition.wakeAll();  // 唤醒线程以便它能检测到停止标志

    // 释放摄像头
    if (m_cap.isOpened()) {
        m_cap.release();
    }
}

cv::Mat OpencvThreadedCamera::captured() const
{
    QMutexLocker locker(&m_frameMutex);
    return m_currentFrame.clone();
}

bool OpencvThreadedCamera::isRunningStatus() const
{
    return m_isRunning.loadAcquire() && !m_shouldStop.loadAcquire();
}

bool OpencvThreadedCamera::isPausedStatus() const
{
    return m_isPaused.loadAcquire();
}

//bool OpencvThreadedCamera::setProperty(int propId, double value)
//{
//    QMutexLocker stateLocker(&m_stateMutex);
//    if (m_cap.isOpened()) {
//        return m_cap.set(propId, value);
//    }
//    return false;
//}

//double OpencvThreadedCamera::getProperty(int propId) const
//{
//    QMutexLocker stateLocker(&m_stateMutex);
//    if (m_cap.isOpened()) {
//        return m_cap.get(propId);
//    }
//    return -1;
//}

void OpencvThreadedCamera::run()
{
    spdlog::info("OpencvThreadedCamera.run()");
    // 大循环检查是否需要停止
    while (!m_shouldStop.loadAcquire()) { 
        if (m_isPaused.loadAcquire()) { // 如果要暂停
            QMutexLocker locker(&m_stateMutex);
            // 使用条件变量等待，直到被唤醒
            m_pauseCondition.wait(&m_stateMutex);
            continue;  // 继续循环检查停止和暂停状态
        }
        
        cv::Mat frame; //当前帧
        {
            QMutexLocker stateLocker(&m_stateMutex); //锁
            if (m_cap.isOpened()) {
                spdlog::trace("OpencvThreadedCamera.isOpened()");
                if (m_cap.read(frame)) {
                    if (!frame.empty()) {
                        {
                            QMutexLocker frameLocker(&m_frameMutex);
                            m_currentFrame = frame.clone();
                            m_currentFrameIndex++;
                        }
                        // 发射新帧就绪信号
                        emit frameReady(frame);
                        emit frameIndex(m_currentFrameIndex);
                    }
                }
                else {
                    // 读取失败，可能需要处理错误情况
                    msleep(10);
                    continue;
                }
            }
            else {
                //spdlog::warn("OpencvThreadedCamera can't open!");
                msleep(10);
                continue;
            }
        }
        // 根据FPS调整捕获频率
        msleep(config_global.frame_gap); // 约30 FPS
    }
    // 线程退出前清理
    {
        QMutexLocker stateLocker(&m_stateMutex);
        if (m_cap.isOpened()) {
            m_cap.release();
        }
    }
}