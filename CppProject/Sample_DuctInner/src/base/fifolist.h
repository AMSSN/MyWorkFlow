#ifndef FIFOLIST_H
#define FIFOLIST_H

#include <QObject>
#include <QMutex>
#include <QWaitCondition>
#include "singleton.h"
#include <queue>
#include <opencv2/opencv.hpp>
#include <mutex> // 互斥量
#include <condition_variable> // 条件变量
// #include "./helper/SettingsHelper.h"

class fifoNode
{
public:
    // 一张图片的相关信息
    QString time;
    int index;
    int batch_index;
    int row_index;
    int column_index;
    cv::Mat image;
};


class FifoList : public QObject
{
    // Q_OBJECT
public:
    SINGLETONG(FifoList)
    explicit FifoList(QObject *parent = nullptr);
    virtual void enQueue(fifoNode* node);
    virtual fifoNode* deQueue();
    bool isEmpty();
    bool isFull();
    int getSize();
    void destroyQueue();
    void notify_all();
    bool isExit;
    std::queue<fifoNode *> m_queue;
    int m_maxSize;

    std::mutex mutex_; // 互斥量，保护队列缓冲区和队列大小
    std::condition_variable not_full_; // 条件变量，当队列满时等待
    std::condition_variable not_empty_; // 条件变量，当队列空时等待

    QMutex m_mutex;
    QWaitCondition m_notFull;
    QWaitCondition m_notEmpty;

private:

// signals:
};

#endif // FIFOLIST_H
