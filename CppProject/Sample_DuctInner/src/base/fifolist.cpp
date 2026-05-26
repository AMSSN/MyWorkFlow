#include "fifolist.h"
#include<QDebug>
#include <QDateTime>
#include <time.h>
FifoList::FifoList(QObject *parent)    : QObject(parent)
{
    m_maxSize = 400;
    isExit=false;
}

void FifoList::enQueue(fifoNode *node)
{
    QMutexLocker locker(&m_mutex);
    while (m_queue.size() >= m_maxSize)
    {
        m_notFull.wait(&m_mutex);
        if(isExit)
        {
            qDebug()<<"enqueue break";
            return;
        }
    }
    m_queue.push(node);
    m_notEmpty.wakeOne();
}
fifoNode * FifoList::deQueue()
{
    QMutexLocker locker(&m_mutex);
    while (m_queue.empty())
    {

        m_notEmpty.wait(&m_mutex);
        if(isExit)
        {
            qDebug()<<"enqueue break";
            return nullptr;
        }
    }
    fifoNode *item = m_queue.front();
    m_queue.pop();
    m_notFull.wakeOne();
    return item;
}

bool FifoList::isEmpty()
{
    QMutexLocker locker(&m_mutex);
    return m_queue.empty();
}

bool FifoList::isFull()
{
    QMutexLocker locker(&m_mutex);
    return m_queue.size() ==m_maxSize;
}

int FifoList::getSize()
{
    QMutexLocker locker(&m_mutex);
    return m_queue.size();
}

void FifoList::destroyQueue()
{
    notify_all();
    QMutexLocker locker(&m_mutex);
    while(!m_queue.empty())
    {
        fifoNode * temp;
        temp=m_queue.front();
        if(temp !=nullptr)
        {
            delete temp;
        }
        m_queue.pop();
    }
}

void FifoList::notify_all()
{
    QMutexLocker locker(&m_mutex);
    m_notEmpty.notify_all();
    m_notFull.notify_all();
}

