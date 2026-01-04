//
// Created by M on 2026/1/4.
//

#ifndef SAMPLE2_THREAD_STDTHREADEXAMPLE_H
#define SAMPLE2_THREAD_STDTHREADEXAMPLE_H

#include <iostream>
#include <thread>
#include <chrono>
#include <mutex>
#include <vector>

class StdThreadExample {
private:
    static int shared_value;
    static std::mutex mtx; // 保护共享数据

public:
    // 示例1: 线程执行普通函数
    static void threadFunction(int id) {
        for (int i = 0; i < 5; ++i) {
            std::lock_guard<std::mutex> lock(mtx);
            std::cout << "Thread " << id << ": " << i << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

     // 示例2: 演示线程传参（值传递、引用传递）
     /*
     std::thread 构造时，参数默认按值拷贝到线程内部。
     std::string s = "Ref Example";
     std::thread t(passArguments, 100, std::ref(s)); // 否则会拷贝 s
     */
    static void passArguments(int val, const std::string &name) {
        std::cout << "Passed - Name: " << name << ", Value: " << val << std::endl;
    }

    // 示例3: 共享数据竞争与互斥锁保护
     /*
     典型的竞态条件场景：多个线程同时对 shared_value 执行 ++ 操作（读-改-写）。
     无锁时结果不确定：可能小于 5000（5 线程 × 1000 次）。
     加锁后保证原子性：每次 ++ 操作都在临界区内完成，最终结果一定是 5000。
     */
    static void incrementSharedValue(int id) {
        for (int i = 0; i < 1000; ++i) {
            std::lock_guard<std::mutex> lock(mtx);
            ++shared_value;
        }
        std::cout << "Thread " << id << " finished. Shared value = " << shared_value << std::endl;
    }

    // 示例4: 主运行函数 —— 展示多个线程使用
    void run() {
        std::cout << "=== Starting Thread Examples ===" << std::endl;

        // 1. 启动多个线程执行相同函数
        std::vector<std::thread> threads;
        for (int i = 1; i <= 3; ++i) {
            threads.emplace_back(threadFunction, i);
        }

        // 等待所有线程完成
        for (auto &t: threads) {
            t.join();
        }

        std::cout << "\n=== Passing Arguments ===" << std::endl;
        std::thread t1(passArguments, 42, "WorkerThread");
        t1.join();

        std::cout << "\n=== Race Condition & Mutex Protection ===" << std::endl;
        std::vector<std::thread> inc_threads;
        for (int i = 1; i <= 5; ++i) {
            inc_threads.emplace_back(incrementSharedValue, i);
        }

        for (auto &t: inc_threads) {
            t.join();
        }

        std::cout << "Final shared value: " << shared_value << std::endl;
    }
};


#endif //SAMPLE2_THREAD_STDTHREADEXAMPLE_H
