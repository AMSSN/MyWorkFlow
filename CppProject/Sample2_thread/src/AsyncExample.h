//
// Created by M on 2026/1/4.
//
// AsyncExample.h
#ifndef ASYNC_EXAMPLE_H
#define ASYNC_EXAMPLE_H

#include <iostream>
#include <future>
#include <chrono>
#include <thread>
#include <vector>
#include <stdexcept>
#include <cmath>

class AsyncExample {
public:
    // 1. 简单异步任务：计算平方
    static int square(int x) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200)); // 模拟耗时
        return x * x;
    }

    // 2. 可能抛出异常的任务
    static double divide(double a, double b) {
        if (b == 0.0) {
            throw std::invalid_argument("Division by zero!");
        }
        return a / b;
    }

    // 3. 递归任务：斐波那契（仅用于演示，实际应避免递归 async）
    static long long fibonacci(int n) {
        if (n <= 1) return n;
        auto f1 = std::async(std::launch::async, fibonacci, n - 1);
        auto f2 = std::async(std::launch::async, fibonacci, n - 2);
        return f1.get() + f2.get();
    }

    // 4. 并行求和：将大数组分段并行计算
    static long long parallelSum(const std::vector<long long>& data, size_t num_threads = 4) {
        size_t size = data.size();
        if (size == 0) return 0;

        size_t chunk_size = size / num_threads;
        std::vector<std::future<long long>> futures;

        for (size_t i = 0; i < num_threads; ++i) {
            size_t start = i * chunk_size;
            size_t end = (i == num_threads - 1) ? size : start + chunk_size;

            futures.push_back(
                std::async(std::launch::async, [&data, start, end]() {
                    long long sum = 0;
                    for (size_t j = start; j < end; ++j) {
                        sum += data[j];
                    }
                    return sum;
                })
            );
        }

        long long total = 0;
        for (auto& fut : futures) {
            total += fut.get(); // 阻塞等待结果
        }
        return total;
    }

    // 5. 演示启动策略差异
    static void demonstrateLaunchPolicy() {
        auto start = std::chrono::high_resolution_clock::now();

        // deferred：延迟执行（直到 get() 被调用）
        auto deferred_task = std::async(std::launch::deferred, []() {
            std::cout << "Deferred task running in thread: "
                      << std::this_thread::get_id() << std::endl;
            return 42;
        });

        auto async_task = std::async(std::launch::async, []() {
            std::cout << "Async task running in thread: "
                      << std::this_thread::get_id() << std::endl;
            return 100;
        });

        std::cout << "Main thread continues immediately...\n";

        // 此时 async_task 可能已在后台运行，deferred_task 尚未启动
        auto mid = std::chrono::high_resolution_clock::now();
        auto mid_ms = std::chrono::duration_cast<std::chrono::milliseconds>(mid - start).count();
        std::cout << "Time before get(): " << mid_ms << " ms\n";

        int d = deferred_task.get(); // 此刻才执行！
        int a = async_task.get();    // 可能已执行完，直接取结果

        auto end = std::chrono::high_resolution_clock::now();
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Total time: " << total_ms << " ms, Results: deferred=" << d << ", async=" << a << "\n";
    }

    // 6. 超时控制示例（C++11 不支持 future::wait_for 超时获取值，但可检查状态）
    static void demonstrateTimeout() {
        auto fut = std::async(std::launch::async, []() {
            std::this_thread::sleep_for(std::chrono::seconds(3));
            return "Result after 3s";
        });

        // 等待最多 1 秒
        std::future_status status;
        do {
            status = fut.wait_for(std::chrono::milliseconds(500));
            if (status == std::future_status::timeout) {
                std::cout << "Still waiting... (timeout)\n";
            } else if (status == std::future_status::ready) {
                std::cout << "Result: " << fut.get() << "\n";
                break;
            }
        } while (status != std::future_status::ready);
    }

    // 主运行函数
    static void run() {
        std::cout << "=== std::async Learning Examples ===\n\n";

        // 示例1：基本用法
        /*
        get() 只能调用一次。future 是 move-only，取完值就空了；需要多次共享结果请用 std::shared_future。
         */
        std::cout << "[1] Basic async task:\n";
        auto fut1 = std::async(square, 10);
        std::cout << "10^2 = " << fut1.get() << "\n\n";

        // 示例2：异常处理
        std::cout << "[2] Exception handling:\n";
        auto fut2 = std::async(divide, 10.0, 0.0);
        try {
            std::cout << "Result: " << fut2.get() << "\n";
        } catch (const std::exception& e) {
            std::cout << "Caught exception: " << e.what() << "\n";
        }
        std::cout << "\n";

        // 示例3：并行求和
        std::cout << "[3] Parallel sum:\n";
        std::vector<long long> big_data(1000000, 1); // 一百万个 1
        auto sum_fut = std::async(parallelSum, std::cref(big_data), 4);
        std::cout << "Parallel sum result: " << sum_fut.get() << "\n\n";

        // 示例4：启动策略对比
        std::cout << "[4] Launch policy demo:\n";
        demonstrateLaunchPolicy();
        std::cout << "\n";

        // 示例5：超时等待（非阻塞检查）
        std::cout << "[5] Timeout demonstration (will take ~3s):\n";
        demonstrateTimeout();
        std::cout << "\n";

        std::cout << "=== All examples completed ===\n";
    }
};

#endif // ASYNC_EXAMPLE_H