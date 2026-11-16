#include <iostream>
#include "src/StdThreadExample.h"
#include "src/AsyncExample.h"
int StdThreadExample::shared_value = 0;
std::mutex StdThreadExample::mtx;
int main() {
    std::cout << "Hello, World!" << std::endl;
    // StdThreadExample example;
    // example.run();
    AsyncExample::run();
    return 0;
}
