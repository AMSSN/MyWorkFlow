# 标准cmake

参考https://modern-cmake-cn.github.io/Modern-CMake-zh_CN/chapters/basics.html

### 一、基础

#### 最低版本要求

这是每个 CMakeLists.txt 都必须包含的第一行

```cmake
cmake_minimum_required(VERSION 3.1)
```

#### 设置一个项目

项目名称就没有什么特别要注意的。在 CMake 3.9，可以通过DESCRIPTION 关键词来添加项目的描述。语言默认是`C CXX`。

```cmake
project(MyProject VERSION 1.0
                  DESCRIPTION "Very nice project"
                  LANGUAGES CXX)
```

#### 生成一个可执行文件

这里有一些语法需要解释。`one` 既是生成的可执行文件的名称，也是创建的 `CMake` 目标(target)的名称。紧接着的是源文件的列表，你想列多少个都可以。

```cmake
add_executable(one two.cpp three.h)
```

#### 生成一个库

你可以选择库的类型，可以是 `STATIC`,`SHARED`, 或者`MODULE`.如果你不选择它，CMake 将会通过 `BUILD_SHARED_LIBS` 的值来选择构建 STATIC 还是 SHARED 类型的库。

在下面的章节中你将会看到，你经常需要生成一个虚构的目标，也就是说，一个不需要编译的目标。例如，只有一个头文件的库。这被叫做 `INTERFACE` 库，这是另一种选择，和上面唯一的区别是后面不能有文件名。

你也可以用一个现有的库做一个 `ALIAS` 库，这只是给已有的目标起一个别名。这么做的一个好处是，你可以制作名称中带有 `::` 的库（你将会在后面看到）[3](https://modern-cmake-cn.github.io/Modern-CMake-zh_CN/chapters/basics.html#fn_3) 。

```cmake
add_library(one STATIC two.cpp three.h)
```

#### 添加目录

[`target_include_directories`](https://cmake.org/cmake/help/latest/command/target_include_directories.html) 为目标添加了一个目录。 `PUBLIC` 对于一个可执行文件目标没有什么含义；但对于库来说，它让 CMake 知道，任何链接到这个目标的目标也必须包含这个目录。其他选项还有 `PRIVATE`（只影响当前目标，不影响依赖），以及 `INTERFACE`（只影响依赖）。

```cmake
target_include_directories(one PUBLIC include)
```

#### 链接

[`target_link_libraries`](https://cmake.org/cmake/help/latest/command/target_link_libraries.html) 可能是 CMake 中最有用也最令人迷惑的命令。这个命令需要指定一个目标 `another`，并且在给出该目标的名字（ `another` ）后为此目标添加一个依赖 `one`。如果 CMake 项目中不存在名称为 `one` 的目标（没有定义该 target/目标），那它会直接添加名字为 `one` 的库到依赖中（一般而言，会去 `/usr`、CMake 项目指定寻找库的路径等所有能找的路径找到叫 `one` 的库——译者注）（这也是命令叫 `target_link_libraries` 的原因）。或者你可以给定一个库的完整路径，或者是链接器标志。最后再说一个有些迷惑性的知识：），经典的 CMake 允许你省略 `PUBLIC` 关键字，但是你在目标链中省略与不省略混用，那么 CMake 会报出错误。

只要记得在任何使用目标的地方都指定关键字，那么就不会有问题。

```cmake
add_library(another STATIC another.cpp another.h)
target_link_libraries(another PUBLIC one)
# 生成一个叫another的静态库，然后为其添加one(也是一个库(Library))的依赖。
```

#### 汇总

```cmake
cmake_minimum_required(VERSION 3.8)
# 必写-最低版本
project(Calculator LANGUAGES CXX)
# 必写-项目名称
add_library(calclib STATIC src/calclib.cpp include/calc/lib.hpp)
# 生成calclib库
target_include_directories(calclib PUBLIC include)
# calclib库包含的头文件
target_compile_features(calclib PUBLIC cxx_std_11)
# 设置编译版本？
add_executable(calc apps/calc.cpp)
# 生产可执行文件calc
target_link_libraries(calc PUBLIC calclib)
# calc依赖calclib
```















