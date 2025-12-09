# My workflow
I'm an industrial-oriented computer vision algorithm engineer. 



国内阿里云源安装torch-GPU

`pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 -f https://mirrors.aliyun.com/pytorch-wheels/cu118`

或者将torch官网的网址替换为

https://mirror.nju.edu.cn/pytorch/whl/cu126



yolov5的环境：
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 -f https://mirrors.aliyun.com/pytorch-wheels/cu117



git设置和取消代理

git config --global http.proxy http://127.0.0.1:7890

git config --global https.proxy http://127.0.0.1:7890

git config --global --unset http.proxy

git config --global --unset https.proxy



TODO

C++要写一个demo，能运行yolov5/v8的检测、分割、实例分割模型。

能运行detr的检测模型。

