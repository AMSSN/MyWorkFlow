#include <iostream>
#include "LongMat.h"
int main() {
    std::cout << "Hello, World!" << std::endl;
    LongMat lm;
    lm.set_m_Stride(960);
    lm.set_m_GapFill(10);

    cv::Mat long_img = cv::imread("E:\\BJBImage\\crop\\20260526104854355_1_T10_0.jpg");
    cv::Mat stitched = lm.gen_stride_mat(long_img);
    cv::imwrite("stitched.png",stitched);

    // 假设在拼接图上检测到 bbox
    cv::Rect det_box(350, 50, 80, 120);
    cv::Rect orig_box = lm.mapping_box(det_box);

    std::cout << "ori image position: " << orig_box << std::endl;


    return 0;
}
