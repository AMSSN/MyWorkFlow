#include "LongMat.h"
#include <algorithm>
#include <cmath>

cv::Mat LongMat::gen_stride_mat(const cv::Mat& image) {
    if (image.empty()) {
        m_target_Mat = cv::Mat();
        return m_target_Mat;
    }

    // 保存原图信息
    m_orinal_Mat = image.clone();
    m_orig_w = image.cols;
    m_orig_h = image.rows;
    m_cell_h = m_Stride;
    m_cell_w = m_orig_w + 2 * m_GapFill;

    // 计算需要切多少片(向上取整)
    m_slice_count = static_cast<int>(std::ceil(
        static_cast<double>(m_orig_h) / m_Stride));

    // 创建目标图: 高度=m_Stride, 宽度=cell_w * slice_count
    // 初始化为黑色(填充不足部分自动为0)
    int target_w = m_cell_w * m_slice_count;
    int target_h = m_Stride;
    m_target_Mat = cv::Mat::zeros(target_h, target_w, image.type());

    for (int i = 0; i < m_slice_count; ++i) {
        // 计算当前切片在原图中的y起始位置
        int y_start = i * m_Stride;
        int y_end = std::min(y_start + m_Stride, m_orig_h);
        int roi_h = y_end - y_start;

        // 提取ROI
        cv::Mat roi = image(cv::Rect(0, y_start, m_orig_w, roi_h));

        // 在目标图中确定放置位置(左右各留m_GapFill)
        int dst_x = i * m_cell_w + m_GapFill;
        int dst_y = 0;

        // 拷贝到目标图(不足m_Stride高度的部分保持黑色)
        roi.copyTo(m_target_Mat(cv::Rect(dst_x, dst_y, m_orig_w, roi_h)));
    }

    return m_target_Mat;
}

cv::Rect LongMat::mapping_box(const cv::Rect& bbox) const {
    if (m_slice_count <= 0 || m_cell_w <= 0) {
        return cv::Rect(0, 0, 0, 0);
    }

    // 1. 确定bbox落在第几个切片中
    int slice_idx = bbox.x / m_cell_w;
    slice_idx = std::max(0, std::min(slice_idx, m_slice_count - 1));

    // 2. 计算bbox在该切片内的局部x偏移(去掉左侧GapFill)
    int local_x = bbox.x - slice_idx * m_cell_w - m_GapFill;

    // 3. 映射回原图坐标
    int orig_x = local_x;           // 原图x = 局部x(因为切片是从x=0开始的)
    int orig_y = slice_idx * m_Stride + bbox.y;

    // 4. 裁剪到原图有效范围内
    int orig_w = std::min(bbox.width, m_orig_w - orig_x);
    int orig_h = std::min(bbox.height, m_orig_h - orig_y);

    // 防止越界
    orig_x = std::max(0, orig_x);
    orig_y = std::max(0, orig_y);
    orig_w = std::max(0, orig_w);
    orig_h = std::max(0, orig_h);

    return cv::Rect(orig_x, orig_y, orig_w, orig_h);
}