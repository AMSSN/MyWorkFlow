#ifndef LONGMAT_H
#define LONGMAT_H

#include <opencv2/opencv.hpp>
#include <vector>

class LongMat {
public:
    LongMat() = default;
    ~LongMat() noexcept = default;

    // 根据输入图像，按规则裁切后拼接成一副图像
    cv::Mat gen_stride_mat(const cv::Mat& image);

    // 将目标图上的bbox映射还原到原图的位置
    cv::Rect mapping_box(const cv::Rect& bbox) const;

    // Get/Set 方法
    int get_m_Stride() const { return m_Stride; }
    void set_m_Stride(int x) { m_Stride = x; }

    int get_m_GapFill() const { return m_GapFill; }
    void set_m_GapFill(int x) { m_GapFill = x; }

    int get_m_Overlap() const { return m_Overlap; }
    void set_m_Overlap(int x) { m_Overlap = x; }

private:
    cv::Mat m_orinal_Mat;   // 原图(保留用于尺寸参考)
    cv::Mat m_target_Mat;   // 拼接后的目标图

    int m_Stride = 200;     // 裁切步长(每份的核心高度)
    int m_GapFill = 10;     // 左右扩充填充像素
    int m_Overlap = 20;     // 裁切重叠长度(注意: 当前横向拼接逻辑中, 
    // Overlap主要用于纵向采样冗余, 
    // 若纯横向等分拼接可不使用, 保留字段供扩展)

// ---- 内部状态(由gen_stride_mat自动计算) ----
    int m_orig_w = 0;       // 原图宽
    int m_orig_h = 0;       // 原图高
    int m_slice_count = 0;  // 实际切片数量
    int m_cell_w = 0;       // 每个切片在目标图中的宽度(含GapFill)
    int m_cell_h = 0;       // 每个切片在目标图中的高度(=m_Stride)
};

#endif // LONGMAT_H