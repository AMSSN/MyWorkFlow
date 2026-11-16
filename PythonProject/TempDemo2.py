import cv2
import os
import numpy as np
import shutil
from pathlib import Path



# def depth_fusion(imgs):
#     """基础景深融合实现"""
#     # 1. 计算清晰度图（拉普拉斯，基于灰度亮度）
#     laplacians = [np.abs(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F)) for img in imgs]
#     # 2. 归一化权重图
#     weights_sum = np.sum(laplacians, axis=0)
#     weights_sum[weights_sum == 0] = 1e-5  # 避免除零
#     normalized_weights = [w / weights_sum for w in laplacians]
#     # 3. 加权融合
#     result = np.zeros_like(imgs[0], dtype=np.float32)
#     for img, weight in zip(imgs, normalized_weights):
#         result += img * weight[..., None]  # (H,W,1) 广播到 (H,W,3)
#     return result.astype(np.uint8)


if __name__ == '__main__':


    pass

