import os
from pathlib import Path

import cv2
import numpy as np


class HalconConvert:
    """
    halconTool软件语义分割生成的mask图片，处理后生成yolo格式的检测box型标签，和mask型标签
    """

    def __init__(self, mode, root_folder):
        self.mode = mode
        self.root_folder = root_folder
        self.image_path_list = []
        self.image_name_list = []
        self.image_suffix_list = []
        # 递归遍历文件夹
        for foldername, subfolders, filenames in os.walk(root_folder):
            for filename in filenames:
                if filename.lower().endswith(('.png', '.jpg', '.bmp')):  # 常见图片格式
                    file_path = os.path.join(foldername, filename)
                    try:
                        directory = os.path.dirname(file_path)  # 获取目录
                        filename = os.path.basename(file_path)  # 获取文件名（含后缀）
                        name, ext = os.path.splitext(filename)  # 分离文件名和后缀
                        self.image_path_list.append(file_path)
                        self.image_name_list.append(name)
                        self.image_suffix_list.append(ext)
                    except Exception as e:
                        print(f"无法读取图片 {file_path}，错误：{e}")
        self.save_folser = "labels"

    def check_src_img(self):
        """
        检查源图片的合法性，
        Returns: 是否合格
        """
        for file in self.image_path_list:
            try:
                image = cv2.imread(file)
            except Exception as e:
                print(f"无法读取图片 {file}，错误：{e}")

    def gen_yolo_labels(self):
        """生成标签"""
        if self.mode == "mask":
            self.gen_yolo_masklabel()
        elif self.mode == "box":
            self.gen_yolo_boxlabel()
        else:
            pass

    def gen_yolo_masklabel(self):
        class_index_list = []
        for i in range(len(self.image_name_list)):
            annotated_information = []
            image = cv2.imread(self.image_path_list[i], cv2.IMREAD_GRAYSCALE)
            shape = image.shape  # HWC
            max_class_num = np.max(image)
            for c in range(max_class_num + 1):
                class_index_list.append(c)
                # 创建掩膜，标记在范围内的像素
                mask = cv2.inRange(image, c + 1, c + 1)
                # # 获取对应区域的像素值
                # pixels_in_range = image[mask != 0]
                # # 如果你想提取这些像素所在的区域图像
                # result_image = cv2.bitwise_and(image, image, mask=mask)

                # _, binary_image = cv2.threshold(image, c + 1, c + 1, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    _contours = {str(c): contours}
                    annotated_information.append(_contours)
                # if contours:
                #     self.annotated_information[str(c+1)] = contours
            # 阈值分割
            _, binary_image = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY)
            # 寻找轮廓
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 获取输出文件名
            if not os.path.isdir(self.save_folser):
                os.makedirs(self.save_folser)
            output_file = os.path.join(self.save_folser, f"{self.image_name_list[i]}.txt")
            # 写入边界框坐标到txt文件
            with open(output_file, 'w') as f:
                for _contours in annotated_information:
                    for key, value in _contours.items():
                        cls = key + " "
                        for box in value:  # 有多少个目标框
                            points = ""
                            for point in box:  # 有多少个点
                                x = round(point[0][0] / shape[1], 3)
                                y = round(point[0][1] / shape[0], 3)
                                points = points + str(x) + " " + str(y) + " "
                            f.write(cls + points + "\n")
            print(f"已保存边界框坐标到：{output_file}")
        print(f"cls_index={set(class_index_list)}")

    def gen_yolo_boxlabel(self):
        class_index_list = []
        for i in range(len(self.image_name_list)):
            self.annotated_information = []
            image = cv2.imread(self.image_path_list[i], cv2.IMREAD_GRAYSCALE)
            shape = image.shape  # HWC
            max_class_num = np.max(image)
            for c in range(max_class_num + 1):
                mask = cv2.inRange(image, c + 1, c + 1)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    x1, y1, w1, h1 = cv2.boundingRect(contour)
                    x = round((x1 + (w1 / 2)) / shape[1], 3)
                    y = round((y1 + (h1 / 2)) / shape[0], 3)
                    w = round(w1 / shape[1], 3)
                    h = round(h1 / shape[0], 3)
                    _contours = {str(c): [x, y, w, h]}
                    class_index_list.append(c)
                    self.annotated_information.append(_contours)
            #         if self.check_folder != "":
            #             contou_ = [[x1, y1], [x1 + w1, y1 + h1]]
            #             cv2.polylines(image, np.array([contou_]), True, (0, 255, 0), 2)
            # cv2.imwrite(self.check_folder + f"\\{self.image_names[i]}.jpg", image)
            # # 阈值分割
            # _, binary_image = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY)
            # # 寻找轮廓
            # contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 获取输出文件名
            if not os.path.isdir(self.save_folser):
                os.makedirs(self.save_folser)
            output_file = os.path.join(self.save_folser, f"{self.image_name_list[i]}.txt")
            # 写入边界框坐标到txt文件
            with open(output_file, 'w') as f:
                for _contours in self.annotated_information:
                    for key, value in _contours.items():
                        cls = key + " "
                        points = ""
                        for box in value:  # 有多少个目标框
                            points = points + str(box) + " "
                        f.write(cls + points + "\n")
            print(f"已保存边界框坐标到：{output_file}")
        print(f"cls_index={set(class_index_list)}")


def check_labes(img_path, label_path, save_path, mode="box"):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    for filename in os.listdir(img_path):
        name, ext = os.path.splitext(filename)
        label_file = os.path.join(label_path, name + ".txt")
        img = cv2.imread(os.path.join(img_path, filename))
        hwc = img.shape
        # 读取txt文件
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                cls = line.split(" ")[0]
                line = line[2:].strip().split(" ")
                if mode == "box":
                    # 绘制坐标框
                    x = float(line[0]) * hwc[1]
                    y = float(line[1]) * hwc[0]
                    w = float(line[2]) * hwc[1]
                    h = float(line[3]) * hwc[0]
                    cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 0, 255),
                                  2)
                    cv2.putText(img, cls, (int(x - w / 2), int(y - h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                elif mode == "mask":
                    x_list = []
                    y_list = []
                    for i in range(len(line)):
                        if i % 2 == 0:
                            x_list.append(float(line[i]) * hwc[1])
                        else:
                            y_list.append(float(line[i]) * hwc[0])
                    pts = np.array([list(zip(x_list, y_list))], np.int32)
                    # 绘制多边形
                    cv2.polylines(img, pts, isClosed=True, color=(0, 0, 255), thickness=2)
                    cv2.putText(img, cls, (int(x_list[0]), int(y_list[0])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    print("mode error!")
            # 画完每一行后保存结果
            cv2.imwrite(os.path.join(save_path, filename), img)


if __name__ == '__main__':
    #
    base_path = Path(r"E:\JingXingImage\0177\det_P2\halcon_export")
    halconconvert = HalconConvert(mode="box", root_folder=base_path / "_labels")
    halconconvert.save_folser = base_path / "labels"
    halconconvert.gen_yolo_labels()
    # TODO 增加一个check标注内容的功能
    check_labes(img_path=base_path / "images",
                label_path=halconconvert.save_folser,
                save_path=base_path / "checks",
                mode="box")

    from utils.dataloaders import autosplit

    autosplit(
        path=base_path / "images",  # 图像目录路径
        weights=(0.8, 0.2, 0),  # 训练、验证和测试分割比例
        annotated_only=False  # 是否仅划分带有标注的图像
    )
