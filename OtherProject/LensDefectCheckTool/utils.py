import sys
import os
import glob
import cv2
import os.path as osp
import numpy as np
import PIL.Image
import io
import math
import time
import inspect
import json
import shutil
import threading
from natsort import natsorted, os_sorted
from loguru import logger
import halcon as ha
from turbojpeg import TurboJPEG, TJFLAG_FASTUPSAMPLE, TJFLAG_FASTDCT
from PyQt5.QtGui import QImage

current_dir = osp.abspath(os.getcwd())

DEBUG = False

# 算法库缺陷类型数量(ALGNUM)，需与 CustomWidgets.EQUIP_DEFECT_TYPES 保持一致
EXPECTED_DEFECT_NUM = 40

# HALCON 的 set_system / 句柄为全局状态，跨线程并发调用需串行化
_halcon_lock = threading.Lock()

VERSION_STR = "version：20260121"

SHORTCUT_STR = (
    "方向键↓:    下一个镜片文件夹\n"
    "方向键↑:    上一个镜片文件夹\n"
    "方向键→:    下一张图片\n"
    "方向键←:    上一张图片\n"
    "Ins键:     复制当前图片到目标文件夹\n"
    "Ctrl+O:    打开目录\n"
    "Ctrl+F:    缺陷图片筛选显示\n"
    "Ctrl+G:    给定关键词，对图片名进行过滤显示\n"
    "Ctrl+R:    图片适应窗口\n"
    "Ctrl+W:    清空面板\n"
    "Ctrl+Q:    关闭软件\n\n"
    "1.鼠标滚轮，对图片进行缩放；按住ctrl键，点击鼠标左键，对图片进行拖动。\n"
    "2.左侧面板的文件夹列表中，若是该文件夹中的图片含有缺陷，左边会有一个缺陷标记显示；若是人工对该文件夹输入了标注信息，则会有一个信息标记显示；若是该文件夹的图片缺少对应的hobj文件，则会有一个感叹号的标记显示。\n"
    "3.人工的核对信息会写入到manu_check_remark.json的文件中，该名称是固定的，不可改变"
)


def clip_num(x, min, max):
    if x < min:
        x = min
    elif x > max:
        x = max
    return x


def cycle_index(x, min, max):
    gap = max - min + 1
    if gap <= 0:
        return min
    if x > max:
        x -= gap
    elif x < min:
        x += gap
    return x


def init_logu(LOG_STATE="INFO"):
    # 输出的日志等级
    # TRACE     5
    # DEBUG     10
    # INFO      20
    # SUCCESS   25
    # WARNING   30
    # ERROR     40
    # CRITICAL  50
    timestamp = time.time()
    local_time = time.localtime(timestamp)
    time_str = time.strftime("%Y%m%d", local_time)

    log_root = "./logu"
    if not os.path.exists(log_root):
        os.makedirs(log_root)

    format_ = "[<lvl>{level:8}</><lvl>| {time:YYYY-MM-DD HH:mm:ss.SSS}] {message}</>"

    logfile = os.path.join(log_root, time_str + ".log")

    if DEBUG:
        handlers = [
            {
                "sink": sys.stderr,
                "format": format_,
                "colorize": True,
                "level": LOG_STATE,
                "filter": lambda record: record["level"].no >= logger.level("INFO").no,
            }
        ]
    else:
        handlers = [
            {
                "sink": logfile,
                "format": format_,
                "colorize": False,
                "retention": "6 days",
                "level": LOG_STATE,
                "filter": lambda record: record["level"].no >= logger.level("INFO").no,
            }
        ]

    logger.configure(handlers=handlers)


class struct(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def apply_exif_orientation(image):
    try:
        exif = image._getexif()
    except AttributeError:
        exif = None

    if exif is None:
        return image

    exif = {PIL.ExifTags.TAGS[k]: v for k, v in exif.items() if k in PIL.ExifTags.TAGS}

    orientation = exif.get("Orientation", None)

    if orientation == 1:
        # do nothing
        return image
    elif orientation == 2:
        # left-to-right mirror
        return PIL.ImageOps.mirror(image)
    elif orientation == 3:
        # rotate 180
        return image.transpose(PIL.Image.ROTATE_180)
    elif orientation == 4:
        # top-to-bottom mirror
        return PIL.ImageOps.flip(image)
    elif orientation == 5:
        # top-to-left mirror
        return PIL.ImageOps.mirror(image.transpose(PIL.Image.ROTATE_270))
    elif orientation == 6:
        # rotate 270
        return image.transpose(PIL.Image.ROTATE_270)
    elif orientation == 7:
        # top-to-right mirror
        return PIL.ImageOps.mirror(image.transpose(PIL.Image.ROTATE_90))
    elif orientation == 8:
        # rotate 90
        return image.transpose(PIL.Image.ROTATE_90)
    else:
        return image


def load_image_file(imgpath):
    try:
        image_pil = PIL.Image.open(imgpath)
    except IOError:
        print("Failed opening image file: {}".format(imgpath))
        return

    # apply orientation to image according to exif
    image_pil = apply_exif_orientation(image_pil)

    with io.BytesIO() as f:
        ext = osp.splitext(imgpath)[1].lower()
        if ext in [".jpg", ".jpeg"]:
            format = "JPEG"
        else:
            format = "PNG"
        image_pil.save(f, format=format)
        f.seek(0)
        return f.read()


def load_image_data(imgpath):
    try:
        in_file = open(imgpath, "rb")
        image_data = in_file.read()
        in_file.close()
        return image_data
    except IOError:
        logger.error(f"Failed opening image file: {imgpath}")
        return bytes()


# def image_data_to_qimage_turbojpeg(image_data):
#     img = turbo_jpeg.decode(image_data, 0, flags=TJFLAG_FASTUPSAMPLE | TJFLAG_FASTDCT)
#     height, width, channels = img.shape
#     bytes_per_line = 3 * width
#     qimage = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
#     return qimage


def image_data_qimage_pillow(image_data):
    try:
        pil_img = PIL.Image.open(io.BytesIO(image_data))
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        width, height = pil_img.size
        data = pil_img.tobytes("raw", "RGB")
        # QImage 不持有 Python bytes 缓冲区的引用，需深拷贝避免悬空内存
        qimage = QImage(data, width, height, QImage.Format_RGB888).copy()
        return qimage
    except Exception as e:
        logger.error(f"image_data_qimage_pillow 解析图片失败: {str(e)}")
        return QImage()


def load_hobj_polygons(hobjpath):
    polygon_list = []
    with _halcon_lock:
        try:
            ha.set_system("tsp_height", 6000)
            ha.set_system("tsp_width", 6000)
            if not osp.exists(hobjpath):
                logger.warning(f"hobjpath: {hobjpath} not exists!")
                return polygon_list

            DefectRegions = ha.read_object(hobjpath)
            def_num = ha.count_obj(DefectRegions)
            if def_num != EXPECTED_DEFECT_NUM:
                logger.warning(f"def_num-{def_num} not equal to {EXPECTED_DEFECT_NUM}!")
            for i in range(def_num):
                if i >= EXPECTED_DEFECT_NUM:
                    logger.warning(f"defect index {i} >= {EXPECTED_DEFECT_NUM}, skip!")
                    continue
                DefectRegion = ha.select_obj(DefectRegions, i + 1)
                DefectSelRegs = ha.connection(DefectRegion)
                def_sel_num = ha.count_obj(DefectSelRegs)
                TempArea, TempRow, TempCol = ha.area_center(DefectRegion)
                if def_sel_num == 1 and TempArea[0] < 1:
                    continue
                for j in range(def_sel_num):
                    DefSelReg = ha.select_obj(DefectSelRegs, j + 1)
                    DefSelDilReg = ha.dilation_circle(DefSelReg, 10)
                    DefSelCont = ha.gen_contour_region_xld(DefSelDilReg, "border")
                    DefSelSmoothCont = ha.smooth_contours_xld(DefSelCont, 15)
                    DefSelPolygon = ha.gen_polygons_xld(DefSelSmoothCont, "ramer", 2)
                    Rows, Cols, Length, Phi = ha.get_polygon_xld(DefSelPolygon)
                    point_num = ha.tuple_length(Rows)
                    points = []
                    for k in range(point_num):
                        points.append([int(Cols[k]), int(Rows[k])])
                    polygon_list.append([i, points])
        except Exception as e:
            logger.error(f"load_hobj_polygons 解析 {hobjpath} 失败: {str(e)}")
            return []
    return polygon_list


def load_hobj_defect_index(hobjpath):
    defect_indexs = []
    with _halcon_lock:
        try:
            ha.set_system("tsp_height", 6000)
            ha.set_system("tsp_width", 6000)
            if not osp.exists(hobjpath):
                logger.warning(f"hobjpath: {hobjpath} not exists!")
                defect_indexs.append(-1)
                return defect_indexs

            DefectRegions = ha.read_object(hobjpath)
            def_num = ha.count_obj(DefectRegions)
            if def_num != EXPECTED_DEFECT_NUM:
                logger.warning(f"def_num-{def_num} not equal to {EXPECTED_DEFECT_NUM}!")
            for i in range(def_num):
                if i >= EXPECTED_DEFECT_NUM:
                    logger.warning(f"defect index {i} >= {EXPECTED_DEFECT_NUM}, skip!")
                    continue
                DefectRegion = ha.select_obj(DefectRegions, i + 1)
                DefectSelRegs = ha.connection(DefectRegion)
                def_sel_num = ha.count_obj(DefectSelRegs)
                TempArea, TempRow, TempCol = ha.area_center(DefectRegion)
                if def_sel_num == 1 and TempArea[0] < 1:
                    continue
                defect_indexs.append(i)
        except Exception as e:
            logger.error(f"load_hobj_defect_index 解析 {hobjpath} 失败: {str(e)}")
            return [-1]
    return defect_indexs


def load_id_direct_hobjs(img_root, cur_direct, imgnames, hobj_direct_name):
    logger.trace("load_id_direct_hobjs...")
    defect_indexs = []
    parent_root = osp.dirname(osp.abspath(img_root))
    hobj_root = osp.join(parent_root, hobj_direct_name)
    hobj_direct_path = osp.join(hobj_root, cur_direct)
    if not os.path.isdir(hobj_direct_path):
        logger.warning(f"hobj_direct_path is not exists! - {hobj_direct_path}")
        defect_indexs.append([-1] * len(imgnames))
        return defect_indexs

    hobjnames = [osp.splitext(imgname)[0] + ".hobj" for imgname in imgnames]
    for hobjname in hobjnames:
        hobjpath = osp.join(hobj_direct_path, hobjname)
        defect_sel_indexs = load_hobj_defect_index(hobjpath)
        defect_indexs.append(defect_sel_indexs)
    return defect_indexs


# def load_all_id_direct_hobjs(img_root, img_ids):
#     defect_indexs_total = {}
#     for img_id in img_ids:
#         direct_path = osp.join(img_root, img_id)
#         extensions = [".jpg", ".bmp"]
#         filenames = os.listdir(direct_path)
#         imgnames = [
#             filename
#             for filename in filenames
#             if filename.lower().endswith(tuple(extensions))
#         ]
#         defect_indexs = load_id_direct_hobjs(img_root, img_id, imgnames)
#         # print(img_id, defect_indexs)
#         defect_indexs_total[img_id] = defect_indexs
#     return defect_indexs_total

"""
   判断鼠标是否在scrollArea的区域内：
    global_mouse_pos = QCursor.pos()
    scrollArea_x1 = self.pos().x() + self.widget_main.geometry().x()
    scrollArea_x2 = scrollArea_x1 + self.widget_main.width()
    scrollArea_y1 = self.pos().y() + self.widget_main.geometry().y()
    scrollArea_y2 = scrollArea_y1 + self.widget_main.height()
    logger.trace(f"{scrollArea_x1} {scrollArea_x2} {scrollArea_y1} {scrollArea_y2}")
    scrollArea_rect = [scrollArea_x1, scrollArea_y1, scrollArea_x2, scrollArea_y2]
    inside_flag = judge_point_inside_rectangle(global_mouse_pos, scrollArea_rect)
"""


def judge_point_inside_rectangle(point, rect):
    point_x1 = clip_num(point.x(), rect[0], rect[2])
    point_y1 = clip_num(point.y(), rect[1], rect[3])
    if point_x1 == point.x() and point_y1 == point.y():
        return True
    else:
        return False


def filter_filename_by_extension(filenames, extensions):
    filenames2 = [
        filename
        for filename in filenames
        if filename.lower().endswith(tuple(extensions))
    ]
    return filenames2


def flatten_nested_list(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten_nested_list(item))
        else:
            result.append(item)
    result = list(set(result))
    return result


def deduplicate_ordered(lst):
    """
    去重但保持顺序
    """
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def get_images_glob(folder_path, extensions=None, recursive=True):
    if extensions is None:
        extensions = ["jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"]

    image_files = []
    if recursive:
        pattern = os.path.join(folder_path, "**", "*")
    else:
        pattern = os.path.join(folder_path, "*")

    for ext in extensions:
        ext_pattern = (
            f"{pattern}.{ext}" if not ext.startswith(".") else f"{pattern}{ext}"
        )
        if recursive:
            found_files = glob.glob(ext_pattern, recursive=True)
        else:
            found_files = glob.glob(ext_pattern)
        image_files.extend(found_files)
    return image_files
