from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import resources
from utils import *

here = osp.dirname(osp.abspath(__file__))

# 设备检测出来的标签和算法库(ALGNUM=40)的40个标签一致，索引 0~39
EQUIP_DEFECT_TYPES = [
    "划痕",        # 0
    "压伤",        # 1
    "碰伤",        # 2
    "凹印",        # 3
    "砂纸痕",      # 4
    "亮印",        # 5
    "线痕",        # 6
    "脏污",        # 7
    "异色点",      # 8
    "笔印",        # 9
    "凹痕",        # 10
    "过磨",        # 11
    "材料线",      # 12
    "杂质点",      # 13
    "材质麻点",    # 14
    "刀痕",        # 15
    "吐酸",        # 16
    "天线盖开裂",  # 17
    "小平面",      # 18
    "顶凸",        # 19
    "毛边",        # 20
    "色差",        # 21
    "光泽",        # 22
    "膜厚",        # 23
    "面花",        # 24
    "酸印",        # 25
    "雾边",        # 26
    "露白",        # 27
    "氧化斑",      # 28
    "镭雕不良",    # 29
    "导电不良",    # 30
    "药水结晶",    # 31
    "流痕",        # 32
    "拖尾",        # 33
    "喷砂不均",    # 34
    "喷砂亮点",    # 35
    "脱脂末净",    # 36
    "残胶",        # 37
    "报废",        # 38
    "板纹线",      # 39
]
DEFECT_TYPE_NUM = len(EQUIP_DEFECT_TYPES)


def valid_defect_index(index):
    """返回合法的缺陷索引；越界或非整数返回 None，防止列表下标越界闪退。"""
    try:
        idx = int(index)
    except (TypeError, ValueError):
        return None
    return idx if 0 <= idx < DEFECT_TYPE_NUM else None


def get_defect_text(index):
    """按缺陷索引取名称，越界时返回可读的占位文本。"""
    idx = valid_defect_index(index)
    if idx is None:
        return f"未知缺陷({index})"
    return EQUIP_DEFECT_TYPES[idx]


def get_defect_color(index):
    """按缺陷索引取颜色，越界时返回兜底颜色。"""
    idx = valid_defect_index(index)
    if idx is None or idx >= len(COLORS):
        return COLORS[0]
    return COLORS[idx]
# 人工核对的标签，进行了一些精简
MANU_DEFECT_TYPES = [
    "毛屑",
    "伤痕",
    "破损",
    "腐蚀点",
    "脏污",
    "其他未知缺陷",
    "毛刺",
    "腐蚀",
    "锈迹",
    "裂纹",
    "裂痕",
    "划痕",
    "凹痕",
    "凹坑",
    "色差",
    "黑笔印"
    "未知缺陷1",
    "未知缺陷2",
    "未知缺陷3",
    "未知缺陷4",
    "未知缺陷5"
]


def newIcon(icon_name):
    icons_dir = osp.join(here, "./icons")
    return QIcon(osp.join(":/", icons_dir, icon_name))


def newAction(
    parent,
    text,
    slot=None,
    shortcut=None,
    icon_path=None,
    tip=None,
    checkable=False,
    enabled=True,
    checked=False,
):
    """Create a new action and assign callbacks, shortcuts, etc."""
    a = QAction(text, parent)
    if icon_path is not None:
        # a.setIconText(text.replace(" ", "\n"))
        a.setIcon(QIcon(icon_path))
    if shortcut is not None:
        if isinstance(shortcut, (list, tuple)):
            a.setShortcuts(shortcut)
        else:
            a.setShortcut(shortcut)
    if tip is not None:
        a.setToolTip(tip)
        a.setStatusTip(tip)
    if slot is not None:
        a.triggered.connect(slot)
    if checkable:
        a.setCheckable(True)
    a.setEnabled(enabled)
    a.setChecked(checked)
    return a


class ToolBar(QToolBar):
    def __init__(self, title):
        super(ToolBar, self).__init__(title)
        layout = self.layout()
        m = (0, 0, 0, 0)
        layout.setSpacing(0)
        layout.setContentsMargins(*m)
        self.setContentsMargins(*m)
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)

    def addAction(self, action):
        if isinstance(action, QWidgetAction):
            return super(ToolBar, self).addAction(action)
        btn = QToolButton()
        btn.setDefaultAction(action)
        btn.setToolButtonStyle(self.toolButtonStyle())
        self.addWidget(btn)

        for i in range(self.layout().count()):
            if isinstance(self.layout().itemAt(i).widget(), QToolButton):
                self.layout().itemAt(i).setAlignment(Qt.AlignCenter)


class ZoomWidget(QSpinBox):
    def __init__(self, value=100):
        super(ZoomWidget, self).__init__()
        self.setButtonSymbols(QAbstractSpinBox.NoButtons)  # 不显示上下调节按钮
        self.setRange(1, 1000)
        self.setSuffix(" %")  # 数值调节框设置后缀文本
        self.setValue(value)
        self.setToolTip("缩放比例")
        self.setStatusTip(self.toolTip())
        self.setAlignment(Qt.AlignCenter)

    def minimumSizeHint(self):
        height = super(ZoomWidget, self).minimumSizeHint().height()
        fm = QFontMetrics(self.font())
        width = fm.width(str(self.maximum()))
        return QSize(width, height)


class CheckableQWidgetAction_(QWidgetAction):
    """A QWidgetAction that supports checkable functionality."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._checked = False

    def setCheckable(self, checkable):
        """Enable or disable checkable functionality."""
        self._checkable = checkable

    def isCheckable(self):
        """Return whether the action is checkable."""
        return getattr(self, "_checkable", False)

    def setChecked(self, checked):
        """Set the checked state."""
        if self.isCheckable():
            self._checked = checked

    def isChecked(self):
        """Return the checked state."""
        if self.isCheckable():
            return self._checked
        return False

    def toggle(self):
        """Toggle the checked state."""
        if self.isCheckable():
            self._checked = not self._checked


class CheckableActionWidget(QWidgetAction):
    widgetClicked = pyqtSignal(bool)

    def __init__(self, parent=None, text=None, shortcut=None):
        super().__init__(parent)
        self.text = text if text else ""
        self._checked = False
        self._checkable = True
        self._shortcut = shortcut
        self.setShortcut(self._shortcut)

        # 创建自定义widget
        self.widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)  # 添加间距

        self.lineEdit_posMesg = QLineEdit()
        self.lineEdit_posMesg.setFixedHeight(20)
        self.lineEdit_posMesg.setAlignment(Qt.AlignCenter)
        self.lineEdit_posMesg.setPlaceholderText("关键词")
        # self.lineEdit_posMesg.setText("P1")
        self.label_name = QLabel(self.text)
        self.label_name.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.lineEdit_posMesg)
        layout.addWidget(self.label_name)
        self.widget.setLayout(layout)
        self.widget.setFixedSize(70, 50)

        self.setDefaultWidget(self.widget)

        self.widget.setMouseTracking(True)
        self.widget.installEventFilter(self)

        self.widget.mousePressEvent = self.on_widget_clicked
        self.triggered.connect(self.on_widget_clicked)

        self.setCheckable(True)
        self.updateWidgetStyle()

    def eventFilter(self, obj, event):
        """事件过滤器，用于处理悬浮效果"""
        if obj == self.widget:
            if event.type() == QEvent.Enter:
                # 鼠标进入，显示悬浮效果（除非已选中）
                if not self._checked:
                    self.widget.setStyleSheet(
                        """
                        QWidget {
                            background-color: #d8e6f2; 
                            border: 1px solid #c0dcf3;
                        }
                        QLineEdit {
                            background-color: #d8e6f2;
                            border: 1px solid #8ca5d2;
                        }
                        QLabel {
                            background-color: #d8e6f2;
                            border: none;
                        }
                        """
                    )
                return True
            elif event.type() == QEvent.Leave:
                # 鼠标离开，恢复原始样式
                self.updateWidgetStyle()
                return True
        return super().eventFilter(obj, event)

    def setCheckable(self, checkable):
        """Enable or disable checkable functionality."""
        self._checkable = checkable

    def isCheckable(self):
        """Return whether the action is checkable."""
        return self._checkable

    def setChecked(self, checked):
        """Set the checked state."""
        if self.isCheckable():
            self._checked = checked
            # print(f"_checked: {self._checked}")
            # 更新widget的样式
            self.updateWidgetStyle()

    def isChecked(self):
        """Return the checked state."""
        if self.isCheckable():
            return self._checked
        return False

    def toggle(self):
        """Toggle the checked state."""
        if self.isCheckable():
            self._checked = not self._checked
            self.setChecked(self._checked)
            self.widgetClicked.emit(self._checked)
        if self.isChecked():
            self.lineEdit_posMesg.setReadOnly(True)
        else:
            self.lineEdit_posMesg.setReadOnly(False)

    def updateWidgetStyle(self):
        """更新widget的样式以反映当前状态"""
        if self._checked:
            self.widget.setStyleSheet(
                """
                QWidget {
                    background-color: #c0dcf3; 
                    border: 1px solid #90c8f6;
                }
                QLineEdit {
                    background-color: #c0dcf3;
                    border: 1px solid #5a7bb5;
                }
                QLabel {
                    background-color: #c0dcf3;
                    border: none;
                }
                """
            )
        else:
            self.widget.setStyleSheet(
                """
                QWidget {
                    background-color: transparent;
                }
                QLabel {
                    background-color: transparent;
                    border: none;
                }
                """
            )

    def on_widget_clicked(self, event):
        """widget被点击时的处理"""
        self.toggle()

    def getLineEditText(self):
        """获取lineEdit中的文本"""
        return self.lineEdit_posMesg.text()


class PlateGridSizeInputDialog(QDialog):
    def __init__(self, parent=None, grid_size_str=None):
        super().__init__(parent)
        self.setWindowTitle("载物盘格子数目设置")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setFixedSize(200, 100)
        self.grid_size_str = grid_size_str

        layout = QVBoxLayout()
        label = QLabel("请输入载物盘每一行的格子数目:")
        layout.addWidget(label)
        self.number_input = QLineEdit()
        self.number_input.setFixedSize(100, 25)
        self.number_input.setStyleSheet("font-size:15px;")
        self.number_input.setText(grid_size_str)
        self.number_input.setValidator(QIntValidator())
        layout.addWidget(self.number_input)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        buttons.accepted.connect(self.validateAndAccept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def validateAndAccept(self):
        if self.number_input.text().strip():
            self.accept()
        else:
            QMessageBox.warning(self, "警告", "请输入有效的数字")


class ImageZoomRatioDialog(QDialog):
    def __init__(self, parent=None, img_zoom_ratio_str=None):
        super().__init__(parent)
        self.setWindowTitle("图片缩放比例设置")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.img_zoom_ratio_str = img_zoom_ratio_str

        layout = QVBoxLayout()
        label = QLabel("请输入当前图片的缩放比例:")
        layout.addWidget(label)
        self.lineEditZoomRatio = QLineEdit()
        self.lineEditZoomRatio.setFixedSize(100, 25)
        self.lineEditZoomRatio.setStyleSheet("font-size:15px;")
        self.lineEditZoomRatio.setText(self.img_zoom_ratio_str)
        self.lineEditZoomRatio.setValidator(QDoubleValidator())
        layout.addWidget(self.lineEditZoomRatio)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        buttons.setContentsMargins(15, 15, 15, 10)
        buttons.accepted.connect(self.validateAndAccept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        layout.setSpacing(15)
        self.setLayout(layout)

    def validateAndAccept(self):
        if self.lineEditZoomRatio.text().strip():
            self.accept()
        else:
            QMessageBox.warning(self, "警告", "请输入有效的图片缩放比例")


class HobjDirectNameDialog(QDialog):
    def __init__(self, parent=None, hobj_direct_name=None):
        super().__init__(parent)
        self.setWindowTitle("hobj文件夹名称设置")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.hobj_direct_name = hobj_direct_name

        layout = QVBoxLayout()
        label = QLabel("请输入hobj文件夹名称:")
        layout.addWidget(label)
        self.lineEditDirectName = QLineEdit()
        self.lineEditDirectName.setFixedSize(150, 25)
        self.lineEditDirectName.setStyleSheet("font-size:15px;")
        self.lineEditDirectName.setText(self.hobj_direct_name)
        layout.addWidget(self.lineEditDirectName)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        buttons.setContentsMargins(15, 15, 15, 10)
        buttons.accepted.connect(self.validateAndAccept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        layout.setSpacing(15)
        self.setLayout(layout)

    def validateAndAccept(self):
        if self.lineEditDirectName.text().strip():
            self.accept()
        else:
            QMessageBox.warning(self, "警告", "请输入有效的文件夹名称")


class ImageCopyRootDialog(QDialog):
    def __init__(self, parent=None, dst_img_root=None):
        super().__init__(parent)
        self.setWindowTitle("图片目标文件夹定义")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.dst_img_root = dst_img_root
        self.setModal(True)

        layout = QVBoxLayout()

        button_widget = QWidget()
        layout_button = QHBoxLayout()
        label = QLabel("目标图片文件夹:")
        self.button_browse = QPushButton("浏览")
        self.button_browse.setFixedSize(60, 25)
        self.button_locate = QPushButton("打开")
        self.button_locate.setFixedSize(60, 25)
        layout_button.addWidget(label)
        layout_button.addStretch(1)
        layout_button.addWidget(self.button_browse)
        layout_button.addWidget(self.button_locate)
        layout_button.setSpacing(5)
        layout_button.setContentsMargins(5, 5, 5, 5)
        button_widget.setLayout(layout_button)
        button_widget.setMaximumHeight(35)

        self.text_edit = QLineEdit()
        self.text_edit.setText(self.dst_img_root)
        self.text_edit.setReadOnly(True)

        spacer = QSpacerItem(15, 15, QSizePolicy.Minimum, QSizePolicy.Minimum)
        buttol_rets = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
        )
        buttol_rets.accepted.connect(self.validateAndAccept)
        buttol_rets.rejected.connect(self.reject)
        buttol_rets.setContentsMargins(10, 20, 10, 10)

        layout.addWidget(button_widget)
        layout.addWidget(self.text_edit)
        layout.addWidget(buttol_rets)
        self.setLayout(layout)

        self.button_browse.clicked.connect(self.selectDirectory)
        self.button_locate.clicked.connect(self.openFileLocation)

    def closeEvent(self, event):
        """重写closeEvent，确保只关闭对话框而不影响父窗口"""
        event.accept()

    def selectDirectory(self):
        directPath = str(
            QFileDialog.getExistingDirectory(
                self,
                self.tr("Open Directory"),
                self.dst_img_root,
                QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
            )
        )
        if directPath == "":
            return
        if not osp.exists(directPath):
            QMessageBox.warning(self, "错误", f"目录不存在: {directPath}")
            return
        self.text_edit.setText(directPath)

    def openFileLocation(self):
        directPath = self.text_edit.text()
        if not (osp.exists(directPath) and osp.isdir(directPath)):
            QMessageBox.warning(self, "错误", f"目录不存在: {directPath}")
            return
        url = QUrl.fromLocalFile(directPath)
        QDesktopServices.openUrl(url)

    def validateAndAccept(self):
        input_path = self.text_edit.text().strip()
        logger.info(f"input_path: {input_path}")
        if input_path:
            self.accept()
        else:
            QMessageBox.warning(self, "警告", "请输入有效的文件夹路径")
            return False


COLORS = [
    (230, 25, 75),  # 红色
    (60, 180, 75),  # 绿色
    (255, 225, 25),  # 黄色
    (0, 130, 200),  # 蓝色
    (245, 130, 48),  # 橙色
    (145, 30, 180),  # 紫色
    (70, 240, 240),  # 青色
    (240, 50, 230),  # 粉色
    (210, 245, 60),  # 黄绿色
    (250, 190, 190),  # 浅粉色
    (0, 128, 128),  # 深青色
    (230, 190, 255),  # 淡紫色
    (170, 110, 40),  # 棕色
    (255, 250, 200),  # 米色
    (128, 0, 0),  # 深红色
    (170, 255, 195),  # 薄荷绿
    (128, 128, 0),  # 橄榄色
    (255, 215, 180),  # 桃色
    (0, 0, 128),  # 海军蓝
    (128, 128, 128),  # 灰色
    (255, 0, 255),  # 品红
    (0, 255, 255),  # 亮青色
    (255, 0, 0),  # 亮红色
    (0, 255, 0),  # 亮绿色
    (0, 0, 255),  # 亮蓝色
    (255, 255, 0),  # 亮黄色
    (139, 69, 19),  # 棕色2
    (128, 0, 128),  # 深紫色
    (0, 128, 0),  # 深绿色
    (192, 192, 192),  # 银色
    (255, 165, 0),  # 橙色2
    (165, 42, 42),  # 棕色3
    (0, 0, 0),  # 黑色
    (255, 192, 203),  # 粉色2
    (0, 255, 127),  # 春绿色
    (255, 20, 147),  # 深粉色
    (138, 43, 226),  # 紫罗兰
    (255, 215, 0),  # 金色
    (70, 130, 180),  # 钢蓝色
    (128, 64, 0),  # 棕色4
]


class Shape(object):
    P_SQUARE = 0
    P_ROUND = 1

    PEN_WIDTH = 2
    line_color = QColor(255, 0, 0, 128)
    fill_color = QColor(0, 0, 0, 64)
    vertex_fill_color = QColor(0, 255, 0, 255)
    select_line_color = QColor(255, 255, 255, 255)
    select_fill_color = QColor(0, 255, 0, 64)
    hvertex_fill_color = QColor(255, 255, 255, 255)
    point_type = P_ROUND
    point_size = 8
    scale = 1.0

    def __init__(self, points, label, img_zoom_ratio):

        self.points = points
        self.label = label
        self.img_zoom_ratio = img_zoom_ratio
        self._closed = False

    def close(self):
        self._closed = True

    def isClosed(self):
        return self._closed

    def _scale_point(self, point: QPointF) -> QPointF:
        x = point.x() * self.scale * self.img_zoom_ratio
        y = point.y() * self.scale * self.img_zoom_ratio
        return QPointF(x, y)

    def getLeftTopPos(self):
        if len(self.points) < 1:
            return 0, 0
        points_array = []
        for point in self.points:
            points_array.append([point.x(), point.y()])
        points_array = np.array(points_array, dtype="int32")
        min_x = np.min(points_array[:, 0])
        min_y = np.min(points_array[:, 1])
        pox_x, pos_y = (
            min_x * self.scale * self.img_zoom_ratio,
            min_y * self.scale * self.img_zoom_ratio,
        )
        return pox_x, pos_y

    def drawVertex(self, path, i):
        d = self.point_size
        point = self._scale_point(self.points[i])

        if self.point_type == self.P_SQUARE:
            path.addRect(point.x() - d / 2, point.y() - d / 2, d, d)
        elif self.point_type == self.P_ROUND:
            path.addEllipse(point, d / 2.0, d / 2.0)
        else:
            assert False, "unsupported vertex shape"

    def paint(self, painter):
        color = get_defect_color(self.label)
        qcolor = QColor(color[0], color[1], color[2], 128)
        pen = QPen(qcolor)
        pen.setWidth(self.PEN_WIDTH)
        painter.setPen(pen)
        font = QFont()
        font_size = int(70 * self.scale * self.img_zoom_ratio)
        font.setPixelSize(font_size)
        font.setBold(True)
        painter.setFont(font)

        if self.points:
            line_path = QPainterPath()
            vrtx_path = QPainterPath()

            line_path.moveTo(self._scale_point(self.points[0]))
            for i, p in enumerate(self.points):
                line_path.lineTo(self._scale_point(p))
            if self.isClosed():
                line_path.lineTo(self._scale_point(self.points[0]))

            painter.drawPath(line_path)

            pos_x, pos_y = self.getLeftTopPos()
            text = get_defect_text(self.label)
            painter.drawText(int(pos_x), int(pos_y - 10), text)
