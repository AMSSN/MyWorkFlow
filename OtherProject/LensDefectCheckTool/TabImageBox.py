from CustomWidgets import *


class Canvas(QWidget):
    zoomRequest = pyqtSignal(int, QPoint)
    scrollBarSlid = pyqtSignal(QPoint)
    mouseMoved = pyqtSignal(QPointF)

    def __init__(self):
        super().__init__()

        self.setStyleSheet("background-color: blue")

        self.scale = 1.0
        self.visible = True
        self.pixmap = QPixmap()
        self.painter = QPainter()
        self.shapes = []
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.WheelFocus)

        self.dragging = False
        self.mouse_within = False
        self.last_mouse_pos = QPoint()
        self.installEventFilter(self)

    def loadPixmap(self, pixmap, clear_shapes=True):
        self.pixmap = pixmap
        self.update()  # update会自动调用paintEvent()

    def loadShapes(self, shapes):
        self.shapes = shapes
        self.update()

    def setShapeVisible(self, value):
        self.visible = value
        self.update()

    def offsetToCenter(self) -> QPointF:
        s = self.scale
        area = super(Canvas, self).size()
        w, h = self.pixmap.width() * s, self.pixmap.height() * s
        aw, ah = area.width(), area.height()
        # print(f'info -- canvas.width:{aw} canvas.height:{ah}')
        x = (aw - w) / (2 * s) if aw > w else 0
        y = (ah - h) / (2 * s) if ah > h else 0
        return QPointF(x, y)

    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        if self.pixmap:
            return self.scale * self.pixmap.size()
        return super(Canvas, self).minimumSizeHint()

    def transformPos(self, point: QPointF) -> QPointF:
        return point / self.scale - self.offsetToCenter()

    def wheelEvent(self, event):
        mods = event.modifiers()
        delta = event.angleDelta()
        self.zoomRequest.emit(delta.y(), event.pos())
        event.accept()  # 接受该事件，并阻止事件继续传播

    def mousePressEvent(self, event):
        parent_mod = self.parent().parent()
        if isinstance(parent_mod, QScrollArea):
            if parent_mod.leftbutton_pressed and parent_mod.ctrl_pressed:
                self.dragging = True
                self.last_mouse_pos = event.globalPos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        try:
            pos = self.transformPos(event.localPos())
        except AttributeError:
            return
        if self.mouse_within:
            self.mouseMoved.emit(pos)

        if self.dragging:
            current_pos = event.globalPos()
            delta = current_pos - self.last_mouse_pos
            self.last_mouse_pos = current_pos
            self.scrollBarSlid.emit(delta)

    def mouseReleaseEvent(self, event):
        parent_mod = self.parent().parent()
        if isinstance(parent_mod, QScrollArea):
            if parent_mod.leftbutton_released:
                self.dragging = False
        super().mouseReleaseEvent(event)

    def eventFilter(self, obj, event):
        if not obj.isEnabled():
            return False
        if event.type() == QEvent.Enter:
            self.mouse_within = True
        elif event.type() == QEvent.Leave:
            self.mouse_within = False
            self.mouseMoved.emit(QPoint(-1, -1))
        return super().eventFilter(obj, event)

    def paintEvent(self, event: QPaintEvent):
        if not self.pixmap:
            return super(Canvas, self).paintEvent(event)

        p = self.painter
        p.begin(self)
        p.setRenderHint(QPainter.Antialiasing)  # 用于绘制平滑的边缘和曲线
        p.setRenderHint(QPainter.HighQualityAntialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)  # 图像的缩放变换过程更加平滑

        # if(not self.pixmap.isNull()):
        p.scale(
            self.scale, self.scale
        )  # 修改绘图的比例，若scale大于1，则绘图的结果变变大
        p.translate(self.offsetToCenter())  # 该过程平移的是缩放前的实际坐标
        p.drawPixmap(0, 0, self.pixmap)
        p.scale(1 / self.scale, 1 / self.scale)

        Shape.scale = self.scale
        for shape in self.shapes:
            if self.visible:
                shape.paint(p)
        p.end()


class HoverButton(QPushButton):
    def __init__(self, icon_path, icon_size, parent=None):
        super().__init__(parent)
        self.icon_path = icon_path
        self.icon_size = icon_size
        self.setStyleSheet("background: transparent;")
        self.setVisible(False)
        self.installEventFilter(self)  # 安装事件过滤器

    def eventFilter(self, obj, event):
        if event.type() == QEvent.HoverEnter:  # 处理鼠标进入事件
            self.setIcon(QIcon(self.icon_path))
            self.setIconSize(self.icon_size)

        elif event.type() == QEvent.HoverLeave:  # 处理鼠标离开事件
            self.setIcon(QIcon())

        return super().eventFilter(obj, event)  # 确保其他事件正常处理


class MessageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            """QLabel {
                           background-color: transparent;
                           font-size: 14px;
                           font-family: "Microsoft YaHei";
                           color: #f0f0f0;
                           }"""
        )
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(2)  # 阴影模糊半径
        shadow.setXOffset(1)  # X方向偏移
        shadow.setYOffset(1)  # Y方向偏移
        shadow.setColor(QColor(0, 0, 0, 255))  # 阴影颜色和透明度
        self.setGraphicsEffect(shadow)


class MyTabWidget(QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # 在TabWidget的选项卡上，添加滚轮事件过滤器
        self.tabBar().installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel:
            return True
        return super().eventFilter(obj, event)


class MyQScrollArea(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ctrl_pressed = False
        self.leftbutton_pressed = False
        self.leftbutton_released = False
        self.setMouseTracking(True)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Enter:
            obj.setFocus()

        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Control:
            logger.trace("crtl press!")
            self.ctrl_pressed = True
            self.viewport().setCursor(Qt.OpenHandCursor)
            return True
        elif event.type() == QEvent.KeyRelease and event.key() == Qt.Key_Control:
            logger.trace("crtl release!")
            self.ctrl_pressed = False
            self.viewport().setCursor(Qt.ArrowCursor)
            return True

        self.leftbutton_pressed = (
            event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton
        )
        self.leftbutton_released = (
            event.type() == QEvent.MouseButtonRelease
            and event.button() == Qt.LeftButton
        )
        if self.ctrl_pressed and self.leftbutton_pressed:
            logger.trace("left button press!")
            self.viewport().setCursor(Qt.ClosedHandCursor)
        elif self.ctrl_pressed and self.leftbutton_released:
            logger.trace("left button release!")
            self.viewport().setCursor(Qt.OpenHandCursor)

        return super().eventFilter(obj, event)


class TabImageBox(MyTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.initUI()

        self.qimage = QImage()

    def initUI(self):
        self.scrollArea = MyQScrollArea(self)
        self.viewWidget = QWidget(self)

        self.addTab(self.scrollArea, "tab1")
        self.setTabText(0, "当前图片")
        self.addTab(self.viewWidget, "tab2")
        self.setTabText(1, "工位汇总")
        self.setStyleSheet("font-size:25px; font:bold;")
        self.setMinimumWidth(300)

        self.canvas = Canvas()
        # self.canvas.setEnabled(False)
        self.scrollArea.setWidget(self.canvas)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # self.scrollArea.setStyleSheet("border: 1px solid red")

        self.pushButton_slideLeft = HoverButton(
            ":/icons/bk_left_arrow.png", QSize(50, 80), self
        )
        self.pushButton_slideLeft.setParent(self.scrollArea)
        self.pushButton_slideRight = HoverButton(
            ":/icons/bk_right_arrow.png", QSize(50, 80), self
        )
        self.pushButton_slideRight.setParent(self.scrollArea)

        self.label_imgMesg = MessageLabel(self)
        self.label_imgMesg.setParent(self.scrollArea)

        self.label_copyMesg = QLabel(self)
        self.label_copyMesg.setStyleSheet(
            """QLabel {
                background-color: rgba(155, 155, 155, 0.5);
                font-size: 14px;
                font-family: "Microsoft YaHei";
                color: gray;
                border-radius: 2px;
                }"""
        )
        self.label_copyMesg.setParent(self.scrollArea)
        self.label_copyMesg.setWordWrap(True)  # 关键设置：启用自动换行
        self.label_copyMesg.setAlignment(Qt.AlignLeft | Qt.AlignTop)  # 左上对齐
        self.label_copyMesg.setContentsMargins(10, 5, 10, 5)
        self.label_copyMesg.setVisible(False)

        self.scrollBars = {
            Qt.Vertical: self.scrollArea.verticalScrollBar(),
            Qt.Horizontal: self.scrollArea.horizontalScrollBar(),
        }

        self.canvas.scrollBarSlid.connect(self.setScrollBar)

    def resetState(self):
        self.canvas.pixmap = None
        self.canvas.shapes.clear()
        self.canvas.update()
        self.canvas.setEnabled(False)
        self.pushButton_slideLeft.setVisible(False)
        self.pushButton_slideRight.setVisible(False)
        self.label_imgMesg.setVisible(False)
        self.update()

    def setSlideButtonState(self):
        self.pushButton_slideLeft.setVisible(True)
        btn1_x_shift = 0
        btn1_y_shift = (
            self.scrollArea.height() / 2 - self.pushButton_slideLeft.height() / 2
        )
        self.pushButton_slideLeft.move(int(btn1_x_shift), int(btn1_y_shift))

        self.pushButton_slideRight.setVisible(True)
        btn2_x_shift = self.scrollArea.width() - self.pushButton_slideRight.width() - 2
        btn2_y_shift = btn1_y_shift
        self.pushButton_slideRight.move(int(btn2_x_shift), int(btn2_y_shift))

    def updateImgMesgLabel(self, imgname, cur_img_index, img_num, img_size, cur_id):
        mesg_text = f"名称: {imgname} [{cur_img_index+1}/{img_num}]\n尺寸: {img_size[0]}x{img_size[1]}"
        self.label_imgMesg.setText(mesg_text)
        self.label_imgMesg.adjustSize()
        self.label_imgMesg.move(10, 10)
        self.label_imgMesg.setVisible(True)

    def setScrollBar(self, delta):
        # print("setScrollBar...")
        slid_x = self.scrollBars[Qt.Horizontal].value() - delta.x()
        self.scrollBars[Qt.Horizontal].setValue(slid_x)
        slid_y = self.scrollBars[Qt.Vertical].value() - delta.y()
        self.scrollBars[Qt.Vertical].setValue(slid_y)
