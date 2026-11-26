from CustomWidgets import *


class HobjLoader(QThread):
    startLoad = pyqtSignal()
    singleIdReady = pyqtSignal(int, str, list, bool)
    finishLoad = pyqtSignal()

    def __init__(self, img_root, img_ids, mesg_ids, hobj_direct_name):
        super().__init__()
        self.img_root = img_root
        self.img_ids = img_ids
        self.mesg_ids = mesg_ids
        self.hobj_direct_name = hobj_direct_name
        self._stopped = False

    def stop(self):
        self._stopped = True

    def run(self):
        try:
            if len(self.img_ids) < 1:
                return

            for ind, img_id in enumerate(self.img_ids):
                if self._stopped:
                    break
                direct_path = osp.join(self.img_root, img_id)
                filenames = os.listdir(direct_path)
                imgnames = filter_filename_by_extension(filenames, [".jpg", ".bmp"])
                imgnames = natsorted(imgnames)
                defect_indexs = load_id_direct_hobjs(
                    self.img_root, img_id, imgnames, self.hobj_direct_name
                )
                mesg_flag = True if img_id in self.mesg_ids else False
                self.singleIdReady.emit(ind, img_id, defect_indexs, mesg_flag)
                self.startLoad.emit()
            self.finishLoad.emit()
        except Exception as e:
            logger.error(f"HobjLoader.run 发生异常: {str(e)}")


class IconItemDelegate(QStyledItemDelegate):
    def __init__(self, parent=None, loaded=False):
        super().__init__(parent)
        self.loaded = False

    def paint(self, painter, option, index):
        logger.trace("IconItemDelegate.paint...")
        # 绘制背景（避免重影）
        if option.state & QStyle.State_Selected:
            painter.fillRect(option.rect, option.palette.highlight())
        else:
            painter.fillRect(option.rect, option.palette.base())

        icon1 = index.data(Qt.UserRole + 1)
        icon2 = index.data(Qt.UserRole + 2)
        icon3 = index.data(Qt.UserRole + 3)
        icon_size = 16

        if isinstance(icon1, QIcon) and not icon1.isNull():
            rect1 = QRect(
                option.rect.x() + 20,
                option.rect.y() + (option.rect.height() - icon_size) // 2,
                icon_size,
                icon_size,
            )
            icon1.paint(painter, rect1, Qt.AlignCenter, QIcon.Normal, QIcon.On)

        if isinstance(icon2, QIcon) and not icon2.isNull():
            rect2 = QRect(
                option.rect.right() - icon_size - 40,
                option.rect.y() + (option.rect.height() - icon_size) // 2,
                icon_size,
                icon_size,
            )
            icon2.paint(painter, rect2, Qt.AlignCenter, QIcon.Normal, QIcon.On)

        if isinstance(icon3, QIcon) and not icon3.isNull():
            rect3 = QRect(
                option.rect.right() - icon_size - 15,
                option.rect.y() + (option.rect.height() - icon_size) // 2,
                icon_size,
                icon_size,
            )
            icon3.paint(painter, rect3, Qt.AlignCenter, QIcon.Normal, QIcon.On)

        text_rect = QRect(option.rect)
        text_rect.setLeft(text_rect.left() + 55)

        # 从item数据中读取是否需要粗体显示
        use_bold = index.data(Qt.UserRole + 4)
        if use_bold:
            # 保存当前字体状态
            original_font = painter.font()
            bold_font = QFont(original_font)
            bold_font.setBold(True)
            painter.setFont(bold_font)

        # 从item中获取loaded状态
        # 通过listWidget获取item
        list_widget = None
        if self.parent() and hasattr(self.parent(), "listWidget_ids"):
            list_widget = self.parent().listWidget_ids
        elif self.parent() and isinstance(self.parent(), QListWidget):
            list_widget = self.parent()

        item = None
        if list_widget:
            item = list_widget.itemFromIndex(index)

        if item and hasattr(item, "loaded") and item.loaded:
            painter.setPen(Qt.black)
        else:
            painter.setPen(Qt.gray)
        painter.drawText(
            text_rect, Qt.AlignVCenter | Qt.AlignLeft, index.data(Qt.DisplayRole)
        )

        # 如果使用了粗体，恢复原始字体
        if use_bold:
            painter.setFont(original_font)

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        return QSize(size.width() + 40, size.height())  # 增加宽度容纳图标


class ClickListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pending_click_item = None
        self._click_timer = QTimer(self)
        self._click_timer.setSingleShot(True)
        self._click_timer.timeout.connect(self._handle_single_click)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            item = self.itemAt(event.pos())
            if item:
                self._pending_click_item = item
                self._click_timer.start(250)  # 250ms延迟判断是单击还是双击
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._click_timer.stop()  # 取消单击处理
            item = self.itemAt(event.pos())
            if item:
                self.setCurrentItem(item)  # 双击时才设置当前项
        super().mouseDoubleClickEvent(event)

    def _handle_single_click(self):
        if self._pending_click_item:
            self._pending_click_item = None


class IdListBoard(QWidget):
    switchItem = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.parent = parent
        self.initUI()

        self.img_ids = []
        self.icon1 = QIcon(":/icons/defect.png")
        self.icon2 = QIcon(":/icons/message.png")
        self.icon3 = QIcon(":/icons/lack.png")
        self.hobj_thread = QThread()
        self.previous_item = None
        self.current_item = None
        self.defect_indexs_total = {}

    def initUI(self):
        self.widget_leftSub0 = QWidget(self)
        layout_leftSub0 = QHBoxLayout()
        self.pushButton_open = QPushButton("打开目录", self)
        self.pushButton_open.setFixedSize(60, 30)
        layout_leftSub0.addWidget(self.pushButton_open)
        layout_leftSub0.setAlignment(Qt.AlignLeft)
        layout_leftSub0.setContentsMargins(0, 5, 0, 5)
        layout_leftSub0.setSpacing(0)
        self.widget_leftSub0.setLayout(layout_leftSub0)

        self.lineEdit_inputPath = QLineEdit(self)
        default_str = "请输入路径，并点击打开按钮"
        self.lineEdit_inputPath.setPlaceholderText(default_str)
        self.lineEdit_inputPath.setStyleSheet("color:gray")
        self.lineEdit_inputPath.setFixedHeight(35)
        self.lineEdit_inputPath.setReadOnly(True)

        self.widget_labelIds = QWidget(self)
        layout_labelIds = QHBoxLayout()
        self.label_ids = QLabel("样本列表", self)
        self.label_ids.setFixedSize(60, 30)
        self.label_ids.setStyleSheet("font-size:15px")
        self.label_hobjName = QLabel("", self)
        self.label_hobjName.adjustSize()
        self.label_hobjName.setStyleSheet("font-size:15px; font-weight: bold;")
        layout_labelIds.addWidget(self.label_ids)
        layout_labelIds.addStretch(1)
        layout_labelIds.addWidget(self.label_hobjName)
        self.widget_labelIds.setLayout(layout_labelIds)

        self.listWidget_ids = QListWidget(self)
        self.listWidget_ids.setStyleSheet("font-size:15px")
        self.listWidget_ids.setMinimumHeight(100)
        self.listWidget_ids.setItemDelegate(IconItemDelegate(self, loaded=False))

        self.pushButton_prev = QPushButton("上一个", self)
        self.pushButton_prev.setFixedSize(60, 30)
        self.pushButton_prev.setEnabled(False)
        self.pushButton_next = QPushButton("下一个", self)
        self.pushButton_next.setFixedSize(60, 30)
        self.pushButton_next.setEnabled(False)

        self.widget_leftSub1 = QWidget(self)
        layout_leftSub1 = QHBoxLayout()
        layout_leftSub1.addWidget(self.pushButton_prev)
        layout_leftSub1.addWidget(self.pushButton_next)
        self.widget_leftSub1.setLayout(layout_leftSub1)

        layout_left = QVBoxLayout()
        layout_left.addWidget(self.widget_leftSub0)
        layout_left.addWidget(self.lineEdit_inputPath)
        layout_left.addWidget(self.widget_labelIds)
        layout_left.addWidget(self.listWidget_ids)
        layout_left.addWidget(self.widget_leftSub1)
        layout_left.setSpacing(5)
        self.setMinimumWidth(200)
        self.setMaximumWidth(300)

        self.setLayout(layout_left)

        self.listWidget_ids.itemSelectionChanged.connect(self.onListWidgetSelectChanged)

    def resetState(self):
        self.defect_indexs_total.clear()
        self.lineEdit_inputPath.clear()
        self.label_hobjName.clear()
        self.listWidget_ids.clear()
        self.pushButton_prev.setEnabled(False)
        self.pushButton_next.setEnabled(False)
        self.current_item = None
        self.previous_item = None

    def initIdListBoard(self, targetDirPath, mesg_ids):
        lineTextStr = osp.normpath(targetDirPath)
        self.lineEdit_inputPath.setText(lineTextStr)
        directs = os.listdir(lineTextStr)
        self.img_ids.clear()
        for direct in directs:
            direct_path = osp.join(lineTextStr, direct)
            if osp.isdir(direct_path):
                self.img_ids.append(direct)
        id_num = len(self.img_ids)
        if id_num < 1:
            return
        self.img_ids = natsorted(self.img_ids)

        # 清空列表并创建初始items，设置loaded=False
        self.listWidget_ids.clear()
        for img_id in self.img_ids:
            item = QListWidgetItem(img_id)
            item.loaded = False  # 初始状态为未加载，显示灰色
            self.listWidget_ids.addItem(item)

        hobj_direct_name = self.parent.settings.value(
            "window/hobj_direct_name", "src_hobj"
        )
        self.label_hobjName.setText(hobj_direct_name)
        self.hobj_thread = HobjLoader(
            lineTextStr, self.img_ids, mesg_ids, hobj_direct_name
        )
        self.hobj_thread.singleIdReady.connect(self.updateIdListWidget)
        self.hobj_thread.finishLoad.connect(self.finishLoadIdListWidget)
        self.hobj_thread.start()

    def updateIdListWidget(self, ind, img_id, defect_indexs, mesg_flag):
        logger.trace("updateIdListWidget...")
        self.defect_indexs_total.update({img_id: defect_indexs})

        defect_indexs_sim = flatten_nested_list(defect_indexs)
        defect_indexs_sim = natsorted(defect_indexs_sim)
        logger.trace(f"{img_id}: {defect_indexs_sim}")

        item = QListWidgetItem()
        item.loaded = True
        defect_flag = False
        lack_flag = False
        if -1 in defect_indexs_sim:
            lack_flag = True
            defect_indexs_sim.remove(-1)
        if len(defect_indexs_sim) > 0:
            defect_flag = True
        if defect_flag:
            icon1 = self.icon1
        else:
            icon1 = QIcon()
        if mesg_flag:
            icon2 = self.icon2
        else:
            icon2 = QIcon()
        if lack_flag:
            icon3 = self.icon3
        else:
            icon3 = QIcon()
        item.setText(img_id)
        item.setData(Qt.UserRole + 1, icon1)
        item.setData(Qt.UserRole + 2, icon2)
        item.setData(Qt.UserRole + 3, icon3)
        old_item = self.listWidget_ids.takeItem(ind)
        if old_item:
            del old_item
        self.listWidget_ids.insertItem(ind, item)
        # self.listWidget_ids.update()

    def updateIdListWidgetMesg(self, img_id, mesg_flag):
        ind = -1
        for row in range(self.listWidget_ids.count()):
            if self.listWidget_ids.item(row).text() == img_id:
                ind = row
                break
        if ind < 0:
            return
        if mesg_flag:
            icon2 = self.icon2
        else:
            icon2 = QIcon()
        old_item = self.listWidget_ids.takeItem(ind)
        new_item = old_item.clone()
        # 保持loaded属性
        if hasattr(old_item, "loaded"):
            new_item.loaded = old_item.loaded
        new_item.setData(Qt.UserRole + 2, icon2)
        if old_item:
            del old_item
        self.listWidget_ids.insertItem(ind, new_item)
        # self.listWidget_ids.update()

    def onListWidgetSelectChanged(self):
        logger.trace("onListWidgetSelectChanged...")
        new_current = self.listWidget_ids.currentItem()

        if new_current != self.current_item:
            self.previous_item = self.current_item
            self.current_item = new_current

        if self.current_item is not None:
            cur_item_text = self.current_item.text()
        else:
            cur_item_text = ""
        if self.previous_item is not None:
            prev_item_text = self.previous_item.text()
        else:
            prev_item_text = ""
        self.switchItem.emit(cur_item_text, prev_item_text)
        logger.trace(
            f"onListWidgetSelectChanged cur_item:{cur_item_text} prev_item:{prev_item_text}"
        )
        if self.previous_item is not None:
            self.previous_item.setData(Qt.UserRole + 4, False)
        # self.listWidget_ids.update()

    def highLightListWidgetId(self, light_flag):
        item = self.listWidget_ids.currentItem()
        if item is None:
            return
        item.setData(Qt.UserRole + 4, light_flag)
        # self.listWidget_ids.update()

    def finishLoadIdListWidget(self):
        self.pushButton_prev.setEnabled(True)
        self.pushButton_next.setEnabled(True)
        # self.listWidget_ids.update()
        self.hobj_thread.quit()

        total_num, defect_num = self.statisticDefectNum()
        defect_text = f"样本总数: {total_num} 缺陷样本: {defect_num}"
        self.parent.label_statusDefect.setText(defect_text)

    def statisticDefectNum(self):
        total_num = len(self.defect_indexs_total)

        defect_num = 0
        for key, defect_indexs in self.defect_indexs_total.items():
            defect_indexs_sim = flatten_nested_list(defect_indexs)
            defect_indexs_sim = natsorted(defect_indexs_sim)
            print(key, defect_indexs_sim)
            if len(defect_indexs_sim) > 0:
                defect_num += 1
        return total_num, defect_num
