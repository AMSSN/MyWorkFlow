from CustomWidgets import *


class HighlightLabel(QLabel):
    style_light = """QLabel {
            font-size: 15px;
            font-weight: bold;
            background-color: yellow;
            color: black;
            border: 3px solid #ef9210;
            border-radius: 5px;
        }
    """
    style_dark = """QLabel {
            font-size: 15px;
            color: black;
            border: 1px solid #d8d4d6;
            border-radius: 5px;
        }
    """

    def __init__(self, text):
        super().__init__(text)

        self.setAlignment(Qt.AlignCenter)
        self.setFixedSize(60, 30)
        self.setStateDark()

    def setStateDark(self):
        self.setStyleSheet(self.style_dark)

    def setStateLight(self):
        self.setStyleSheet(self.style_light)


class HideComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)  # 启用鼠标跟踪
        self.view().viewport().installEventFilter(self)  # 为下拉列表视图安装事件过滤器
        self.installEventFilter(self)  # 为自身安装事件过滤器
        self.setStyleSheet("font-size: 14px;")

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Leave:
            pop_flag = self.view().isVisible()
            if pop_flag:
                if obj == self.view().viewport():
                    if not self.underMouse():
                        self.hide()
            else:
                self.hide()
        return super().eventFilter(obj, event)


class DeletableListWidget(QListWidget):
    itemCountChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

    # 对选中的item，点击Del键进行删除
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            self.deleteSelectedItems()
        else:
            super().keyPressEvent(event)

    def deleteSelectedItems(self):
        for item in self.selectedItems():
            self.takeItem(self.row(item))

    def addItem(self, item):
        super().addItem(item)
        self.itemCountChanged.emit()

    def addItems(self, item):
        super().addItems(item)
        self.itemCountChanged.emit()

    def takeItem(self, item):
        super().takeItem(item)
        self.itemCountChanged.emit()


class CheckResultBoard(QWidget):

    button_style1 = """
                QPushButton {
                    background-color: #e5e5e5; border: 1px solid #7a7a7a; font-size: 15px; border-radius: 2px;
                }
                QPushButton:hover {
                    background-color: #d5d5d5;
                }
                QPushButton:pressed {
                    background-color: #c5c5c5;
                }
            """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.parent = parent
        grid_size_str = self.parent.settings.value("window/grid_size", "5")
        self.grid_size = int(grid_size_str)
        logger.trace(f"setting grid_size: {self.grid_size}")
        self.initUI()

    def initUI(self):

        # -----------------widget_top1-------------------
        self.widget_top1 = QWidget(self)
        self.widget_top1.setProperty("name", "widget_top1")
        layout_top1 = QHBoxLayout()
        self.label_plateTitle = QLabel(self)
        self.label_plateTitle.setText("表盘编号：")
        self.label_plateTitle.setStyleSheet("font-size:20px; font:bold")
        self.label_plateNum = QLabel(self)
        self.label_plateNum.setStyleSheet("font-size:20px;")
        layout_top1.addWidget(self.label_plateTitle)
        layout_top1.addWidget(self.label_plateNum)
        self.widget_top1.setLayout(layout_top1)
        self.widget_top1.setStyleSheet(
            """QWidget[name="widget_top1"] {border:1px solid black; background-color: #7c85a4}"""
        )
        ##7c85a4  #929ec7

        # -----------------widget_top2-------------------
        self.widget_top2 = QWidget(self)
        self.widget_top2.setProperty("name", "widget_top2")
        layout_top2 = QVBoxLayout()

        self.label_equipTitle = QLabel(self)
        self.label_equipTitle.setText("设备检测结果：")
        self.label_equipTitle.setStyleSheet("font-size:20px; font:bold")
        self.checkBox_draw = QCheckBox()
        self.checkBox_draw.setCheckState(Qt.Checked)
        self.checkBox_draw.setText("缺陷画框显示")
        self.checkBox_draw.setStyleSheet("font-size:15px;")
        self.listWidget_equipRlt = QListWidget(self)
        self.listWidget_equipRlt.setStyleSheet("font-size:15px; color: gray;")
        self.listWidget_equipRlt.setSelectionMode(QListWidget.NoSelection)  # 禁用选择
        self.listWidget_equipRlt.setFocusPolicy(Qt.NoFocus)  # 移除焦点框
        self.listWidget_equipRlt.setMinimumHeight(160)

        # 添加设备检测结果OK和NG的标签
        widget_equipLabel = QWidget(self)
        layout_equipLabel = QHBoxLayout()
        self.label_equipOk = HighlightLabel("OK")
        self.label_equipNg = HighlightLabel("NG")
        layout_equipLabel.addStretch(1)
        layout_equipLabel.addWidget(self.label_equipOk)
        layout_equipLabel.addWidget(self.label_equipNg)
        layout_equipLabel.setContentsMargins(0, 0, 0, 0)
        layout_equipLabel.setSpacing(15)
        widget_equipLabel.setLayout(layout_equipLabel)

        layout_top2.addWidget(self.label_equipTitle)
        layout_top2.addWidget(self.checkBox_draw)
        layout_top2.addWidget(self.listWidget_equipRlt)
        layout_top2.addWidget(widget_equipLabel)
        layout_top2.setSpacing(10)
        self.widget_top2.setLayout(layout_top2)
        self.widget_top2.setStyleSheet(
            """QWidget[name="widget_top2"] {border:1px solid black; background-color: #929ec7}"""
        )

        # -----------------widget_top3------------------
        self.widget_top3 = QWidget(self)
        self.widget_top3.setProperty("name", "widget_top3")
        layout_top3 = QVBoxLayout()

        self.label_manuTitle = QLabel(self)
        self.label_manuTitle.setText("人工核对结果：")
        self.label_manuTitle.setStyleSheet("font-size:20px; font:bold;")

        self.pushButton_manuAdd = QPushButton("新增", self)
        self.pushButton_manuAdd.setFixedSize(50, 25)
        self.pushButton_manuAdd.setStyleSheet(self.button_style1)
        self.pushButton_manuAdd.setEnabled(False)
        self.listWidget_manuRlt = DeletableListWidget(self)
        self.listWidget_manuRlt.setStyleSheet("font-size:15px")
        self.listWidget_manuRlt.setMinimumHeight(160)
        # self.comboBox_addDefect = QComboBox(self)
        # self.comboBox_addDefect.hide()

        # 添加人工检测结果OK和NG的标签
        widget_manuLabel = QWidget(self)
        layout_manuLabel = QHBoxLayout()
        self.label_manuOk = HighlightLabel("OK")
        self.label_manuNg = HighlightLabel("NG")
        layout_manuLabel.addStretch(1)
        layout_manuLabel.addWidget(self.label_manuOk)
        layout_manuLabel.addWidget(self.label_manuNg)
        layout_manuLabel.setContentsMargins(0, 0, 0, 0)
        layout_manuLabel.setSpacing(15)
        widget_manuLabel.setLayout(layout_manuLabel)

        layout_top3.addWidget(self.label_manuTitle)
        layout_top3.addWidget(self.pushButton_manuAdd)
        layout_top3.addWidget(self.listWidget_manuRlt)
        layout_top3.addWidget(widget_manuLabel)
        layout_top3.setSpacing(10)
        self.widget_top3.setLayout(layout_top3)
        self.widget_top3.setStyleSheet(
            """QWidget[name="widget_top3"] {border:1px solid black; background-color: #a3a8bc}"""
        )

        # -----------------widget_top4------------------
        self.widget_top4 = QWidget(self)
        self.widget_top4.setProperty("name", "widget_top4")
        layout_top4 = QVBoxLayout()
        self.label_commentTitle = QLabel(self)
        self.label_commentTitle.setText("缺陷备注：")
        self.label_commentTitle.setStyleSheet("font-size:20px; font:bold")
        self.textEdit_comment = QTextEdit(self)
        self.textEdit_comment.setMinimumHeight(100)
        self.textEdit_comment.setStyleSheet("font-size: 15px;")
        self.textEdit_comment.setEnabled(False)
        layout_top4.addWidget(self.label_commentTitle)
        layout_top4.addWidget(self.textEdit_comment)
        self.widget_top4.setLayout(layout_top4)
        self.widget_top4.setStyleSheet(
            """QWidget[name="widget_top4"] {border:1px solid black; background-color: #bcbfca}"""
        )

        # ----------------------------------------------
        layout_main = QVBoxLayout()
        layout_main.addWidget(self.widget_top1)
        layout_main.addWidget(self.widget_top2)
        layout_main.addWidget(self.widget_top3)
        layout_main.addWidget(self.widget_top4)
        self.setLayout(layout_main)

        self.setFixedWidth(200)
        self.setMaximumWidth(300)

        # connections
        self.pushButton_manuAdd.clicked.connect(self.showDefectTypeComboBox)
        self.listWidget_manuRlt.itemCountChanged.connect(self.updateManuLabelState)

    def resetState(self):
        self.label_plateNum.clear()
        self.listWidget_equipRlt.clear()
        self.label_equipOk.setStateDark()
        self.label_equipNg.setStateDark()
        self.pushButton_manuAdd.setEnabled(False)
        self.listWidget_manuRlt.clear()
        self.label_manuOk.setStateDark()
        self.label_manuNg.setStateDark()
        self.textEdit_comment.clear()
        self.textEdit_comment.setEnabled(False)

    def updateCheckListBoard(self, cur_direct, defect_indexs, cur_check_raw_dict):
        defect_indexs_sim = flatten_nested_list(defect_indexs)
        defect_indexs_sim = natsorted(defect_indexs_sim)
        # widget_top1
        try:
            cur_id = int(cur_direct.split("_")[1].strip())
            plate_grids = self.grid_size * self.grid_size
            logger.trace(f"plate_grids: {plate_grids}")
            num_of_plate = cur_id // plate_grids + 1
            index_in_plate = cur_id % plate_grids
            pos_text = f"{num_of_plate} - {index_in_plate}"
        except ValueError:
            logger.error(f"cur_id: {cur_id} error!")
            pos_text = f"? - ?"
        self.label_plateNum.setText(pos_text)

        # widget_top2
        self.listWidget_equipRlt.clear()
        # defect_indexs_sim中去掉-1
        defect_indexs_sim2 = [x for x in defect_indexs_sim if x != -1]
        num = len(defect_indexs_sim2)
        if num < 1:
            self.label_equipOk.setStateLight()
            self.label_equipNg.setStateDark()
        else:
            equip_defect_texts = [get_defect_text(i) for i in defect_indexs_sim2]
            equip_defect_texts = os_sorted(equip_defect_texts)
            self.listWidget_equipRlt.addItems(equip_defect_texts)
            self.label_equipOk.setStateDark()
            self.label_equipNg.setStateLight()

        # widget_top3
        self.label_manuOk.setStateLight()
        self.label_manuNg.setStateDark()
        self.pushButton_manuAdd.setEnabled(True)
        self.listWidget_manuRlt.clear()
        manu_defects = cur_check_raw_dict["manu_defects"]
        for manu_defect in manu_defects:
            self.listWidget_manuRlt.addItem(manu_defect)

        # widget_top4
        self.textEdit_comment.setEnabled(True)
        self.textEdit_comment.clear()
        remarks = cur_check_raw_dict["remarks"]
        remark_str = "\n".join(remarks)
        self.textEdit_comment.setPlainText(remark_str)

        self.update()

    def updateManuLabelState(self):
        item_num = self.listWidget_manuRlt.count()
        logger.info("manu_item_num: {item_num}")
        if item_num < 1:
            self.label_manuOk.setStateLight()
            self.label_manuNg.setStateDark()
        else:
            self.label_manuOk.setStateDark()
            self.label_manuNg.setStateLight()

    def showDefectTypeComboBox(self):
        manu_defect_texts = MANU_DEFECT_TYPES
        self.comboBox_addDefect = HideComboBox(self)
        self.comboBox_addDefect.addItems(manu_defect_texts)
        pos = self.pushButton_manuAdd.mapToGlobal(
            QPoint(0, self.pushButton_manuAdd.height())
        )
        pos = self.mapFromGlobal(pos)
        self.comboBox_addDefect.move(pos)
        self.comboBox_addDefect.show()

        self.comboBox_addDefect.activated.connect(self.addSelectedDefect)

    def addSelectedDefect(self):
        selected_text = self.comboBox_addDefect.currentText()
        self.listWidget_manuRlt.addItem(selected_text)
        self.listWidget_manuRlt.setFocus()
        self.comboBox_addDefect.hide()
        self.comboBox_addDefect.deleteLater()
