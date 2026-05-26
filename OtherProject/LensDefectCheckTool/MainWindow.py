from PyQt5.QtWidgets import QApplication, QMainWindow
from TabImageBox import TabImageBox
from CheckResultBoard import CheckResultBoard
from IdListBoard import IdListBoard
from CustomWidgets import *
from label_file import *


class ImageLoader(QThread):
    sendImage = pyqtSignal(int, str, bytes)
    finishLoad = pyqtSignal()

    def __init__(self, cur_direct_path, imgnames):
        super().__init__()
        self.cur_direct_path = cur_direct_path
        self.imgnames = imgnames

    def run(self):
        try:
            if len(self.imgnames) < 1:
                self.finishLoad.emit()
                return

            for ind, imgname in enumerate(self.imgnames):
                imgpath = osp.join(self.cur_direct_path, imgname)
                if not osp.exists(imgpath):
                    logger.error(f"{imgpath} is not exists!")
                    continue
                imageData = load_image_data(imgpath)
                self.sendImage.emit(ind, imgname, imageData)
            self.finishLoad.emit()
        except Exception as e:
            logger.error(f"ImageLoader.run 发生异常: {str(e)}")


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.settings = QSettings("Lens", "LensDefectCheckTool")
        self.scale = 1.0

        self.initUI()

        self.targetDirPath = None
        self.img_datas = {}
        self.cur_direct = None
        self.cur_img_index = -1
        self.zoom_value = 1.0
        self.qimage = QImage()
        self.last_size = self.size()
        self.image_thread = QThread()

        self.prev_direct = None
        self.label_file = None

    def initUI(self):
        self.setWindowTitle("镜片缺陷核对工具")
        self.setStyleSheet("QMainWindow {background-color: #d1d1d7;}")
        self.resize(1480, 900)

        self.widget_left = IdListBoard(self)
        self.widget_main = TabImageBox(self)
        self.widget_right = CheckResultBoard(self)

        self.layout_page = QHBoxLayout()
        self.layout_page.setContentsMargins(15, 5, 15, 5)
        self.layout_page.setSpacing(10)
        self.layout_page.addWidget(self.widget_left)
        self.layout_page.addWidget(self.widget_main)
        self.layout_page.addWidget(self.widget_right)
        self.widget_page = QWidget(self)
        self.widget_page.setLayout(self.layout_page)
        self.setCentralWidget(self.widget_page)

        # 菜单栏
        self.menu_file = self.menuBar().addMenu("文件")
        self.menu_setting = self.menuBar().addMenu("设置")
        self.menu_help = self.menuBar().addMenu("帮助")
        self.menuBar().setStyleSheet(
            """
            QMenuBar {
                font-size: 15px;
                background-color: #585e73;
            }
            QMenuBar::item {
                font-size: 15px;
            }
            QMenu {
                font-size: 12px;
            }
        """
        )

        # 工具栏
        self.toolbar = ToolBar("Tool")
        self.toolbar.setStyleSheet("QToolBar { background-color: #898d9c; }")
        self.toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.addToolBar(Qt.TopToolBarArea, self.toolbar)  # 将工具栏放到顶部区域
        self.toolbar.toggleViewAction().setEnabled(
            False
        )  # 禁用工具栏的显示/隐藏切换功能

        self.zoomWidget = ZoomWidget(int(self.scale * 100))
        act_zoom = QWidgetAction(self)
        layout_zoomBox = QVBoxLayout()
        label_zoom = QLabel(self.tr("缩放"))
        label_zoom.setAlignment(Qt.AlignCenter)
        layout_zoomBox.addWidget(self.zoomWidget)
        layout_zoomBox.addWidget(label_zoom)
        act_zoom.setDefaultWidget(QWidget())  # 将自定义窗口放到QWidgetAction中
        act_zoom.defaultWidget().setLayout(layout_zoomBox)
        act_zoom.setEnabled(False)

        act_filterPosImage = CheckableActionWidget(self, "图片筛选", "Ctrl+G")
        act_filterPosImage.widgetClicked.connect(self.filterTargetImage)

        # 建立actions
        act_openDir = newAction(
            self,
            self.tr("打开目录"),
            self.openDirDialog,
            "Ctrl+O",
            ":/icons/open_dir.png",
            self.tr("打开目录"),
        )
        act_getPrevListId = newAction(
            self,
            self.tr("上个镜片"),
            self.getPrevListId,
            Qt.Key_Up,
            ":/icons/to_up.png",
            self.tr("打开上个镜片文件夹的图片"),
        )
        act_getNextListId = newAction(
            self,
            self.tr("下个镜片"),
            self.getNextListId,
            Qt.Key_Down,
            ":/icons/to_down.png",
            self.tr("打开下个镜片文件夹的图片"),
        )
        act_getPrevImg = newAction(
            self,
            self.tr("前张图片"),
            self.getPrevImg,
            Qt.Key_Left,
            ":/icons/to_left.png",
            self.tr("打开上一张图片"),
        )
        act_getNextImg = newAction(
            self,
            self.tr("后张图片"),
            self.getNextImg,
            Qt.Key_Right,
            ":/icons/to_right.png",
            self.tr("打开下一张图片"),
        )
        act_setFitWindow = newAction(
            self,
            self.tr("适应窗口"),
            self.setFitWindow,
            "Ctrl+R",
            ":/icons/scale_window2.png",
            self.tr("适应窗口"),
        )
        act_filterDefectImage = newAction(
            self,
            self.tr("缺陷筛选"),
            self.filterTargetImage,
            "Ctrl+F",
            ":/icons/filter.png",
            self.tr("缺陷筛选"),
            checkable=True,
        )
        act_exportFile = newAction(
            self,
            self.tr("导出文件"),
            self.exportFile,
            "",
            ":/icons/export.png",
            self.tr("导出文件"),
        )
        act_clearPanel = newAction(
            self,
            self.tr("清空面板"),
            self.clearPanel,
            "Ctrl+W",
            "",
            self.tr("清空面板"),
        )
        act_quit = newAction(
            self, self.tr("退出"), self.close, "Ctrl+Q", "", self.tr("退出")
        )

        act_setPlateGrids = newAction(self, self.tr("载物盘格子数"), self.setPlateGrids)
        act_setImageZoomRatio = newAction(
            self, self.tr("图片缩放比例"), self.setImageZoomRatio
        )
        act_setHobjDirectName = newAction(
            self, self.tr("hobj文件夹"), self.setHobjDirectName
        )
        act_setImageCopyRoot = newAction(
            self, self.tr("图片目标文件夹定义"), self.setImageCopyRoot
        )

        act_copyImage = newAction(
            self,
            self.tr("复制图片"),
            self.copyImage,
            Qt.Key_Insert,
            self.tr("复制图片"),
        )

        act_showVersionInfo = newAction(self, self.tr("版本信息"), self.showVersionInfo)
        act_showShortcutKey = newAction(self, self.tr("快捷键"), self.showShortcutKey)

        # 工具栏最右边的logo
        widget_spacer = QWidget()
        widget_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        widget_logo = QWidget()
        layout_logo = QHBoxLayout(widget_logo)
        layout_logo.setContentsMargins(10, 0, 15, 0)
        layout_logo.setSpacing(10)
        label_logo = QLabel()
        pixmap = QPixmap(":/icons/nashi.png")
        label_logo.setPixmap(
            pixmap.scaled(140, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        layout_logo.addWidget(label_logo)

        self.actions = struct(
            act_openDir=act_openDir,
            act_getPrevListId=act_getPrevListId,
            act_getNextListId=act_getNextListId,
            act_getPrevImg=act_getPrevImg,
            act_getNextImg=act_getNextImg,
            act_setFitWindow=act_setFitWindow,
            act_filterDefectImage=act_filterDefectImage,
            act_exportFile=act_exportFile,
            act_clearPanel=act_clearPanel,
            act_quit=act_quit,
            act_zoom=act_zoom,
            act_filterPosImage=act_filterPosImage,
            act_copyImage=act_copyImage,
        )

        self.toolbar.addAction(act_openDir)
        self.toolbar.addAction(act_getPrevListId)
        self.toolbar.addAction(act_getNextListId)
        self.toolbar.addAction(act_getPrevImg)
        self.toolbar.addAction(act_getNextImg)
        self.toolbar.addSeparator()
        self.toolbar.addAction(act_setFitWindow)
        self.toolbar.addAction(act_zoom)
        self.toolbar.addSeparator()
        self.toolbar.addAction(act_filterDefectImage)
        self.toolbar.addSeparator()
        self.toolbar.addAction(act_filterPosImage)
        self.toolbar.addSeparator()
        self.toolbar.addWidget(widget_spacer)
        self.toolbar.addWidget(widget_logo)

        # 状态栏
        self.label_statusDefect = QLabel()
        self.label_statusDefect.setFixedWidth(320)
        self.label_statusDefect.setContentsMargins(20, 2, 5, 2)
        self.label_statusDefect.setStyleSheet("font-size: 12px")

        self.label_statusPos = QLabel()
        self.label_statusPos.setFixedWidth(200)
        self.label_statusPos.setStyleSheet("font-size: 12px")
        self.label_statusPos.setContentsMargins(5, 2, 5, 2)

        self.statusBar().addWidget(self.label_statusDefect)
        self.statusBar().addWidget(self.label_statusPos)
        self.statusBar().setStyleSheet("QStatusBar { background-color: #585e73; }")
        self.statusBar().show()
        # 菜单栏
        self.menu_file.addAction(act_copyImage)
        self.menu_file.addAction(act_clearPanel)
        self.menu_file.addAction(act_quit)
        self.menu_setting.addAction(act_setPlateGrids)
        self.menu_setting.addAction(act_setImageZoomRatio)
        self.menu_setting.addAction(act_setHobjDirectName)
        self.menu_setting.addAction(act_setImageCopyRoot)
        self.menu_help.addAction(act_showVersionInfo)
        self.menu_help.addAction(act_showShortcutKey)

        # 连接信号
        self.widget_left.lineEdit_inputPath.textChanged.connect(
            self.inputPathTextChange
        )
        self.widget_left.pushButton_open.clicked.connect(self.openDirDialog)
        self.widget_left.pushButton_prev.clicked.connect(self.getPrevListId)
        self.widget_left.pushButton_next.clicked.connect(self.getNextListId)
        self.widget_left.listWidget_ids.itemClicked.connect(self.switchSelectedListId)
        self.widget_left.switchItem.connect(self.onSwitchItem)
        self.widget_main.pushButton_slideLeft.clicked.connect(self.getPrevImg)
        self.widget_main.pushButton_slideRight.clicked.connect(self.getNextImg)
        self.zoomWidget.valueChanged.connect(self.paintCanvas)
        self.widget_main.canvas.zoomRequest.connect(self.zoomRequest)
        self.widget_main.canvas.mouseMoved.connect(self.statusMouseCoordinates)
        self.widget_right.checkBox_draw.stateChanged.connect(self.showEquipDetectResult)

        self.resetState()

    def inputPathTextChange(self):
        lineTextStr = self.widget_left.lineEdit_inputPath.text()
        if lineTextStr:
            self.widget_left.lineEdit_inputPath.setStyleSheet("color:black;font:14px")
        else:
            # 恢复到默认样式
            self.widget_left.lineEdit_inputPath.setStyleSheet("color:gray")
        lineTextStr1 = osp.normpath(lineTextStr)
        # print(lineTextStr1)

    def onReceiveImage(self, ind, imgname, imageData):
        logger.trace(f"onReceiveImage {ind} ...")
        self.img_datas[imgname] = imageData
        if ind == 0:
            qimage = image_data_qimage_pillow(imageData)
            self.loadImage(imgname, qimage)

    def loadImage(self, imgname, qimage):
        hobj_direct_name = self.settings.value("window/hobj_direct_name", "src_hobj")

        self.widget_main.resetState()
        self.widget_main.canvas.setEnabled(False)
        self.qimage = qimage
        self.widget_main.canvas.loadPixmap(QPixmap.fromImage(self.qimage))
        if imgname is not None:
            parent_root = osp.dirname(self.targetDirPath)
            hobjname = osp.splitext(imgname)[0] + ".hobj"
            hobjpath = osp.join(
                parent_root, hobj_direct_name, self.cur_direct, hobjname
            )
            polygon_list = load_hobj_polygons(hobjpath)
            self.loadLabels(polygon_list)
        self.widget_main.canvas.setEnabled(True)
        logger.info(f"cur_img_index: {self.cur_img_index}")
        if self.cur_img_index == 0:
            self.adjustScale(force_scale=True)
        else:
            self.adjustScale(force_scale=False)
        self.paintCanvas()

    def loadLabels(self, polygon_list):
        shapes = []
        img_zoom_ratio_str = self.settings.value("window/img_zoom_ratio", "1.0")
        img_zoom_ratio = float(img_zoom_ratio_str)
        for polygon in polygon_list:
            label = polygon[0]
            points = polygon[1]
            points1 = []
            for x, y in points:
                points1.append(QPointF(x, y))
            shape = Shape(points1, label, img_zoom_ratio)
            shape.close()
            shapes.append(shape)
        self.widget_main.canvas.loadShapes(shapes)

    def openDirDialog(self):
        # defaultOpenDirPath = "D:/my_code/pyqt5_proj2/ceshi_ids/src_img"
        defaultOpenDirPath = self.settings.value("window/data_root", "D:/")
        OPEN_DIR_FLAG = 1
        if OPEN_DIR_FLAG:
            self.targetDirPath = str(
                QFileDialog.getExistingDirectory(
                    self,
                    self.tr("Open Directory"),
                    defaultOpenDirPath,
                    QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
                )
            )
        else:
            self.targetDirPath = defaultOpenDirPath

        if self.targetDirPath == "":
            return

        if self.widget_left.hobj_thread.isRunning():
            self.widget_left.hobj_thread.terminate()

        self.resetState()

        self.settings.setValue("window/data_root", self.targetDirPath)
        logger.info(f"targetDirPath: {self.targetDirPath}")

        if not osp.exists(self.targetDirPath):
            logger.error(f"目录不存在: {self.targetDirPath}")
            QMessageBox.warning(self, "错误", f"目录不存在: {self.targetDirPath}")
            self.resetState()
            return

        logger.info(f"open directory : {self.targetDirPath}")
        targetFileDir = osp.dirname(self.targetDirPath)
        self.label_file = LabelFile(targetFileDir)
        self.label_file.loadManuCheckResult()

        mesg_ids = self.label_file.manu_check_result.keys()

        self.widget_left.initIdListBoard(self.targetDirPath, mesg_ids)
        self.toggleActions(True)
        self.widget_left.hobj_thread.startLoad.connect(self.onIdListBoardLoaded)

    def openCurrentIdDirect(self):
        if self.cur_direct is None:
            return
        # 获取面板上的prev_direct的信息，并更新到json文件中
        prev_check_edit_dict = self.acquireCheckBoardIdDict()
        logger.info(f"prev_direct:{self.prev_direct} cur_direct:{self.cur_direct}")
        self.label_file.updateCheckIdDict(self.prev_direct, prev_check_edit_dict)

        # 将id_listwidget控件中的prev_direct的item的状态更新
        if self.prev_direct is not None:
            mesg_flag = self.label_file.judgeCheckDictHasMesg(prev_check_edit_dict)
            self.widget_left.updateIdListWidgetMesg(self.prev_direct, mesg_flag)

        self.img_datas.clear()
        if self.image_thread.isRunning():
            self.image_thread.terminate()
        cur_direct_path = osp.join(self.targetDirPath, self.cur_direct)
        filenames = os.listdir(cur_direct_path)
        imgnames = filter_filename_by_extension(filenames, [".jpg", ".bmp"])
        imgnames = natsorted(imgnames)

        if self.cur_direct in self.widget_left.defect_indexs_total.keys():
            defect_indexs = self.widget_left.defect_indexs_total[self.cur_direct]
        else:
            defect_indexs = []

        defect_filter_flag = self.actions.act_filterDefectImage.isChecked()
        pos_filter_flag = self.actions.act_filterPosImage.isChecked()
        if defect_filter_flag:
            if len(defect_indexs) != len(imgnames):
                logger.error(
                    f"{self.cur_direct} defect_indexs.size not equal to imgnames.size!"
                )
            imgnames2 = []
            for i, defect_ind in enumerate(defect_indexs):
                if len(defect_ind) < 1:
                    continue
                if defect_ind[0] == -1:
                    continue
                imgnames2.append(imgnames[i])
            imgnames = imgnames2

        if pos_filter_flag:
            imgnames2 = []
            keyword = self.actions.act_filterPosImage.getLineEditText()
            logger.info(f"keyword: {keyword}")
            for imgname in imgnames:
                if keyword in imgname:
                    imgnames2.append(imgname)
            imgnames = imgnames2

        img_num = len(imgnames)
        # print(f"img_num: {img_num}")
        if img_num < 1:
            self.loadImage(None, QImage())
            # return

        for imgname in imgnames:
            self.img_datas[imgname] = bytes()

        self.image_thread = ImageLoader(cur_direct_path, imgnames)
        self.image_thread.sendImage.connect(self.onReceiveImage)
        self.image_thread.finishLoad.connect(self.onFinishLoadImages)
        self.image_thread.start()

        # 将label_file中的manu信息更新到右侧的面板上
        cur_check_raw_dict = self.label_file.getCheckIdDict(self.cur_direct)
        logger.trace(f"{self.cur_direct}: {cur_check_raw_dict}")
        self.widget_right.updateCheckListBoard(
            self.cur_direct, defect_indexs, cur_check_raw_dict
        )

        self.cur_img_index = 0

    def onSwitchItem(self, cur_item_text, prev_item_text):
        logger.trace("onSwitchItem...")
        self.cur_direct = cur_item_text
        self.prev_direct = prev_item_text

    def getPrevListId(self):
        logger.trace("getPrevListId...")
        list_num = self.widget_left.listWidget_ids.count()
        if list_num < 1:
            return
        index = self.widget_left.listWidget_ids.currentIndex().row()
        if index == 0:
            return
        index2 = clip_num(index - 1, 0, list_num - 1)
        self.widget_left.highLightListWidgetId(False)
        self.widget_left.listWidget_ids.setCurrentRow(index2)
        self.openCurrentIdDirect()

    def getNextListId(self):
        logger.trace("getNextListId...")
        list_num = self.widget_left.listWidget_ids.count()
        if list_num < 1:
            return
        index = self.widget_left.listWidget_ids.currentIndex().row()
        if index == list_num - 1:
            return
        index2 = clip_num(index + 1, 0, list_num - 1)
        self.widget_left.highLightListWidgetId(False)
        self.widget_left.listWidget_ids.setCurrentRow(index2)
        self.openCurrentIdDirect()

    def switchSelectedListId(self):
        self.openCurrentIdDirect()

    def getPrevImg(self):
        logger.trace("getPrevImg...")
        img_num = len(self.img_datas)
        if img_num < 1:
            return
        cur_img_num = 0
        imgnames = natsorted(list(self.img_datas.keys()))
        # 在切换id_direct的时候，实时的计算当前已加载的图片的数量cur_img_num
        for imgname in imgnames:
            if self.img_datas[imgname] != bytes():
                cur_img_num += 1
        new_img_index = cycle_index(self.cur_img_index - 1, 0, cur_img_num - 1)
        imgname = imgnames[new_img_index]
        imageData = self.img_datas[imgname]
        qimage = image_data_qimage_pillow(imageData)
        self.cur_img_index = new_img_index
        self.loadImage(imgname, qimage)

    def getNextImg(self):
        logger.trace("getNextImg...")
        img_num = len(self.img_datas)
        if img_num < 1:
            return
        cur_img_num = 0
        imgnames = natsorted(list(self.img_datas.keys()))
        for imgname in imgnames:
            if self.img_datas[imgname] != bytes():
                cur_img_num += 1
        new_img_index = cycle_index(self.cur_img_index + 1, 0, cur_img_num - 1)
        imgname = imgnames[new_img_index]
        imageData = self.img_datas[imgname]
        qimage = image_data_qimage_pillow(imageData)
        self.cur_img_index = new_img_index
        self.loadImage(imgname, qimage)

    def acquireCheckBoardIdDict(self):
        logger.trace("acquireCheckBoardIdDict...")
        manu_defects = []
        for i in range(self.widget_right.listWidget_manuRlt.count()):
            manu_defect = self.widget_right.listWidget_manuRlt.item(i).text()
            manu_defects.append(manu_defect)
        manu_defects = natsorted(manu_defects)
        remarks = []
        remark_text = self.widget_right.textEdit_comment.toPlainText().strip()
        for remark in remark_text.split("\n"):
            remarks.append(remark)
        check_raw_dict = {"manu_defects": manu_defects, "remarks": remarks}
        return check_raw_dict

    def statusMouseCoordinates(self, pos):
        if pos.x() < 0 and pos.y() < 0:
            # self.statusBar().showMessage("", 1000)
            self.label_statusPos.clear()
        else:
            pos_str = "x=%.2f, y=%.2f" % (pos.x(), pos.y())
            # self.statusBar().showMessage(pos_str, 1000)
            self.label_statusPos.setText(pos_str)

    def filterTargetImage(self):
        logger.trace("filterTargetImage...")
        defect_filter_flag = self.actions.act_filterDefectImage.isChecked()
        pos_filter_flag = self.actions.act_filterPosImage.isChecked()

        self.img_datas.clear()
        if self.image_thread.isRunning():
            self.image_thread.terminate()
        cur_direct_path = osp.join(self.targetDirPath, self.cur_direct)
        filenames = os.listdir(cur_direct_path)
        imgnames = filter_filename_by_extension(filenames, [".jpg", ".bmp"])
        imgnames = natsorted(imgnames)

        if self.cur_direct in self.widget_left.defect_indexs_total.keys():
            defect_indexs = self.widget_left.defect_indexs_total[self.cur_direct]
        else:
            defect_indexs = []

        if defect_filter_flag:
            # print(self.cur_direct, defect_indexs)
            if len(defect_indexs) != len(imgnames):
                logger.error(
                    f"{self.cur_direct} defect_indexs.size not equal to imgnames.size!"
                )
            imgnames2 = []
            for i, defect_ind in enumerate(defect_indexs):
                if len(defect_ind) < 1:
                    continue
                if defect_ind[0] == -1:
                    continue
                imgnames2.append(imgnames[i])
            imgnames = imgnames2

        if pos_filter_flag:
            imgnames2 = []
            keyword = self.actions.act_filterPosImage.getLineEditText().strip()
            logger.info(f"keyword: {keyword}")
            if keyword != "":
                for imgname in imgnames:
                    if keyword in imgname:
                        imgnames2.append(imgname)
                imgnames = imgnames2

        img_num = len(imgnames)
        # print(f"img_num: {img_num}")
        if img_num < 1:
            self.loadImage(None, QImage())
            return

        for imgname in imgnames:
            self.img_datas[imgname] = bytes()

        self.image_thread = ImageLoader(cur_direct_path, imgnames)
        self.image_thread.sendImage.connect(self.onReceiveImage)
        self.image_thread.finishLoad.connect(self.onFinishLoadImages)
        self.image_thread.start()

        self.cur_img_index = 0

    def filterPosImage(self, isChecked):
        logger.info("filterPosImage...")

    def exportFile(self):
        logger.trace("exportFile...")

    def clearPanel(self):
        logger.trace("clearPanel...")

        if self.widget_left.hobj_thread.isRunning():
            self.widget_left.hobj_thread.terminate()

        cur_check_edit_dict = self.acquireCheckBoardIdDict()
        if self.label_file is not None:
            self.label_file.updateCheckIdDict(self.cur_direct, cur_check_edit_dict)
        self.resetState()

    def toggleActions(self, value):
        self.actions.act_getPrevListId.setEnabled(value)
        self.actions.act_getNextListId.setEnabled(value)
        self.actions.act_getPrevImg.setEnabled(value)
        self.actions.act_getNextImg.setEnabled(value)
        self.actions.act_setFitWindow.setEnabled(value)
        self.actions.act_filterDefectImage.setEnabled(value)
        self.actions.act_exportFile.setEnabled(value)
        self.actions.act_zoom.setEnabled(value)
        self.actions.act_filterPosImage.setEnabled(value)
        self.actions.act_copyImage.setEnabled(value)
        if not value:
            self.actions.act_filterDefectImage.setChecked(value)
            self.actions.act_filterPosImage.setChecked(value)

    def resetState(self):
        self.qimage = None
        self.widget_left.resetState()
        self.widget_main.resetState()
        self.widget_right.resetState()
        self.label_statusDefect.clear()
        self.label_statusPos.clear()
        self.toggleActions(False)

    def setFitWindow(self):
        self.adjustScale(force_scale=True)

    def scaleFitWindow(self):
        e = 0.0
        w1 = self.widget_main.scrollArea.width() - e
        h1 = self.widget_main.scrollArea.height() - e
        a1 = w1 / h1
        w2 = self.widget_main.canvas.pixmap.width() - 0.0
        h2 = self.widget_main.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        scale = w1 / w2 if a2 >= a1 else h1 / h2
        return scale

    def addZoom(self, increment=1.1):
        new_scale = self.scale * increment
        if increment > 1:
            zoom_value = math.ceil(new_scale * 100)
        else:
            zoom_value = math.floor(new_scale * 100)
        self.zoomWidget.setValue(zoom_value)
        self.scale = new_scale

    def zoomRequest(self, delta, pos):
        logger.trace(
            f"{inspect.currentframe().f_code.co_filename} {inspect.currentframe().f_lineno}"
        )
        canvas_width_old = self.widget_main.canvas.width()
        units = 1.1
        if delta < 0:
            units = 0.9
        self.addZoom(units)

        canvas_width_new = self.widget_main.canvas.width()
        scrollArea_width = self.widget_main.scrollArea.width()
        tab_window_width = self.widget_main.width()
        logger.trace(
            f"tab_window_width:{tab_window_width} scrollArea_width:{scrollArea_width} canvas_width_old:{canvas_width_old} canvas_width_new:{canvas_width_new}"
        )
        if canvas_width_old != canvas_width_new:
            canvas_scale_factor = canvas_width_new / canvas_width_old

            x_shift = round(pos.x() * canvas_scale_factor) - pos.x()
            y_shift = round(pos.y() * canvas_scale_factor) - pos.y()

            bar_x_value0 = self.widget_main.scrollBars[Qt.Horizontal].value()
            bar_x_value1 = bar_x_value0 + x_shift
            # print(f'bar_x_value0:{bar_x_value0} x_shift:{x_shift} bar_x_value1:{bar_x_value1}')
            self.widget_main.scrollBars[Qt.Horizontal].setValue(int(bar_x_value1))

            bar_y_value0 = self.widget_main.scrollBars[Qt.Vertical].value()
            bar_y_value1 = bar_y_value0 + y_shift
            # print(f'bar_y_value0:{bar_y_value0} y_shift:{y_shift} bar_y_value1:{bar_y_value1}')
            self.widget_main.scrollBars[Qt.Vertical].setValue(int(bar_y_value1))

    def paintCanvas(self):
        # 检查图像是否为空
        if self.qimage.isNull():
            # logger.warning("警告: 图像为空")
            return

        self.widget_main.canvas.scale = self.scale
        self.widget_main.canvas.adjustSize()
        self.widget_main.setSlideButtonState()

        if (
            self.cur_img_index >= 0
            and len(self.img_datas) > self.cur_img_index
            and self.cur_direct
        ):
            imgnames = natsorted(list(self.img_datas.keys()))
            cur_imgname = imgnames[self.cur_img_index]
            cur_img_size = [self.qimage.width(), self.qimage.height()]
            img_num = len(self.img_datas)
            try:
                cur_id = int(self.cur_direct.split("_")[1])
                self.widget_main.updateImgMesgLabel(
                    cur_imgname, self.cur_img_index, img_num, cur_img_size, cur_id
                )
            except (IndexError, ValueError) as e:
                logger.error(f"解析镜片ID时出错: {e}")

        self.widget_main.canvas.update()

    def adjustScale(self, force_scale=True):
        logger.trace(f"adjustScale... scale:{self.scale}")
        if self.widget_main.canvas.pixmap.isNull():
            self.scale = 1.0
        if force_scale:
            self.scale = self.scaleFitWindow()

        value = int(round(100 * self.scale))
        self.zoomWidget.setValue(value)

    def showEquipDetectResult(self):
        state = self.widget_right.checkBox_draw.checkState()
        value = state == Qt.Checked
        self.widget_main.canvas.setShapeVisible(value)

    def onIdListBoardLoaded(self):
        if self.widget_left.listWidget_ids.count() > 0:
            self.widget_left.hobj_thread.startLoad.disconnect(self.onIdListBoardLoaded)
            self.cur_direct = self.widget_left.listWidget_ids.item(0).text()
            self.widget_left.listWidget_ids.setCurrentRow(0)
            # self.widget_left.listWidget_ids.update()
            self.openCurrentIdDirect()

    def onFinishLoadImages(self):
        logger.trace("onFinishLoadImages...")
        logger.info(f"cur_direct:{self.cur_direct} img_length:{len(self.img_datas)}")
        self.image_thread.quit()
        self.widget_left.highLightListWidgetId(True)

    def setPlateGrids(self):
        logger.trace("setPlateGrids...")
        grid_size_str = self.settings.value("window/grid_size", "5")
        dialog = PlateGridSizeInputDialog(self, grid_size_str)
        if dialog.exec_() == QDialog.Accepted:
            grid_size = int(dialog.number_input.text())
            logger.info(f"grid_size: {grid_size}")
            self.settings.setValue("window/grid_size", str(grid_size))

    def setImageZoomRatio(self):
        logger.trace("setImageZoomRatio...")
        img_zoom_ratio_str = self.settings.value("window/img_zoom_ratio", "1.0")
        dialog = ImageZoomRatioDialog(self, img_zoom_ratio_str)
        if dialog.exec_() == QDialog.Accepted:
            img_zoom_ratio_str = dialog.lineEditZoomRatio.text()
            logger.info(f"img_zoom_ratio: {img_zoom_ratio_str}")
            self.settings.setValue("window/img_zoom_ratio", img_zoom_ratio_str)

    def setHobjDirectName(self):
        logger.trace("setHobjDirectName...")
        hobj_direct_name = self.settings.value("window/hobj_direct_name", "src_hobj")
        dialog = HobjDirectNameDialog(self, hobj_direct_name)
        if dialog.exec_() == QDialog.Accepted:
            hobj_direct_name = dialog.lineEditDirectName.text()
            logger.info(f"hobj_direct_name: {hobj_direct_name}")
            self.settings.setValue("window/hobj_direct_name", hobj_direct_name)

            if self.widget_left.hobj_thread.isRunning():
                self.widget_left.hobj_thread.terminate()
            self.resetState()

            mesg_ids = self.label_file.manu_check_result.keys()
            self.widget_left.initIdListBoard(self.targetDirPath, mesg_ids)
            self.toggleActions(True)
            self.widget_left.hobj_thread.startLoad.connect(self.onIdListBoardLoaded)

    def setImageCopyRoot(self):
        logger.info("setImageCopyRoot...")
        pictures_path = os.path.join(os.path.expanduser("~"), "Pictures")
        logger.info(f"pictures_path: {pictures_path}")
        dst_img_root = self.settings.value("window/dst_img_root", pictures_path)
        logger.info(f"dst_img_root: {dst_img_root}")
        dialog = ImageCopyRootDialog(self, dst_img_root)
        if dialog.exec_() == QDialog.Accepted:
            dst_img_root = dialog.text_edit.text()
            logger.info(f"dst_img_root: {dst_img_root}")
            self.settings.setValue("window/dst_img_root", dst_img_root)

    def copyImage(self):
        logger.info("copyImage...")
        if (
            self.widget_main.canvas.pixmap is None
            or self.widget_main.canvas.pixmap.isNull()
        ):
            return
        imgnames = natsorted(list(self.img_datas.keys()))
        cur_imgname = imgnames[self.cur_img_index]

        pictures_path = os.path.join(os.path.expanduser("~"), "Pictures")
        dst_img_root = self.settings.value("window/dst_img_root", pictures_path)
        if not osp.exists(dst_img_root):
            os.makedirs(dst_img_root)

        scroll_width = self.widget_main.scrollArea.width()
        copy_text = f"{cur_imgname} 复制到 目标文件夹"
        self.widget_main.label_copyMesg.clear()
        self.widget_main.label_copyMesg.setText(copy_text)
        self.widget_main.label_copyMesg.adjustSize()
        label_width = self.widget_main.label_copyMesg.width()
        self.widget_main.label_copyMesg.move(scroll_width - label_width - 20, 20)
        self.widget_main.label_copyMesg.show()
        QTimer.singleShot(1000, self.widget_main.label_copyMesg.hide)

        cur_imgpath = osp.join(self.targetDirPath, self.cur_direct, cur_imgname)
        logger.trace(f"cur_imgpath: {cur_imgpath}")
        dst_imgpath = osp.join(dst_img_root, cur_imgname)
        shutil.copyfile(cur_imgpath, dst_imgpath)
        logger.info(f"copy {cur_imgname} to {dst_img_root}")

    def showVersionInfo(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.NoIcon)
        msg.setWindowTitle("版本信息")
        msg.setText(VERSION_STR)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def showShortcutKey(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.NoIcon)
        msg.setWindowTitle("快捷键")
        msg.setText(SHORTCUT_STR)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def resizeEvent(self, event):
        super().resizeEvent(event)

        logger.trace("resizeEvent...")
        current_size = self.size()
        if current_size != self.last_size and not self.qimage.isNull():
            self.last_size = current_size
            self.adjustScale(force_scale=True)
            self.paintCanvas()

    def closeEvent(self, event):
        logger.trace("close")
        cur_check_edit_dict = self.acquireCheckBoardIdDict()
        if self.label_file is not None:
            self.label_file.updateCheckIdDict(self.cur_direct, cur_check_edit_dict)

        super().closeEvent(event)


if __name__ == "__main__":

    if DEBUG:
        log_type = "INFO"
    else:
        log_type = "WARNING"
    init_logu(log_type)

    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
