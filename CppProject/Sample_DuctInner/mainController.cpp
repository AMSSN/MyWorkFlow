#include "mainController.h"


MainController::MainController(QMainWindow* parent) : QMainWindow(parent), ui(new Ui::MainWindow)
{
	// 初始化界面
	ui->setupUi(this);
	beautifyUI();
	//读取配置文件
	load_config_file("./config.json");
	camThread = new OpencvThreadedCamera(0, this); // 使用默认摄像头
	// 检测模型初始化
	// -------- model1 --------
	cfm.model_path = config_global.model_path;
	cfm.infer_size = config_global.infer_size;
	cfm.postproc_type = config_global.postprocess_type;
	cfm.conf_threshold = config_global.conf;
	cfm.nms_threshold = 0.3;
	cfm.batch_size = 1;
	spdlog::info("YoloDetecter path = {}", cfm.model_path);
	// 1 for fp16 , 2 for int8;
	int ret = generate_engine_model(cfm.model_path, cfm.batch_size, cfm.infer_size, 1);
	if (0 != ret)
	{
		spdlog::error("!!!!!!!!-YoloDetecter generate engine model failed!");
	}
	this->yolodet = new YOLODetector();
	ret = this->yolodet->model_init(cfm.model_path, 0);
	if (0 != ret)
	{
		spdlog::error("!!!!!!!!-YoloDetecter not initialized!");
	}
	spdlog::info("########-YoloDetecter was initialized.");


	//load_configfile("GlobalConfig.json"); // 加载全局配置文件
	//// 检测类
	//beadDetector = new M_BeadDetect();
	//beadDetector->fifoList = FifoList::getInstance();
	//// 相机类
	//camManager = new CameraManager();
	//camManager->run_all();
	//// 加载本地文件类
	//thread_loadimg = new QThread(this);


	//// 绑定控件和函数
	connect(ui->pushButton_start, SIGNAL(clicked()), this, SLOT(start_detect()));
	connect(ui->pushButton_pause, SIGNAL(clicked()), this, SLOT(pause_detect()));
	connect(ui->pushButton_stop, SIGNAL(clicked()), this, SLOT(stop_detect()));
	connect(ui->pushButton_count, SIGNAL(clicked()), this, SLOT(count_sample()));

	connect(camThread, SIGNAL(frameReady(const cv::Mat&)), this, SLOT(Image_process(const cv::Mat&)));
	//connect(ui->action_detect, SIGNAL(triggered()), this, SLOT(change_widget_to_detect()));
	//connect(ui->action_data, SIGNAL(triggered()), this, SLOT(change_widget_to_data()));
	//connect(ui->action_setting, SIGNAL(triggered()), this, SLOT(change_widget_to_setting()));
	//// setting界面-switchbutton
	//connect(ui->switchButton_algorithm, SIGNAL(checkedChanged(bool)), this, SLOT(change_algorithm_stage(bool)));
	//connect(ui->switchButton_localimg, SIGNAL(checkedChanged(bool)), this, SLOT(change_localimg_stage(bool)));
	//// info页面
	//connect(ui->pushButton_temp1, SIGNAL(clicked()), this, SLOT(temp1_soft_trigger()));
	////connect(cameraThread, SIGNAL(captured(QImage)), this, SLOT(Image_process(QImage))); // 相机采图
	//// setting页面
	//connect(ui->pushButton_set_localimg_path, SIGNAL(clicked()), this, SLOT(set_localimg_path()));
	//// detect页面
	//connect(beadDetector, SIGNAL(detect_over(cv::Mat, int, int)), this, SLOT(show_detect_result_in_listwidget(cv::Mat, int, int)));
	//connect(beadDetector, SIGNAL(detect_num(int)), ui->progressRound_detectnum, SLOT(setValue(int)));
	//// connect(beadDetector, SIGNAL(detect_num(int)), ui->progressRound_detectnum, SLOT(setValue(int)));
	//// data页面
	//// 查询和导出
	//connect(ui->pushButton_selectdata, SIGNAL(clicked()), this, SLOT(select_data_from_sql()));
	//connect(ui->pushButton_exportdata, SIGNAL(clicked()), this, SLOT(export_data_from_sql()));

	camThread->stream();
}
MainController::~MainController()
{
	delete ui;
}

void MainController::beautifyUI() {
	// 图标和logo
	setWindowTitle("内壁检测系统 v1.0");
	setWindowIcon(QIcon("./resources/LeiLei.ico"));
	QPixmap pixmap("./resources/slogan.png");
	ui->label_logo->setPixmap(pixmap.scaled(200, 72, Qt::KeepAspectRatio, Qt::SmoothTransformation));
	// 启动按钮
	ui->pushButton_start->setText("启动");
	ui->pushButton_start->setIcon(QIcon("./resources/start.png"));
	ui->pushButton_start->setIconSize(ui->pushButton_start->size());
	ui->pushButton_start->setStyleSheet(R"(
    QPushButton {
		font-size: 15px;
        border: 2px groove gray;                   /* 无边框 */
		border-radius: 20px;
        background: transparent;      /* 透明背景 */
        padding: 3px;                  /* 无内边距 */
        margin: 3px;                   /* 无外边距 */
    }
    QPushButton:hover {
        background: #e0e0e0;           /* 悬停效果 */
        border-radius: 20px;
    }
    QPushButton:pressed {
        background: #c0c0c0;           /* 按下效果 */
    }
)");
	// 暂停按钮
	ui->pushButton_pause->setText("暂停");
	ui->pushButton_pause->setIcon(QIcon("./resources/pause.png"));
	ui->pushButton_pause->setIconSize(ui->pushButton_pause->size());
	ui->pushButton_pause->setStyleSheet(R"(
    QPushButton {
		font-size: 15px;
        border: 2px groove gray;                   /* 无边框 */
		border-radius: 20px;
        background: transparent;      /* 透明背景 */
        padding: 3px;                  /* 无内边距 */
        margin: 3px;                   /* 无外边距 */
    }
    QPushButton:hover {
        background: #e0e0e0;           /* 悬停效果 */
        border-radius: 20px;
    }
    QPushButton:pressed {
        background: #c0c0c0;           /* 按下效果 */
    }
)");
	// 停止按钮
	ui->pushButton_stop->setText("停止");
	ui->pushButton_stop->setIcon(QIcon("./resources/stop.png"));
	ui->pushButton_stop->setIconSize(ui->pushButton_stop->size());
	ui->pushButton_stop->setStyleSheet(R"(
    QPushButton {
		font-size: 15px;
        border: 2px groove gray;                   /* 无边框 */
		border-radius: 20px;
        background: transparent;      /* 透明背景 */
        padding: 3px;                  /* 无内边距 */
        margin: 3px;                   /* 无外边距 */
    }
    QPushButton:hover {
        background: #e0e0e0;           /* 悬停效果 */
        border-radius: 20px;
    }
    QPushButton:pressed {
        background: #c0c0c0;           /* 按下效果 */
    }
)");
	// 计数按钮
	ui->pushButton_count->setText("计数");
	ui->pushButton_count->setIcon(QIcon("./resources/data.png"));
	ui->pushButton_count->setIconSize(QSize(30, 30));
	ui->pushButton_count->setStyleSheet(R"(
    QPushButton {
		font-size: 20px;
        border: 2px groove gray;                   /* 无边框 */
		border-radius: 20px;
        background: transparent;      /* 透明背景 */
        padding: 3px;                  /* 无内边距 */
        margin: 3px;                   /* 无外边距 */
    }
    QPushButton:hover {
        background: #e0e0e0;           /* 悬停效果 */
        border-radius: 20px;
    }
    QPushButton:pressed {
        background: #c0c0c0;           /* 按下效果 */
    }
	)");

	// 计数器
	ui->label_cam_state->setStyleSheet(R"(
    QLabel {
        font-size: 18px;              /* 大字号 */
        font-family: "DIN Alternate";  /* 等宽数字字体 */
        font-weight: bold;
        background: transparent;      /* 透明背景 */
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        padding: 4px 12px;
		}
	)");
	ui->label_cam_state->setText("样本数量:");

	

}

void MainController::start_detect() {
	spdlog::info("MainController::start_detect()");
	camThread->stream();
}

void MainController::pause_detect() {
	spdlog::info("MainController::pause_detect()");
	camThread->pause();
}

void MainController::stop_detect() {
	spdlog::info("MainController::stop_detect()");
	camThread->stop();
}

void MainController::count_sample() {
	SampleNumber = SampleNumber+1;
	ui->label_cam_state->setText("样本数量:"+QString::number(SampleNumber));
	int randomInt = QRandomGenerator::global()->bounded(100);  // 0~99
	if (0<= randomInt<=70) {
		config_global.ok_num ++;
		ui->label_ok_num->setText("OK："+QString::number(config_global.ok_num));
	}
	else if (70 < randomInt <= 80) {
		config_global.ng_num++;
		ui->label_ng_num->setText("NG：" + QString::number(config_global.ng_num));
	}
	else {
		config_global.dpd_num++;
		ui->label_sample_index->setText("待判定：" + QString::number(config_global.dpd_num));
	}
	//// 刷新相机的流程
	//camThread->stop();
	//// 等待线程完全结束
	//if (camThread->wait(3000)) {  // 等待最多3秒
	//	// 线程已结束，重新启动相机
	//	if (camThread->stream()) {
	//		qDebug() << "Camera refreshed successfully";
	//	}
	//	else {
	//		qDebug() << "Failed to restart camera after refresh";
	//	}
	//}
	//else {
	//	// 线程没有正常结束，强制终止
	//	camThread->terminate();
	//	camThread->wait();  // 确保线程结束

	//	// 重新启动相机
	//	if (camThread->stream()) {
	//		qDebug() << "Camera refreshed successfully after force termination";
	//	}
	//	else {
	//		qDebug() << "Failed to restart camera after refresh";
	//	}
	//}
}

void MainController::Image_process(const cv::Mat frame)
{
	//cv::Mat resized;
	//cv::resize(frame, resized, cv::Size(ui->label_img->width(), ui->label_img->height()), 0, 0, cv::INTER_LINEAR);
	//int height = resized.rows;
	//int width = resized.cols;
	//int bytes_per_line = resized.step;
	//QImage q_img(resized.data, width, height, bytes_per_line, QImage::Format_BGR888);
	//ui->label_img->setPixmap(QPixmap::fromImage(q_img));


	// 1. 计算等比例缩放后的尺寸
	int labelW = ui->label_img->width();
	int labelH = ui->label_img->height();
	double scaleW = (double)labelW / frame.cols;
	double scaleH = (double)labelH / frame.rows;
	double scale = std::min(scaleW, scaleH);  // 取小值，确保完整显示
	int newW = static_cast<int>(frame.cols * scale);
	int newH = static_cast<int>(frame.rows * scale);
	// 2. 等比例缩放
	cv::Mat resized;
	cv::resize(frame, resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);
	// 3. BGR → RGB（Qt 用 RGB 顺序）
	cv::Mat rgb;
	cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
	// 4. QImage 深拷贝（安全！）
	QImage q_img(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
	QImage safe_img = q_img.copy();  // ✅ 深拷贝，脱离 cv::Mat 内存
	// 5. 显示
	ui->label_img->setPixmap(QPixmap::fromImage(safe_img));
	ui->label_img->setAlignment(Qt::AlignCenter);  // 居中显示


	//int camera_index = (cameraThread->global_index - 1) % appconfig.PictureNum;
	//spdlog::info("ViewController::Image_process:{}", camera_index);
	//ui->videoPanel->set_video(camera_index, img);
	//cv::Mat cvimg = qimageToMat(img);
	//// 入队
	//fifoNode* temp_node = new fifoNode();
	//temp_node->image = cvimg.clone();
	//temp_node->index = camera_index;
	//QString time;
	//time = QDateTime::currentDateTime().toString("yyyyMMddhhmmss");
	//temp_node->time = time;
	//beadDetector->fifoList->enQueue(temp_node);
}