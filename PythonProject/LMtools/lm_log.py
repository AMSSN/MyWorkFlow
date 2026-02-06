import logging
from logging.handlers import TimedRotatingFileHandler
import time
import os

# 定义log文件存放路径
log_path = "log"  # 在哪里import，这里的log_path就是相对与哪里的路径


def log_init():
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    # 创建一个 logger
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)

    # 创建 TimedRotatingFileHandler，并设置文件名、滚动间隔和保留日志文件个数
    file_handler = TimedRotatingFileHandler(filename=log_path + "/logfile.log", when="midnight", interval=1,
                                            backupCount=7, encoding='utf-8')
    file_handler.suffix = "%Y-%m-%d.log"
    file_handler.encoding = "utf-8"
    file_handler.setFormatter(
        logging.Formatter("[%(filename)s:%(lineno)d]-[%(asctime)s - %(levelname)s]-[%(funcName)s]-\t%(message)s "))
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    # 创建控制台处理器，输出日志到控制台
    console_handler = logging.StreamHandler()
    console_handler.encoding = "utf-8"
    console_handler.setFormatter(
        logging.Formatter("[%(filename)s:%(lineno)d]-[%(asctime)s - %(levelname)s]-[%(funcName)s]-\t%(message)s "))
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)

    # 输出当前时间
    timestamp = time.time()
    current_time = time.localtime(timestamp)
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)
    logger.info("------------------------------Logging Inited!------------------------------")
    logger.info(f"--------------------Current time: {formatted_time}--------------------")
    return logger
