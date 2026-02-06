__version__ = "1.0.0"
__author__ = "JLM"

# 只能导入一次该文件夹  from tools import *


from .globalM import set_value, get_value
from .lm_log import log_path, log_init
from .lm_sql import init_db
from .lm_tools import OrderedSetList
from .LicenseManager import LicenseManager
import os

# 初始化相关工具
logger = log_init()
res = init_db()
logger.info(f"init SQL {res}")

__all__ = ["os",  # 通用库
           "log_path", "logger",  # log相关
           "set_value", "get_value",  # tools相关
           "OrderedSetList",  # tools相关
           "LicenseManager",  # 授权相关
           ]
