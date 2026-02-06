import sqlite3
import os


def init_db():
    try :
        if not os.path.exists(os.path.join(os.getcwd(), 'dataset.db')):
            # 连接到 SQLite 数据库，如果文件不存在会创建
            with sqlite3.connect('dataset.db') as conn:
                cursor = conn.cursor()  # 创建一个游标对象
            return "success"
        else:
            return "exist"

    except Exception as e:
        return str(e)

if __name__ == '__main__':
    print("hello world")
