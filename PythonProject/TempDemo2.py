import os
import shutil
from pathlib import Path

if __name__ == '__main__':
    img_path = r"E:\FAU\1102_feipai\20261120"
    # 1、新建文件夹
    img_path = Path(img_path)
    feipai_path = img_path.parent
    print(feipai_path)
    folder_name = 11200000
    for file in img_path.iterdir():
        folder_name+=1
        print(file)
        folder_path = feipai_path/str(folder_name)
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)
        shutil.copy2(file, folder_path)
        os.rename(os.path.join(folder_path,file.name), os.path.join(folder_path,str(folder_name)+"_P0_I0.jpg"))

