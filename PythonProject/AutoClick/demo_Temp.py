"""列出本机所有逻辑盘符（Windows）。"""

import ctypes
import string

DRIVE_TYPES = {
    0: "未知",
    1: "无效根目录",
    2: "可移动磁盘",
    3: "本地磁盘",
    4: "网络驱动器",
    5: "光驱",
    6: "RAM 磁盘",
}


def get_logical_drives():
    bitmask = ctypes.windll.kernel32.GetLogicalDrives()
    if bitmask == 0:
        raise ctypes.WinError()
    return [
        f"{letter}:\\"
        for i, letter in enumerate(string.ascii_uppercase)
        if bitmask >> i & 1
    ]


def get_drive_type(drive):
    return DRIVE_TYPES.get(ctypes.windll.kernel32.GetDriveTypeW(drive), "未知")


if __name__ == "__main__":
    drives = get_logical_drives()
    print(f"共发现 {len(drives)} 个盘符：")
    for drive in drives:
        print(f"  {drive}  {get_drive_type(drive)}")
