import os
import json
import base64
from datetime import datetime
from typing import Dict, List, Optional

try:
    import wmi
    import pythoncom
    from pyDes import des, CBC, PAD_PKCS5

    HAS_CRYPTO_LIBS = True
except ImportError:
    HAS_CRYPTO_LIBS = False


class LicenseManager:
    """
    独立的许可证管理类，负责机器码生成、注册码验证和加密解密功能
    可嵌入QT界面或其他任何Python程序中
    """

    @staticmethod
    def check_dependencies() -> Dict[str, bool]:
        """检查所需依赖库是否已安装"""
        # globals() 是 Python 内置函数，用于获取一个包含当前全局命名空间中所有变量和导入模块的字典。
        return {
            'wmi': 'wmi' in globals(),
            'pythoncom': 'pythoncom' in globals(),
            'pyDes': 'des' in globals(),
            'all_required': HAS_CRYPTO_LIBS
        }

    @staticmethod
    def install_instructions() -> str:
        """返回缺失依赖的安装说明"""
        if HAS_CRYPTO_LIBS:
            return "所有依赖均已满足"

        instructions = [
            "请安装以下依赖库：",
            "pip install WMI",
            "pip install pyDes"
        ]
        return "\n".join(instructions)

    @staticmethod
    def get_hardware_info() -> Dict[str, List]:
        """获取硬件信息（CPU、硬盘、网卡、主板）"""
        if not HAS_CRYPTO_LIBS:
            raise RuntimeError("缺少必要的依赖库，请先安装: " + LicenseManager.install_instructions())
        # 初始化当前线程的COM库，用于支持后续的COM对象调用，常在使用Windows COM组件前调用。
        pythoncom.CoInitialize()

        try:
            wmi_obj = wmi.WMI()

            # CPU信息
            cpu_info = []
            for processor in wmi_obj.Win32_Processor():
                cpu_info.append({
                    "Name": processor.Name,
                    "Serial Number": processor.ProcessorId,  # 序列号
                    "CoreNum": processor.NumberOfCores
                })

            # 硬盘信息
            disk_info = []
            physical_media = wmi_obj.Win32_PhysicalMedia()
            for drive in wmi_obj.Win32_DiskDrive():
                serial = physical_media[0].SerialNumber.strip().strip('.') if physical_media else ""
                disk_info.append({
                    "Serial": serial,
                    "ID": drive.deviceid,
                    "Caption": drive.Caption,
                    "size": str(int(float(drive.Size) / 1024 / 1024 / 1024)) if drive.Size else "0"
                })

            # 网络信息
            network_info = []
            for adapter in wmi_obj.Win32_NetworkAdapterConfiguration():
                if adapter.MacAddress:
                    network_info.append({
                        "MAC": adapter.MacAddress,
                        "ip": adapter.IPAddress
                    })

            # 主板信息
            mainboard_info = []
            for board in wmi_obj.Win32_BaseBoard():
                mainboard_info.append(board.SerialNumber.strip().strip('.'))

            return {
                "cpu": cpu_info,
                "disk": disk_info,
                "network": network_info,
                "mainboard": mainboard_info
            }

        finally:
            pythoncom.CoUninitialize()

    @staticmethod
    def generate_machine_code() -> str:
        """生成机器码"""
        hw_info = LicenseManager.get_hardware_info()

        mac = hw_info["network"][0]["MAC"] if hw_info["network"] else ""
        cpu_sn = hw_info["cpu"][0]["Serial Number"] if hw_info["cpu"] else ""
        disk_sn = hw_info["disk"][0]["Serial"] if hw_info["disk"] else ""
        board_sn = hw_info["mainboard"][0] if hw_info["mainboard"] else ""

        machine_str = f"{mac}{cpu_sn}{disk_sn}{board_sn}"
        select_index = [3, 6, 15, 16, 17, 30, 32, 38, 43, 46, 54, 55]

        return "".join(machine_str[i] for i in select_index if i < len(machine_str))

    @staticmethod
    def decrypt_register_code(encrypted_str: str) -> str:
        """解密注册码"""
        if not encrypted_str or not HAS_CRYPTO_LIBS:
            return "0000"

        try:
            decoded = base64.b32decode(encrypted_str)
            key = '3ef25a8n'
            iv = '477bdb68'
            cipher = des(key, CBC, iv, pad=None, padmode=PAD_PKCS5)
            decrypted = cipher.decrypt(decoded)
            return str(decrypted).replace("b'", '').replace("'", '')
        except Exception:
            return "0000"

    @staticmethod
    def verify_license_file(machine_code: str, license_path: str = "./license.lc") -> int:
        """
        验证许可证文件
        返回值：
        0 - 验证通过
        1 - 注册码无效
        2 - 注册码过期
        3 - 许可证文件不存在
        """
        if not os.path.exists(license_path):
            return 3

        try:
            with open(license_path, 'r', encoding='utf-8') as f:
                license_data = json.load(f)

            if 'register_code' not in license_data:
                return 1

            decrypted = LicenseManager.decrypt_register_code(license_data['register_code'])
            if decrypted == "0000":
                return 1

            expire_time = decrypted.replace(machine_code, '').strip()
            if not expire_time:
                return 1

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if expire_time < current_time:
                return 2

            return 0

        except Exception:
            return 1

    @staticmethod
    def save_license(register_code: str, machine_code: str, license_path: str = "./license.lc") -> bool:
        """保存许可证到文件"""
        try:
            decrypted = LicenseManager.decrypt_register_code(register_code)
            if machine_code not in decrypted:
                return False

            expire_time = decrypted.replace(machine_code, '').strip()
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            license_data = {
                "generate_time": current_time,
                "expire_time": expire_time,
                "machine_code": machine_code,
                "register_code": register_code,
            }

            with open(license_path, 'w', encoding='utf-8') as f:
                json.dump(license_data, f, ensure_ascii=False, indent=2)

            return True

        except Exception:
            return False


# 简单脚本使用
if __name__ == "__main__":
    # 检查依赖
    print(LicenseManager.install_instructions())
    # 检查依赖
    if not LicenseManager.check_dependencies()["all_required"]:
        exit(1)
    # 获取所有的硬件信息
    print(LicenseManager.get_hardware_info())
    # 生成机器码
    machine_code = LicenseManager.generate_machine_code()
    print(f"机器码: {machine_code}")
    # 验证许可证
    result = LicenseManager.verify_license_file(machine_code)
    print(f"验证结果: {result}")
