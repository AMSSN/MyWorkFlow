str_list = ["7-掉墨", "8-掉墨", "9-溢墨", "10-端面不良-线状", "11-掉墨", "13-墨渣，端面", "14-脏污", "16-伤痕",
            "20-溢墨", "21-台阶裂边", "24-端面", "38-裂边", "44-溢墨", "45-掉墨，脱膜", "46-端面", "47-掉墨", "50-墨渣",
            "52-端面", "61-墨渣", "74-掉墨", "84-伤痕", "86-溢墨", "97-伤痕", "100-端面不良-线状"]
result = []
for item in str_list:
    global_id_str, defect = item.split('-', 1)
    global_id = int(global_id_str)
    tray_no = (global_id - 1) // 25 + 1
    item_no = (global_id - 1) % 25 + 1
    result.append(f"{tray_no},{item_no}-{defect}-{global_id_str}")

# 输出结果
print("\n".join(result))
