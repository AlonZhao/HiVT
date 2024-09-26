import os

def check_file_pairs(directory):
    # 获取目录下的所有文件
    files = os.listdir(directory)
    
    # 提取文件名中的日期和 A/B 标识
    file_dates_A = set()
    file_dates_B = set()
    for file in files:
        if '_lines' in file:
            date = file.split('_lines')[0]
            file_dates_A.add(date)
        elif '_obs' in file:
            date = file.split('_obs')[0]
            file_dates_B.add(date)
    
    # 检查哪些 A 文件或 B 文件没有对应的配对
    missing_A = file_dates_B - file_dates_A
    missing_B = file_dates_A - file_dates_B
    
    # 打印没有配对的结果
    if not missing_A and not missing_B:
        print("所有文件都成对。")
    else:
        if missing_A:
            print(f"缺少对应的 A 文件：{missing_A}")
        if missing_B:
            print(f"缺少对应的 B 文件：{missing_B}")

# 调用函数，指定你的文件目录
check_file_pairs("/home/alon/Learning/HiVT/data_root/train/data")
