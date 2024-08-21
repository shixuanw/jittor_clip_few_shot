# 读取文件并处理
with open('features.txt', 'r') as file:
    lines = file.readlines()

# 提取类别名称并写入新文件
with open('feature.txt', 'w') as file:
    i=0
    for line in lines:
        line = line.strip(':') + f" {i}\n"  # 去除行末尾的换行符并添加新的内容
        file.write(line)  # 写入新文件
        i += 1