# 打开原始文件和目标文件
with open('Arabidopsis_all.txt', 'r') as f_input, open('Arabidopsis_all.txt', 'w') as f_output:
    # 逐行读取原始文件
    for line in f_input:
        # 按空格分割每一行
        columns = line.strip().split()
        # 交换第一列和第二列
        updated_line = columns[1] + ' ' + columns[0] + ' ' + ' '.join(columns[2:]) + '\n'
        # 将更新后的行写入目标文件
        f_output.write(updated_line)