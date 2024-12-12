import numpy as np

# 加载.npy文件
data = np.load('ZEA_data.npy')

# 查看数据的形状
print("数据形状：", data.shape)

# 查看数据的类型
print("数据类型：", data.dtype)

# 查看数据内容
print("数据内容：", data)

# 对数据进行操作，比如获取某一行或某一列
# 获取第一行数据
first_row = data[0, :]
print("第一行数据：", first_row)

# 获取第一列数据
first_column = data[:, 0]
print("第一列数据：", first_column)
