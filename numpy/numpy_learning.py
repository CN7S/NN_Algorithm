import numpy as np

# 创建数组
arr1 = np.array([1, 2, 3, 4, 5])                    # 从列表
arr2 = np.zeros(10)                                 # 全零数组
arr3 = np.ones((3, 4))                              # 全一数组
arr4 = np.full((2, 3), 7)                           # 填充值
arr5 = np.arange(0, 10, 2)                          # 类似range
arr6 = np.linspace(0, 1, 5)                         # 线性间隔
arr7 = np.random.rand(3, 3)                         # 随机数组
arr8 = np.eye(4)                                    # 单位矩阵

print("数组形状:", arr3.shape)
print("数组维度:", arr3.ndim)
print("数组大小:", arr3.size)
print("数据类型:", arr3.dtype)

# 一维数组索引
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(arr[2])           # 单个元素
print(arr[2:5])         # 切片
print(arr[::2])         # 步长
print(arr[::-1])        # 反转

# 二维数组索引
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(matrix[1, 2])     # 第二行第三列
print(matrix[0:2, 1:3]) # 子矩阵
print(matrix[:, 1])     # 第二列
print(matrix[1, :])     # 第二行

# 布尔索引
bool_idx = arr > 5
print(arr[bool_idx])    # 大于5的元素

# 花式索引
print(arr[[1, 3, 5]])   # 选择特定索引


arr = np.arange(12)

# 重塑形状
reshaped = arr.reshape(3, 4)
print("重塑后:\n", reshaped)

# 展平
flattened = reshaped.flatten()
print("展平:", flattened)

# 转置
transposed = reshaped.T
print("转置:\n", transposed)

# 增加维度
expanded = np.expand_dims(arr, axis=0)
print("增加维度后形状:", expanded.shape)

# 拼接数组
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
concat1 = np.concatenate([a, b], axis=0)  # 垂直拼接
concat2 = np.concatenate([a, b], axis=1)  # 水平拼接


a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 基本运算
print("加法:", a + b)
print("减法:", a - b)
print("乘法:", a * b)      # 逐元素乘法
print("除法:", a / b)
print("幂运算:", a ** 2)

# 矩阵运算
matrix_a = np.array([[1, 2], [3, 4]])
matrix_b = np.array([[5, 6], [7, 8]])
print("矩阵乘法:\n", np.dot(matrix_a, matrix_b))
print("矩阵乘法:\n", matrix_a @ matrix_b)

# 通用函数
print("平方根:", np.sqrt(a))
print("指数:", np.exp(a))
print("对数:", np.log(a))
print("三角函数:", np.sin(a))

# 聚合函数
arr = np.array([1, 2, 3, 4, 5])
print("求和:", np.sum(arr))
print("均值:", np.mean(arr))
print("标准差:", np.std(arr))
print("最大值:", np.max(arr))
print("最小值:", np.min(arr))

# 标量与数组
arr = np.array([1, 2, 3])
print("标量广播:", arr + 5)

# 不同形状数组
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
vector = np.array([10, 20, 30])
print("向量广播:\n", matrix + vector)

# 三维广播
arr_3d = np.ones((2, 3, 4))
arr_1d = np.array([1, 2, 3, 4])
print("三维广播形状:", (arr_3d + arr_1d).shape)

# 创建矩阵
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 基本线性代数
print("矩阵乘法:\n", A @ B)
print("点积:\n", np.dot(A, B))
print("转置:\n", A.T)
print("逆矩阵:\n", np.linalg.inv(A))
print("行列式:", np.linalg.det(A))
print("特征值:", np.linalg.eigvals(A))

# 解线性方程组
# Ax = b
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])
x = np.linalg.solve(A, b)
print("方程解:", x)

# 各种分布
uniform = np.random.rand(5)          # [0,1)均匀分布
normal = np.random.randn(5)          # 标准正态分布
integers = np.random.randint(0, 10, 5) # 整数随机数

print("均匀分布:", uniform)
print("正态分布:", normal)
print("随机整数:", integers)

# 设置随机种子
np.random.seed(42)
reproducible = np.random.rand(3)
print("可重现随机数:", reproducible)

# 随机排列
arr = np.arange(10)
shuffled = np.random.permutation(arr)
print("随机排列:", shuffled)

# 随机选择
choices = np.random.choice(arr, size=5, replace=False)
print("随机选择:", choices)

# 创建示例数据
data = np.random.rand(5, 3)

# 保存到文件
np.savetxt('data.txt', data, delimiter=',', fmt='%.3f')
np.save('data.npy', data)

# 从文件加载
loaded_txt = np.loadtxt('data.txt', delimiter=',')
loaded_npy = np.load('data.npy')

print("从文本文件加载:\n", loaded_txt)
print("从npy文件加载:\n", loaded_npy)

# CSV文件操作
np.savetxt('data.csv', data, delimiter=',', header='col1,col2,col3')
csv_data = np.genfromtxt('data.csv', delimiter=',', skip_header=1)


# 条件操作
arr = np.array([1, 2, 3, 4, 5])
result = np.where(arr > 3, arr, 0)  # 大于3的保留，其他为0
print("条件操作:", result)

# 向量化函数
def custom_func(x):
    return x**2 + 2*x + 1

vectorized_func = np.vectorize(custom_func)
print("向量化函数:", vectorized_func(arr))

# 网格坐标
x = np.linspace(-2, 2, 5)
y = np.linspace(-2, 2, 5)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2
print("网格坐标Z:\n", Z)

# 排序和搜索
arr = np.array([3, 1, 4, 1, 5, 9, 2])
sorted_arr = np.sort(arr)
argsorted = np.argsort(arr)
print("排序:", sorted_arr)
print("排序索引:", argsorted)

# 唯一值
unique_vals = np.unique(arr)
print("唯一值:", unique_vals)

import time

# 向量化 vs 循环
def slow_sum(arr):
    """慢速的Python循环"""
    result = 0
    for i in range(len(arr)):
        result += arr[i]
    return result

def fast_sum(arr):
    """快速的NumPy向量化"""
    return np.sum(arr)

# 性能比较
large_arr = np.random.rand(1000000)

start = time.time()
result1 = slow_sum(large_arr)
time1 = time.time() - start

start = time.time()
result2 = fast_sum(large_arr)
time2 = time.time() - start

print(f"Python循环: {time1:.4f}秒")
print(f"NumPy向量化: {time2:.4f}秒")
print(f"加速比: {time1/time2:.1f}x")

# 内存视图
arr = np.arange(10)
view = arr[::2]  # 创建视图，不复制数据
view[0] = 100    # 修改视图会影响原数组
print("原数组:", arr)  # 第一个元素也被修改

# 模式1: 创建特定模式的数组
def create_patterns():
    """创建常见模式数组"""
    # 对角线矩阵
    diag = np.diag([1, 2, 3])
    
    # 重复模式
    repeated = np.tile([1, 2, 3], 4)
    
    # 网格
    grid = np.indices((3, 3))
    
    return diag, repeated, grid

# 模式2: 数据预处理
def data_preprocessing(data):
    """数据预处理管道"""
    # 标准化
    normalized = (data - np.mean(data)) / np.std(data)
    
    # 归一化到[0,1]
    scaled = (data - np.min(data)) / (np.max(data) - np.min(data))
    
    # 处理缺失值
    data_with_nan = np.array([1, 2, np.nan, 4, 5])
    cleaned = np.nan_to_num(data_with_nan, nan=0.0)
    
    return normalized, scaled, cleaned

# 模式3: 批量操作
def batch_operations():
    """批量数据处理"""
    # 批量矩阵乘法
    batch_A = np.random.rand(10, 3, 4)
    batch_B = np.random.rand(10, 4, 5)
    batch_result = np.matmul(batch_A, batch_B)
    
    # 沿特定轴操作
    data = np.random.rand(100, 5)
    column_means = np.mean(data, axis=0)
    row_sums = np.sum(data, axis=1)
    
    return batch_result, column_means, row_sums