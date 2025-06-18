import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置字体为SimHei（黑体），如果你系统安装了其他字体，请使用其他中文字体名
rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

# 导入数据
def load_xyz_file(filename):
    """从 .xyz 文件中加载粒子坐标"""
    with open(filename, 'r') as f:
        nparticles = int(f.readline().strip())  # 第一行粒子数
        f.readline()  # 第二行注释
        #data = np.loadtxt(f, usecols=(2, 3, 4))  # 读取 x, y, z 列
        data = np.loadtxt(f, usecols=(1, 2, 3))  # 读取 x, y, z 列
        return data, nparticles


# 计算相邻粒子之间的单位向量
def calculate_unit_vectors(positions):
    """
    计算相邻粒子之间的单位向量
    :param positions: 粒子的坐标数组 (N, 3)，表示 N 个粒子的三维坐标
    :return: 相邻粒子之间的单位向量 (N-1, 3)
    """
    vectors = positions[1:] - positions[:-1]  # 计算相邻粒子之间的向量
    norms = np.linalg.norm(vectors, axis=1)[:, np.newaxis]  # 计算每个向量的模长
    unit_vectors = vectors / norms  # 归一化得到单位向量
    return unit_vectors


# 计算角度关联函数
def calculate_cos_theta(unit_vectors, s_prime):
    """
    计算单位向量之间相隔 s' 的点积的平均值，代表角度关联函数
    :param unit_vectors: 相邻粒子的单位向量数组 (N-1, 3)
    :param s_prime: 相隔的距离 s'
    :return: 角度关联函数的平均值
    """
    n_vectors = len(unit_vectors)
    if s_prime >= n_vectors:
        raise ValueError("s' 不能大于链的长度")

    cos_theta_values = []
    for i in range(n_vectors - s_prime):
        cos_theta = np.dot(unit_vectors[i], unit_vectors[i + s_prime])
        cos_theta_values.append(cos_theta)

    return np.mean(cos_theta_values)


def plot_cos_theta_vs_sprime(unit_vectors):
    """计算并绘制所有 s' 的角度关联函数"""
    # max_s_prime = len(unit_vectors) - 1  # 最大 s' 为粒子数 - 2
    max_s_prime = 8  # 最大 s' 为粒子数 - 2
    s_prime_values = range(1, max_s_prime + 1)
    cos_theta_values = []

    for s_prime in s_prime_values:
        avg_cos_theta = calculate_cos_theta(unit_vectors, s_prime)
        cos_theta_values.append(avg_cos_theta)

    # 绘图
    plt.figure(figsize=(8, 6))
    plt.plot(s_prime_values, cos_theta_values, marker='o', linestyle='-', color='b')
    plt.xlabel("s'", fontsize=14)
    plt.ylabel(r'$\langle \cos(\theta(s\')) \rangle$', fontsize=14)
    plt.title(r'角度关联函数 $\langle \cos(\theta(s\')) \rangle$ 随 s\' 变化', fontsize=16)
    plt.grid(True)
    plt.show()


def main():
    input_filename = 'coarse_grained_structure.xyz'
    # input_filename = 'structure.xyz'

    # 加载粒子坐标
    positions, nparticles = load_xyz_file(input_filename)
    print(f"粒子数: {nparticles}")

    # 计算相邻粒子之间的单位向量
    unit_vectors = calculate_unit_vectors(positions)

    # 绘制所有 s' 的角度关联函数图
    plot_cos_theta_vs_sprime(unit_vectors)


if __name__ == '__main__':
    main()
