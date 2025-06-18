import numpy as np
from numba import cuda
import numpy as np
import math
import matplotlib.pyplot as plt
def parse_trajectory(file_path):
    with open(file_path, 'r') as file:
        frame_data = []
        for line in file:
            if line.startswith('ITEM: TIMESTEP'):
                if frame_data:  # 如果已经有数据，先处理上一帧
                    yield np.array(frame_data)
                    frame_data = []  # 重置为下一帧
            elif line.startswith('ITEM: ATOMS'):
                pass  # 这里是ATOMS行，下一行开始是原子数据
            else:
                parts = line.split()
                if len(parts) == 8:  # 确保这是原子数据行
                    # 仅获取原子的x, y, z坐标
                    coords = list(map(float, parts[2:5]))
                    # print(coords)
                    frame_data.append(coords)
        if frame_data:  # 确保最后一帧也被处理
            yield np.array(frame_data)


# CUDA设备函数计算向量的模长
@cuda.jit(device=True)
def magnitude(vector):
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)


# CUDA设备函数计算叉积
@cuda.jit(device=True)
def cross(v1, v2):
    return (v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0])


# CUDA设备函数计算点积
@cuda.jit(device=True)
def dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


@cuda.jit(device=True)
def sub(v1, v2):
    # 手动实现向量减法
    return (v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2])


@cuda.jit(device=True)
def add(v1, v2):
    # 手动实现向量加法
    return (v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2])


@cuda.jit(device=True)
def divide(v, divisor):
    # 手动实现向量除法
    return (v[0] / divisor, v[1] / divisor, v[2] / divisor)


# CUDA kernel计算所有phi
@cuda.jit
def calculate_phi_kernel(beads, phis, T):
    i = cuda.grid(1)  # 获取全局索引
    if i < (beads.shape[0] - int(1.25 * T)):
        A = beads[i]
        B = beads[int(i + 0.5 * T)]
        C = beads[int(i + 0.75 * T)]
        D = beads[int(i + 1.25 * T)]

        E = divide(add(A, B), 2)
        F = divide(add(C, D), 2)

        AB = sub(B, A)
        CD = sub(D, C)
        EF = sub(F, E)

        cp = cross(CD, AB)
        dp = dot(EF, cp)

        # 避免除以零
        magnitudes = magnitude(EF) * magnitude(CD) * magnitude(AB)

        if magnitudes > 0:
            phis[i] = dp / magnitudes
        else:
            phis[i] = 0.0

if __name__ == '__main__':
    # 假设你有一个 NumPy 数组 beads_positions 包含所有 beads 的位置
    # 你需要把它传递到 GPU
    # filepath = '../Output/trajectory_ems_0.dump'
    filepath = './Frame_2.5/frame_40000.txt'
    # genfun = parse_trajectory(filepath)

    # for item in genfun:
        # beads_positions = np.array(list(parse_trajectory(filepath)))[1000]
    beads_positions = np.loadtxt(filepath)[:, 2:5]
    beads_positions = np.ascontiguousarray(beads_positions)
    # print(beads_positions.shape)
    beads_positions_device = cuda.to_device(beads_positions)

    T=125
    # 创建一个空的 NumPy 数组来存储结果
    phis = np.zeros(beads_positions.shape[0] - int(1.25*T), dtype=np.float32)

    phis_device = cuda.to_device(phis)

    threadsperblock = 32
    blockspergrid = (phis.size + (threadsperblock - 1)) // threadsperblock
    calculate_phi_kernel[blockspergrid, threadsperblock](beads_positions_device, phis_device, T)
    phis = phis_device.copy_to_host()

    plt.plot(phis)
    plt.xlabel("$bead Id$")
    plt.ylabel("$chirality$")
    plt.show()
    print(np.mean(phis))