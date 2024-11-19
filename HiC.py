import numpy as np
from numba import cuda
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

@cuda.jit(device=True)
def f(mu, rc, r):
    return 0.5 * (1 + math.tanh(mu * (rc - r)))

@cuda.jit
def calculate_distance_matrix(coords, distance_matrix, mu, rc):
    i, j = cuda.grid(2)
    if i < coords.shape[0] and j < coords.shape[0] and i != j:
        dx = coords[i, 0] - coords[j, 0]
        dy = coords[i, 1] - coords[j, 1]
        dz = coords[i, 2] - coords[j, 2]
        dist = (dx * dx + dy * dy + dz * dz) ** 0.5
        distance_matrix[i, j] = f(mu, rc, dist)


mu = 1.0
rc = 4.0
if __name__ == '__main__':
    filepath = r'./trajectory_kappa5.0_5.0.dump'
    genfun = parse_trajectory(filepath)

    i = 0
    distance_matrix_sum = np.zeros((1500, 1500), dtype=np.float32)
    t = 0
    for coords in genfun:
        i += 1
        if(i>38000 and i < 40000):
            t+=1
            N = coords.shape[0]
            distance_matrix = np.zeros((N, N), dtype=np.float32)
            # 将数据从主机复制到设备
            d_coords = cuda.to_device(coords)
            d_distance_matrix = cuda.to_device(distance_matrix)
            # 定义 CUDA 网格大小
            threads_per_block = (16, 16)
            blocks_per_grid_x = int(np.ceil(coords.shape[0] / threads_per_block[0]))
            blocks_per_grid_y = int(np.ceil(coords.shape[0] / threads_per_block[1]))
            blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
            # 调用 CUDA 内核
            # rc = 5.0
            calculate_distance_matrix[blocks_per_grid, threads_per_block](d_coords, d_distance_matrix, mu, rc)

            # 将结果从设备复制到主机
            d_distance_matrix.copy_to_host(distance_matrix)

            distance_matrix_sum += distance_matrix

    distance_matrix_sum = distance_matrix_sum/t
    np.savetxt("contact_matrix_0.0_5.0.txt", distance_matrix_sum)
    plt.imshow(distance_matrix_sum)
    plt.colorbar()
    plt.show()




