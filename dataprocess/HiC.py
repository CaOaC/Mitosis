import numpy as np
from numba import cuda
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def parse_trajectory(file_path):
    with open(file_path, 'r') as file:
        frame_data = []
        for line in file:
            if line.startswith('ITEM: TIMESTEP'):
                if frame_data:
                    yield np.array(frame_data)
                    frame_data = []
            elif line.startswith('ITEM: ATOMS'):
                pass
            else:
                parts = line.split()
                if len(parts) == 8:
                    coords = list(map(float, parts[2:5]))
                    frame_data.append(coords)
        if frame_data:
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

mu = 2.27
rc = 2.91
if __name__ == '__main__':
    filepath = r'trajectory_kappa0.5_0.5_1.dump.dump'
    genfun = parse_trajectory(filepath)

    i = 0
    distance_matrix_sum = np.zeros((1500, 1500), dtype=np.float32)
    t = 0
    for coords in genfun:
        i += 1
        if 0 < i < 20000:
            t += 1
            N = coords.shape[0]
            distance_matrix = np.zeros((N, N), dtype=np.float32)
            d_coords = cuda.to_device(coords)
            d_distance_matrix = cuda.to_device(distance_matrix)
            threads_per_block = (16, 16)
            blocks_per_grid_x = int(np.ceil(coords.shape[0] / threads_per_block[0]))
            blocks_per_grid_y = int(np.ceil(coords.shape[0] / threads_per_block[1]))
            blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
            calculate_distance_matrix[blocks_per_grid, threads_per_block](d_coords, d_distance_matrix, mu, rc)
            d_distance_matrix.copy_to_host(distance_matrix)
            distance_matrix_sum += distance_matrix

    distance_matrix_sum = distance_matrix_sum / t
    np.savetxt("contact_matrix_0.0_5.0.txt", distance_matrix_sum)

    # 绘制热图
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    im = ax.imshow(distance_matrix_sum, cmap='hot_r', norm=LogNorm(vmin=1e-4, vmax=1))

    # 设置颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Contact Probability", fontsize=14)
    cbar.set_ticks([1e-3, 1e-2, 1e-1, 1])  # 手动设置刻度
    cbar.ax.set_yticklabels([r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$1$"])  # 设置刻度标签

    # 设置坐标轴标题和字号
    ax.set_xlabel("Particle Index", fontsize=14)
    ax.set_ylabel("Particle Index", fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # 设置坐标轴范围
    ax.set_xlim([0, 1500])  # 设置x轴范围
    ax.set_ylim([0, 1500])  # 设置y轴范围

    # 显示图像
    plt.tight_layout()
    plt.savefig("contact_matrix_visualization.png")
    plt.show()