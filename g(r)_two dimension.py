import numpy as np
import matplotlib.pyplot as plt

def read_data(filename):
    # 读取粒子数据
    data = np.loadtxt(filename)
    x = data[:, 1]
    y = data[:, 2]
    z = data[:, 3]
    return x, y, z

def cartesian_to_cylindrical(x, y, z):
    # 将笛卡尔坐标转换为圆柱坐标
    r_perp = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    r_parallel = z
    return r_perp, theta, r_parallel

def compute_gr(distances, bin_width, max_distance):
    # 计算径向分布函数g(r)
    bins = np.arange(0, max_distance + bin_width, bin_width)
    hist, edges = np.histogram(distances, bins=bins)
    # 计算归一化因子
    rho = len(distances) / (np.pi * max_distance**2)
    shell_volume = np.pi * (edges[1:]**2 - edges[:-1]**2)
    number_density = hist / shell_volume
    gr = number_density / rho
    # 在r=0处设置g(r)=0
    gr[edges[:-1] == 0] = 0
    return edges[:-1], gr

def compute_gz(distances, bin_width, max_distance):
    # 计算沿z方向的g(r_parallel)
    bins = np.arange(0, max_distance + bin_width, bin_width)
    hist, edges = np.histogram(distances, bins=bins)
    # 计算归一化因子
    rho = len(distances) / (2 * max_distance)
    shell_length = edges[1:] - edges[:-1]
    number_density = hist / shell_length
    gr = number_density / rho
    # 在r=0处设置g(r)=0
    gr[edges[:-1] == 0] = 0
    return edges[:-1], gr

def calculate_gr_perp(x, y):
    # 计算所有粒子对的r_perp距离
    positions = np.column_stack((x, y))
    distances = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            delta_r = positions[i] - positions[j]
            r_perp = np.linalg.norm(delta_r)
            distances.append(r_perp)
    return np.array(distances)

def calculate_gr_parallel(z):
    # 计算所有粒子对的r_parallel距离
    distances = []
    for i in range(len(z)):
        for j in range(i + 1, len(z)):
            delta_z = abs(z[i] - z[j])
            distances.append(delta_z)
    return np.array(distances)

# 读取第一个数据集
x1, y1, z1 = read_data('structure_combined.txt')
r_perp1, theta1, r_parallel1 = cartesian_to_cylindrical(x1, y1, z1)

# 读取第二个数据集
x2, y2, z2 = read_data('structure_combined_1.txt')
r_perp2, theta2, r_parallel2 = cartesian_to_cylindrical(x2, y2, z2)

# 计算r_perp和r_parallel的距离分布
distances_r_perp1 = calculate_gr_perp(x1, y1)
distances_r_parallel1 = calculate_gr_parallel(z1)
distances_r_perp2 = calculate_gr_perp(x2, y2)
distances_r_parallel2 = calculate_gr_parallel(z2)

# 设置参数
bin_width_perp = 0.05
max_distance_perp = max(np.max(distances_r_perp1), np.max(distances_r_perp2))
bin_width_parallel = 0.05
max_distance_parallel = max(np.max(distances_r_parallel1), np.max(distances_r_parallel2))

# 计算g(r_perp)
r_perp_bins1, gr_perp1 = compute_gr(distances_r_perp1, bin_width_perp, max_distance_perp)
r_perp_bins2, gr_perp2 = compute_gr(distances_r_perp2, bin_width_perp, max_distance_perp)

# 计算g(r_parallel)
r_parallel_bins1, gr_parallel1 = compute_gz(distances_r_parallel1, bin_width_parallel, max_distance_parallel)
r_parallel_bins2, gr_parallel2 = compute_gz(distances_r_parallel2, bin_width_parallel, max_distance_parallel)

# 在r=0处设置g(r)=0
gr_perp1[0] = 0
gr_perp2[0] = 0
gr_parallel1[0] = 0
gr_parallel2[0] = 0

# 设置字体大小
font_size = 14
plt.rcParams.update({'font.size': font_size})

# 设置曲线透明度
alpha_value = 0.5  # 您可以修改此值来改变曲线的透明度（0到1之间）

# 绘制g(r_perp)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(r_perp_bins1, gr_perp1, label='Simulation structure', alpha=1)
plt.plot(r_perp_bins2, gr_perp2, label='Fitting structure', alpha=alpha_value)
plt.xlabel('$r_{\\perp}$', fontsize=font_size)
plt.ylabel('$g(r_{\\perp})$', fontsize=font_size)
plt.title('$g(r_{\\perp})$ Comparison', fontsize=font_size + 2)
plt.legend(fontsize=font_size)
plt.tick_params(axis='both', which='major', labelsize=font_size)

# 绘制g(r_parallel)
plt.subplot(1, 2, 2)
plt.plot(r_parallel_bins1, gr_parallel1, label='Simulation structure', alpha=1)
plt.plot(r_parallel_bins2, gr_parallel2, label='Fitting structure', alpha=alpha_value)
plt.xlabel('$r_{\\parallel}$', fontsize=font_size)
plt.ylabel('$g(r_{\\parallel})$', fontsize=font_size)
plt.title('$g(r_{\\parallel})$ Comparison', fontsize=font_size + 2)
plt.legend(fontsize=font_size)
plt.tick_params(axis='both', which='major', labelsize=font_size)

plt.tight_layout()
plt.show()
