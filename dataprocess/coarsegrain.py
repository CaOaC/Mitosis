import numpy as np

def load_structure(filename):
    """加载结构文件，返回x, y, z坐标的数组"""
    data = np.loadtxt(filename)
    x = data[:, 2]  # 第二列为x坐标
    y = data[:, 3]  # 第三列为y坐标
    z = data[:, 4]  # 第四列为z坐标
    return x, y, z

def coarse_grain(x, y, z, coarse_level):
    """
    根据粗粒化程度进行粗粒化
    :param x: x坐标数组
    :param y: y坐标数组
    :param z: z坐标数组
    :param coarse_level: 每多少个粒子组成一个粗粒粒子
    :return: 粗粒化后的x, y, z坐标
    """
    n_particles = len(x)
    n_coarse_particles = n_particles // coarse_level  # 粗粒粒子的数量

    cg_x = np.zeros(n_coarse_particles)
    cg_y = np.zeros(n_coarse_particles)
    cg_z = np.zeros(n_coarse_particles)

    for i in range(n_coarse_particles):
        start = i * coarse_level
        end = start + coarse_level
        cg_x[i] = np.mean(x[start:end])
        cg_y[i] = np.mean(y[start:end])
        cg_z[i] = np.mean(z[start:end])

    return cg_x, cg_y, cg_z

def save_coarse_grained_structure_xyz(filename, cg_x, cg_y, cg_z):
    """以XYZ格式保存粗粒化后的结构"""
    n_particles = len(cg_x)
    with open(filename, 'w') as f:
        f.write(f"{n_particles}\n")
        f.write("Coarse-grained structure\n")  # 注释行
        for i in range(n_particles):
            f.write(f"C {cg_x[i]:.6f} {cg_y[i]:.6f} {cg_z[i]:.6f}\n")  # 假设元素为 "C"（碳）

def main():
    #input_filename = 'structure.txt'  # 输入的结构文件名
    input_filename = 'traj_2.5_2.5.txt'  # 输入的结构文件名
    output_filename = 'coarse_grained_structure.xyz'  # 输出的粗粒化结构文件名
    coarse_level = 150  # 粗粒化的程度，比如每10个粒子合成一个粗粒粒子

    # 加载原始结构
    x, y, z = load_structure(input_filename)

    # 进行粗粒化
    cg_x, cg_y, cg_z = coarse_grain(x, y, z, coarse_level)

    # 保存粗粒化后的结构为标准的.xyz文件
    save_coarse_grained_structure_xyz(output_filename, cg_x, cg_y, cg_z)
    print(f"粗粒化结构已保存到 {output_filename}")

if __name__ == '__main__':
    main()
