import numpy as np
from numba import cuda
import matplotlib.pyplot as plt

@cuda.jit(device=True)
def dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]

@cuda.jit(device=True)
def norm(u):
    l = (u[0]*u[0] + u[1]*u[1] + u[2]*u[2])**0.5
    return (u[0]/l, u[1]/l, u[2]/l)

@cuda.jit(device=True)
def sub(v1, v2):
    # 手动实现向量减法
    return (v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2])

@cuda.jit
def orientation_kernel(beads, d, orien):
    id = cuda.grid(1)  # 获取全局索引
    if id < (beads.shape[0] - (d+4)):
        A = beads[int(id)]
        B = beads[int(id + 4)]
        AB = sub(B,A)
        AB = norm(AB)

        C = beads[int(id+d)]
        D = beads[int(id+d+4)]
        CD = sub(D,C)
        CD = norm(CD)

        orien[id] = dot(AB, CD)

if __name__ == '__main__':
    path = '../Output/traj.txt'
    traj = np.loadtxt(path)[:,2:5]

    beads_positions = np.ascontiguousarray(traj)
    beads_positions_device = cuda.to_device(beads_positions)

    len = 570
    _oriens = []
    for d in range(1,len,1):
        # 创建一个空的 NumPy 数组来存储结果
        oriens = np.zeros(beads_positions.shape[0] - int(d+4), dtype=np.float32)
        oriens_device = cuda.to_device(oriens)

        threadsperblock = 8
        blockspergrid = (oriens.size + (threadsperblock - 1)) // threadsperblock
        orientation_kernel[blockspergrid, threadsperblock](beads_positions_device, d, oriens_device)
        oriens = oriens_device.copy_to_host()
        _oriens.append(np.mean(oriens))

    fft_oriens = np.fft.fft(_oriens)
    freqs = np.fft.fftfreq(fft_oriens.size, 1)
    plt.plot(freqs[:freqs.size // 2], np.abs(fft_oriens)[:fft_oriens.size // 2])
    plt.xscale('log')
    plt.show()