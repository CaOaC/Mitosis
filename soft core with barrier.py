import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# 定义Lennard-Jones势能函数
def U_LJ(r, epsilon, sigma):
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6 + 1/4)

# 通过解方程 U_LJ(r_c) = E_cut 来找到 r_c（在 r_c 处 LJ 势能达到 E_cut）
def find_r_c(E_cut, epsilon, sigma):
    def equation(r):
        return U_LJ(r, epsilon, sigma) - E_cut
    r_c_initial_guess = sigma  # 从sigma开始猜测
    r_c_solution = fsolve(equation, r_c_initial_guess)
    return r_c_solution[0]

# 通过解方程 U_LJ(r_0) = E_cut/2 来找到 r_0
def find_r0(E_cut, epsilon, sigma):
    def equation(r):
        return U_LJ(r, epsilon, sigma) - E_cut / 2
    r0_initial_guess = sigma  # 从sigma开始猜测
    r0_solution = fsolve(equation, r0_initial_guess)
    return r0_solution[0]

# 定义软核势能函数
def U_sc(r, r0, E_cut, epsilon, sigma):
    if r < r0:
        U_LJ_r = U_LJ(r, epsilon, sigma)
        return 0.5 * E_cut * (1 + np.tanh(2 * U_LJ_r / E_cut - 1))
    elif r0 <= r <= sigma * 2**(1/6):
        return U_LJ(r, epsilon, sigma)
    else:
        return 0

# 定义新的势能函数，r = a 处势能突变到 E_a
def U_new(r, r0, r_c, E_cut, E_a, epsilon, sigma, a, b):
    if r == b:
        return E_a  # 在r = b处势能突变到E_a
    elif a > r > b:
        return (E_a - E_cut) * r / (b - a) + (E_cut * b - E_a * a) / (b - a)
    elif r > a:
        return U_sc(r, r0, E_cut, epsilon, sigma)  # r > a 时使用原来的软核势
    else:
        return E_a  # r < b 时仍然使用原来的软核势

# 参数设定
epsilon = 1.0
sigma = 1.0
E_cut = 10
E_a = 100.0  # 突变后的势能值 E_a
a = 0.8  # 在 r = a 处势能突变
b = 0.4  # 在 r = b 处势能到达平台
r0 = find_r0(E_cut, epsilon, sigma)  # 找到 r_0
r_c = find_r_c(E_cut, epsilon, sigma)  # 找到 r_c

print(f"Calculated r_0: {r0}")
print(f"Calculated r_c: {r_c}")

# 生成r值范围
r_values = np.linspace(0.1, 2.5, 500)
U_values = [U_new(r, r0, r_c, E_cut, E_a, epsilon, sigma, a, b) for r in r_values]

# 绘制势能曲线
plt.figure(dpi=300)  # 设置图片 DPI
plt.plot(r_values, U_values, label=f'Soft-core Potential with higher barrier', color='#BD514A')
plt.axvline(x=sigma * 2**(1/6), color='#B4B4B6', linestyle='--', label='$\sigma 2^{1/6}$')
plt.axvline(x=r0, color='#488B87', linestyle='--', label='$r_0$')
plt.axvline(x=a, color='#8FC2C7', linestyle='--', label=f'$r = r_a$')
plt.axvline(x=b, color='#C7E2E4', linestyle='--', label=f'$r = r_b$')

# 添加图像标题和标签
plt.title('Modified Soft-core Potential with higher barrier', fontsize=12)
plt.xlabel('Distance $r_{i,j}$', fontsize=12)
plt.ylabel('Potential $U_{SC}(r_{i,j})$', fontsize=12)

# 设置坐标轴刻度字号
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# 显示图例和网格
plt.legend(fontsize=8)
plt.grid(True, linestyle='--', alpha=0.3)

# 显示绘图
plt.show()