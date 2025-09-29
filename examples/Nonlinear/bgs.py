import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 参数设置
epsilon = 0.1
dx = 5e-2  # 空间步长
dt = 5e-3  # 时间步长
x = np.arange(-1, 1 + dx, dx)  # 空间网格
t = np.arange(0, 1 + dt, dt)  # 时间网格
Nx = len(x)
Nt = len(t)

# 初始条件
u = -np.sin(np.pi * x)


# 边界条件函数（这里假设为常数边界条件）
def Leftbdry(t):
    return 0


def Rightbdry(t):
    return 0


# 初始化解矩阵
U = np.zeros((Nx, Nt))
U[:, 0] = u  # 初始条件

# 向后欧拉迭代求解
for n in range(Nt - 1):
    u_n = U[:, n]  # 当前时间步的解
    u_np1 = u_n  # 下一时间步的解，初始猜测为当前时间步的解

    # 迭代求解隐式方程
    max_iter = 100
    tol = 1e-6
    for iter in range(max_iter):
        # 计算 u_x 和 u_xx（这里使用简单的二阶中心差分，注意边界处理）
        u_x = np.zeros(Nx)
        u_xx = np.zeros(Nx)

        # 中心差分
        u_x[1:-1] = (u_np1[2:] - u_np1[:-2]) / (2 * dx)
        u_xx[1:-1] = (u_np1[2:] - 2 * u_np1[1:-1] + u_np1[:-2]) / (dx ** 2)

        # 边界处理
        u_x[0] = (u_np1[1] - u_np1[0]) / dx
        u_x[-1] = (u_np1[-1] - u_np1[-2]) / dx

        u_xx[0] = (u_np1[1] - 2 * u_np1[0] + Leftbdry(t[n + 1])) / (dx ** 2)
        u_xx[-1] = (Rightbdry(t[n + 1]) - 2 * u_np1[-1] + u_np1[-2]) / (dx ** 2)

        # 计算 u * u_x
        uu_x = u_np1 * u_x

        # 向后欧拉方程
        u_np1_new = u_n - dt * (uu_x - epsilon * u_xx)

        # 检查收敛性
        if np.linalg.norm(u_np1_new - u_np1, np.inf) < tol:
            break
        u_np1 = u_np1_new

    # 更新解矩阵
    U[:, n + 1] = u_np1

    # 边界条件修正（如果需要）
    U[0, n + 1] = Leftbdry(t[n + 1])
    U[-1, n + 1] = Rightbdry(t[n + 1])

# anmiation U by time
# 绘制数值解的动画
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot(x, U[:, 0], color='blue')
ax.set_ylim(-1.2, 1.2)
time_text = ax.text(0.0, 0.95, '', transform=ax.transAxes)
ax.set_title("Burgers' Equation Solution (Live Update)")


# 动画更新函数
def update(frame):
    line.set_ydata(U[:, frame])
    time_text.set_text(f"Time = {frame * dt:.6f}")
    return line, time_text


# 创建动画
ani = FuncAnimation(fig, update, frames=Nt, interval=100, blit=True)
plt.show()

# # 使用 meshgrid 生成网格数据
# T, X = np.meshgrid(t, x)
#
# # 绘图显示数值解
# fig = plt.figure(figsize=(12, 12))
#
# # 第一个视角
# ax1 = fig.add_subplot(221, projection='3d')
# ax1.plot_surface(T, X, U, cmap='viridis', edgecolor='none')
# ax1.set_title('View from angle 1')
# ax1.set_xlabel('t')
# ax1.set_ylabel('x')
# ax1.set_zlabel('u(x,t)')
# ax1.view_init(elev=30, azim=45)
#
# # 第二个视角
# ax2 = fig.add_subplot(222, projection='3d')
# ax2.plot_surface(T, X, U, cmap='viridis', edgecolor='none')
# ax2.set_title('View from angle 2')
# ax2.set_xlabel('t')
# ax2.set_ylabel('x')
# ax2.set_zlabel('u(x,t)')
# ax2.view_init(elev=30, azim=135)
#
# # 第三个视角
# ax3 = fig.add_subplot(223, projection='3d')
# ax3.plot_surface(T, X, U, cmap='viridis', edgecolor='none')
# ax3.set_title('View from angle 3')
# ax3.set_xlabel('t')
# ax3.set_ylabel('x')
# ax3.set_zlabel('u(x,t)')
# ax3.view_init(elev=60, azim=45)
#
# # 第四个视角
# ax4 = fig.add_subplot(224, projection='3d')
# ax4.plot_surface(T, X, U, cmap='viridis', edgecolor='none')
# ax4.set_title('View from angle 4')
# ax4.set_xlabel('t')
# ax4.set_ylabel('x')
# ax4.set_zlabel('u(x,t)')
# ax4.view_init(elev=60, azim=135)
#
# plt.suptitle('Burgers Equation Solution from Multiple Angles')
# plt.show()
