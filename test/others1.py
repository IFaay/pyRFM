import time
import torch
import pyrfm


# 精确解定义
def u(x):
    return 0.5 + (10 / 3) * x - (x ** 2) / 2 + (x ** 3) / 3 - (10 / 12) * x ** 4


# 设置默认设备
torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义问题域
domain = pyrfm.Line1D(0, 1)

# 创建 RFM 模型
model = pyrfm.RFMBase(dim=1, n_hidden=100, domain=domain, n_subdomains=4)

# 内部点采样（不包含边界）
x_in = domain.in_sample(6000, with_boundary=False)

# 构建 PDE 方程部分：u'' + 1 - 2x + 10x^2 = 0
u_in = model.features(x_in).cat(dim=1)
uxx = model.features_second_derivative(x_in, axis1=0, axis2=0).cat(dim=1)

A_in = uxx
f_in = -(+ 1.0 - 2 * x_in + 10 * x_in ** 2)

# 采样边界点
n = 1000
x_left = domain.on_sample(n)[:n // 2]
x_right = domain.on_sample(n)[-n // 2 + 1:]

# Dirichlet 边界条件：u(0) = 1/2
u_left = model.features(x_left).cat(dim=1)

# Neumann 边界条件：u'(1) = 0
u_x_right = model.features_derivative(x_right, axis=0).cat(dim=1)

# 构建系统矩阵 A 和向量 b
A = pyrfm.concat_blocks([
    [A_in],
    [u_left],
    [u_x_right]
])

b = pyrfm.concat_blocks([
    [f_in],
    [torch.full((u_left.shape[0], 1), 0.5)],
    [torch.zeros(u_x_right.shape[0], 1)]
])

# 求解
model.compute(A).solve(b)

# 测试与误差计算
x_test = domain.in_sample(100, with_boundary=True)
u_pre = model(x_test)
u_true = u(x_test)

# 打印误差
print('Error:', ((u_pre - u_true).norm() / u_true.norm()).item())

# 绘图
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(x_test.cpu().numpy(), u_pre.cpu().numpy(), label='Predicted', color='blue')
plt.plot(x_test.cpu().numpy(), u_true.cpu().numpy(), label='Exact', color='orange', linestyle='--')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Comparison of Predicted and Exact Solutions')
plt.legend()
plt.grid()
plt.show()
