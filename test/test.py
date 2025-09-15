# -*- coding: utf-8 -*-
"""
Created on 2025/4/18

@author: Yifei Sun
"""
import sympy as sp

# 定义符号
x = sp.Symbol('x')

# 定义精确解 u(x)
u = 0.5 + (10 / 3) * x - (1 / 2) * x ** 2 + (1 / 3) * x ** 3 - (10 / 12) * x ** 4

# 计算 u''(x)
u_xx = sp.diff(u, x, 2)

# 构造左边项：u'' + 1 - 2x + 10x^2
lhs = u_xx + 1 - 2 * x + 10 * x ** 2

# 化简表达式
simplified = sp.simplify(lhs)

# 打印结果
print("u''(x) + 1 - 2x + 10x^2 =", simplified)

# 检查是否恒等于0
if simplified == 0:
    print("✅ 这个真解满足微分方程。")
else:
    print("❌ 这个真解不满足微分方程。")

# 验证边界条件
u_prime = sp.diff(u, x)

u_0 = u.subs(x, 0)
u_prime_1 = u_prime.subs(x, 1)

print("u(0) =", u_0)
print("u'(1) =", u_prime_1)

if u_0 == 0.5:
    print("✅ 满足 u(0) = 1/2")
else:
    print("❌ 不满足 u(0) = 1/2")

if abs(u_prime_1.evalf()) < 1e-6:  # 使用 evalf() 来处理浮点数比较
    print("✅ 满足 u'(1) = 0")
else:
    print("❌ 不满足 u'(1) = 0")

import numpy as np
import matplotlib.pyplot as plt

# 使用 numpy 生成 x 取值
x_vals = np.linspace(0, 1, 200)

# 将 sympy 表达式转换为 numpy 函数
u_func = sp.lambdify(x, u, modules=['numpy'])

# 计算对应的 u(x) 值
y_vals = u_func(x_vals)

# 绘图
plt.figure(figsize=(8, 5))
plt.plot(x_vals, y_vals, label="Exact solution u(x)", linewidth=2)
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("Plot of the Exact Solution")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
