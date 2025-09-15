# -*- coding: utf-8 -*-
"""
Created on 2025/2/26

@author: Yifei Sun
"""
import sympy as sp

# 定义变量
m1, m2, m3, phi1, phi2, phi3 = sp.symbols('m1 m2 m3 phi1 phi2 phi3')
mxx1, mxx2, mxx3 = sp.symbols('mxx1 mxx2 mxx3')
m = sp.Matrix([m1, m2, m3])
mxx = sp.Matrix([mxx1, mxx2, mxx3])
phi = sp.Matrix([phi1, phi2, phi3])

print(m.cross(phi.cross(mxx)))

# # 定义u和v (去掉小数部分，使用分数)
# u = sp.Rational(1, 10) * y * ((x + 10) * sp.sin(y) + (y + 5) * sp.cos(x))
# v = sp.Rational(1, 60) * y * ((30 + 5 * x * sp.sin(5 * x)) * (4 + sp.exp(-5 * y)) - 100)
#
# u_x = sp.diff(u, x)  # u 关于x的一阶导数
# u_y = sp.diff(u, y)  # u 关于y的一阶导数
# v_x = sp.diff(v, x)  # v 关于x的一阶导数
# v_y = sp.diff(v, y)  # v 关于y的一阶导数
#
# # 计算二阶导数
# u_xx = sp.diff(u, x, 2)  # u 关于x的二阶导数
# u_xy = sp.diff(u, x, y)  # u 关于x和y的混合二阶导数
# u_yy = sp.diff(u, y, 2)  # u 关于y的二阶导数
#
# v_xx = sp.diff(v, x, 2)  # v 关于x的二阶导数
# v_xy = sp.diff(v, x, y)  # v 关于x和y的混合二阶导数
# v_yy = sp.diff(v, y, 2)  # v 关于y的二阶导数
#
# # 输出结果
# print("u_x = ", u_x)
# print("u_y = ", u_y)
# print("v_x = ", v_x)
# print("v_y = ", v_y)
#
# print("u_xx = ", u_xx)
# print("u_xy = ", u_xy)
# print("u_yy = ", u_yy)
# print("v_xx = ", v_xx)
# print("v_xy = ", v_xy)
# print("v_yy = ", v_yy)
