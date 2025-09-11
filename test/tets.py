# -*- coding: utf-8 -*-
"""
Created on 2025/8/14

@author: Yifei Sun
"""

import sympy as sp

x, y = sp.symbols('x y', real=True)

u = x + x ** 2 - 2 * x * y + x ** 3 - 3 * x * y ** 2 + x ** 2 * y
v = -y - 2 * x * y + y ** 2 - 3 * x ** 2 * y + y ** 3 - x * y ** 2
p = x * y + x + y + x ** 3 * y ** 2 - sp.Rational(4, 3)

# 固定压力常数：p(0,0) = -4/3，减去它（不影响∇p与f）
p_fixed = p - p.subs({x: 0, y: 0})

grad_p = sp.Matrix([sp.diff(p_fixed, x), sp.diff(p_fixed, y)])
lap_u = sp.diff(u, x, 2) + sp.diff(u, y, 2)
lap_v = sp.diff(v, x, 2) + sp.diff(v, y, 2)

print(sp.Matrix([-lap_u, -lap_v]))
fvec = sp.simplify(sp.Matrix([-lap_u, -lap_v]) + grad_p)
div_u = sp.simplify(sp.diff(u, x) + sp.diff(v, y))

print("f(x,y) =", fvec)  # Matrix([[3*x**2*y**2 - y - 1], [2*x**3*y + 3*x - 1]])
print("div u =", div_u)  # 0
