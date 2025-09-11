from sympy import symbols, Eq, solve, sqrt, simplify, Rational

# 定义符号变量
z = symbols('z', real=True)

# 使用 Rational 保持符号精度
x_val = Rational(3, 4)
y_val = Rational(3, 4)

# 表达式构造
psi = (4 * x_val ** 2 - 1) ** 2 + (4 * y_val ** 2 - 1) ** 2 + (4 * z ** 2 - 1) ** 2 \
      + 16 * (x_val ** 2 + y_val ** 2 - 1) ** 2 \
      + 16 * (x_val ** 2 + z ** 2 - 1) ** 2 \
      + 16 * (y_val ** 2 + z ** 2 - 1) ** 2 - 16

psi = simplify(psi)
print("简化后的 ψ(z):")
print(psi)

# 解 ψ(z) = 0
solutions = solve(Eq(psi, 0), z)

print("\n精确表达式形式的实数解 z：")
for s in solutions:
    if s.is_real:
        print(s)  # 输出符号表达式
        print(s.evalf())  # 输出数值形式（可选）
