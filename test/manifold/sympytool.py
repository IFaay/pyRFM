import sympy as sp
from sympy.printing.pycode import pycode
import re


def generate_laplace_beltrami_rhs_for_surface(u_expr: sp.Expr,
                                              phi_expr: sp.Expr,
                                              return_expr=False) -> str:
    """
    生成 PyTorch 格式的 Laplace–Beltrami 算子右端函数 func_rhs(p)，
    保留精确分数，支持任意给定的隐式曲面 φ(x,y,z) = 0
    """
    x, y, z = sp.symbols('x y z')

    # 梯度和 Hessian
    grad_u = sp.Matrix([sp.diff(u_expr, var) for var in (x, y, z)])
    Hess_u = sp.Matrix([[sp.diff(u_expr, vi, vj) for vj in (x, y, z)] for vi in (x, y, z)])
    lap_u = sum([sp.diff(u_expr, var, var) for var in (x, y, z)])

    # 法向量和曲率项
    grad_phi = sp.Matrix([sp.diff(phi_expr, var) for var in (x, y, z)])
    norm_grad_phi = sp.sqrt(sum([g ** 2 for g in grad_phi]))
    n = grad_phi / norm_grad_phi

    partial_n_u = (n.T * grad_u)[0]
    n_H_n = (n.T * Hess_u * n)[0]
    div_n = sum([sp.diff(n[i], var) for i, var in enumerate((x, y, z))])
    mean_curvature = div_n / 2

    # Laplace–Beltrami 表达式
    lap_beltrami = lap_u - 2 * mean_curvature * partial_n_u - n_H_n
    lap_beltrami_exact = sp.simplify(sp.together(lap_beltrami, deep=True))

    if return_expr:
        return lap_beltrami_exact

    # 转为 PyTorch 表达式字符串
    rhs_code = pycode(lap_beltrami_exact)
    rhs_code = rhs_code.replace('math.', 'torch.')

    # 正则替换变量
    rhs_code = re.sub(r'\bx\b', 'p[:, 0]', rhs_code)
    rhs_code = re.sub(r'\by\b', 'p[:, 1]', rhs_code)
    rhs_code = re.sub(r'\bz\b', 'p[:, 2]', rhs_code)

    # 拼接 PyTorch 函数
    func_str = f"""
def func_rhs(p: torch.Tensor) -> torch.Tensor:
    \"\"\"
    Laplace–Beltrami of u(x,y,z) on given implicit surface
    \"\"\"
    return ({rhs_code}).unsqueeze(-1)
"""

    return func_str


def main():
    x, y, z = sp.symbols('x y z')
    u_expr = sp.sin(x) * sp.exp(sp.cos(y - z))

    # 使用 Rational 明确分数
    phi_list = [
        ("ellipsoid", (x / sp.Rational(3, 2)) ** 2 + y ** 2 + (z / sp.Rational(1, 2)) ** 2 - 1),
        ("torus", (sp.sqrt(x ** 2 + y ** 2) - 1) ** 2 + z ** 2 - sp.Rational(1, 16)),
        ("genus-2-torus", ((x + 1) * x ** 2 * (x - 1) + y ** 2) ** 2 + z ** 2 - sp.Rational(1, 100)),
        ("cheese", ((4 * x ** 2 - 1) ** 2 + (4 * y ** 2 - 1) ** 2 + (4 * z ** 2 - 1) ** 2
                    + 16 * (x ** 2 + y ** 2 - 1) ** 2
                    + 16 * (x ** 2 + z ** 2 - 1) ** 2
                    + 16 * (y ** 2 + z ** 2 - 1) ** 2 - 16)),
    ]

    for name, phi_expr in phi_list:
        print(f"# === Surface: {name} ===")
        func_code = generate_laplace_beltrami_rhs_for_surface(u_expr, phi_expr)
        print(func_code)
        print("\n")


if __name__ == "__main__":
    main()
