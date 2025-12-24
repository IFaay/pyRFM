import sympy as sp
from sympy.printing.pycode import pycode
from sympy.printing.mathematica import mathematica_code
import re
import json


def generate_laplace_beltrami_rhs_for_surface(u_expr: sp.Expr,
                                              phi_expr: sp.Expr,
                                              return_expr=False) -> str:
    """
    生成 PyTorch 格式的 Laplace–Beltrami 算子右端函数 func_rhs(p)，
    或返回 SymPy 表达式（return_expr=True）

    φ(x,y,z) = 0 为隐式曲面
    """
    x, y, z = sp.symbols('x y z')

    # 梯度和 Hessian
    grad_u = sp.Matrix([sp.diff(u_expr, var) for var in (x, y, z)])
    Hess_u = sp.Matrix([[sp.diff(u_expr, vi, vj) for vj in (x, y, z)]
                        for vi in (x, y, z)])
    lap_u = sum([sp.diff(u_expr, var, var) for var in (x, y, z)])

    # φ 的梯度、单位法向量等
    grad_phi = sp.Matrix([sp.diff(phi_expr, var) for var in (x, y, z)])
    norm_grad_phi = sp.sqrt(sum([g ** 2 for g in grad_phi]))
    n = grad_phi / norm_grad_phi

    partial_n_u = (n.T * grad_u)[0]
    n_H_n = (n.T * Hess_u * n)[0]
    div_n = sum([sp.diff(n[i], var) for i, var in enumerate((x, y, z))])
    mean_curvature = div_n / 2

    # Laplace–Beltrami 表达式 (Δ_Γ u)
    lap_beltrami = lap_u - 2 * mean_curvature * partial_n_u - n_H_n
    lap_beltrami_exact = sp.simplify(sp.together(lap_beltrami, deep=True))

    if return_expr:
        # 返回 SymPy 表达式，供 XML / 其他格式使用
        return lap_beltrami_exact

    # ============== 原来的 PyTorch 代码生成部分 ==============
    rhs_code = pycode(lap_beltrami_exact)
    rhs_code = rhs_code.replace('math.', 'torch.')

    # 正则替换变量
    rhs_code = re.sub(r'\bx\b', 'p[:, 0]', rhs_code)
    rhs_code = re.sub(r'\by\b', 'p[:, 1]', rhs_code)
    rhs_code = re.sub(r'\bz\b', 'p[:, 2]', rhs_code)

    func_str = f"""
def func_rhs(p: torch.Tensor) -> torch.Tensor:
    \"\"\"
    Laplace–Beltrami of u(x,y,z) on given implicit surface
    \"\"\"
    return ({rhs_code}).unsqueeze(-1)
"""

    # 生成 mean curvature 的 PyTorch 表达式字符串
    mean_curv_code = pycode(mean_curvature)
    mean_curv_code = mean_curv_code.replace('math.', 'torch.')
    mean_curv_code = re.sub(r'\bx\b', 'p[:, 0]', mean_curv_code)
    mean_curv_code = re.sub(r'\by\b', 'p[:, 1]', mean_curv_code)
    mean_curv_code = re.sub(r'\bz\b', 'p[:, 2]', mean_curv_code)

    func_mc_str = f"""
def func_mean_curvature(p: torch.Tensor) -> torch.Tensor:
    \"\"\"
    Mean curvature H(x,y,z) of given implicit surface
    \"\"\"
    return ({mean_curv_code}).unsqueeze(-1)
"""

    return func_str + "\n" + func_mc_str


# ========= 已有：SymPy Expr 转 <FunctionExpr> XML 字符串 =========

def sympy_expr_to_functionexpr(expr: sp.Expr,
                               func_id: int,
                               dim: int = 3,
                               indent: int = 4) -> str:
    """
    将 SymPy 表达式 expr 转为

    <Function type="FunctionExpr" id="func_id" dim="dim">
        ...
    </Function>

    的字符串形式，使用 x,y,z, sin, cos, exp, pi, ^ 等。
    """
    # 用 sstr 得到比较“干净”的字符串（无 sympy. 前缀）
    s = sp.sstr(expr)

    # 把 ** 改为 ^，符合你示例中的风格
    s = s.replace('**', '^')

    indent_str = " " * indent
    return (
        f'<Function type="FunctionExpr" id="{func_id}" dim="{dim}">\n'
        f'{indent_str}{s}\n'
        f'</Function>'
    )


# ========= 新增：SymPy Expr → JSON 中的 "phi","u","f","alpha","nu" =========

def sympy_expr_to_math_style(expr: sp.Expr) -> str:
    """
    将 SymPy 表达式转成类似 Mathematica 的字符串形式，
    如：Sin[Pi*x]*Sin[Pi*y*z]，(x^2 + y^2 + z^2)^(1/2) 等。
    """
    s = mathematica_code(expr)
    return s


def make_json_block(phi_expr: sp.Expr,
                    u_expr: sp.Expr,
                    f_expr: sp.Expr,
                    alpha_sym=None,  # <<< 新增
                    nu_sym=None  # <<< 新增
                    ) -> str:
    """
    生成一段 JSON 字符串：

    {
        "phi":   "...",
        "u":     "...",
        "f":     "...",
        "alpha": "...",
        "nu":    "..."
    }
    """
    phi_str = sympy_expr_to_math_style(phi_expr)
    u_str = sympy_expr_to_math_style(u_expr)
    f_str = sympy_expr_to_math_style(f_expr)

    data = {
        "phi": phi_str,
        "u": u_str,
        "f": f_str,
    }

    # # 把 alpha, nu 也写进去（如果给了）
    # if alpha_sym is not None:  # <<<
    #     data["alpha"] = sympy_expr_to_math_style(alpha_sym)
    # if nu_sym is not None:  # <<<
    #     data["nu"] = sympy_expr_to_math_style(nu_sym)

    # 交给 json.dumps 处理转义，保证可以直接粘到文件里
    return json.dumps(data, ensure_ascii=False, indent=4)


def main():
    x, y, z = sp.symbols('x y z')
    alpha, nu = sp.symbols('alpha nu')  # <<< PDE 参数符号

    # 这里用的是 -sin(pi*x)*exp(cos(y - z))
    u_expr = -sp.sin(sp.pi * x) * sp.exp(sp.cos(y - z))

    # 使用 Rational 明确分数
    phi_list = [
        ("sphere", x ** 2 + y ** 2 + z ** 2 - 2),
        ("ellipsoid",
         (x / sp.Rational(3, 2)) ** 2 + y ** 2 + (z / sp.Rational(1, 2)) ** 2 - 1),
        ("torus",
         (sp.sqrt(x ** 2 + y ** 2) - 1) ** 2 + z ** 2 - sp.Rational(1, 16)),
        ("genus-2-torus",
         ((x + 1) * x ** 2 * (x - 1) + y ** 2) ** 2 + z ** 2 - sp.Rational(1, 100)),
        ("cheese",
         ((4 * x ** 2 - 1) ** 2 + (4 * y ** 2 - 1) ** 2 + (4 * z ** 2 - 1) ** 2
          + 16 * (x ** 2 + y ** 2 - 1) ** 2
          + 16 * (x ** 2 + z ** 2 - 1) ** 2
          + 16 * (y ** 2 + z ** 2 - 1) ** 2 - 16)),
    ]

    for name, phi_expr in phi_list:
        print(f"# === Surface: {name} ===")

        # 1) 得到 Laplace–Beltrami 的 SymPy 表达式 Δ_Γ u
        lap_expr = generate_laplace_beltrami_rhs_for_surface(
            u_expr, phi_expr, return_expr=True
        )

        # 这里构造 PDE 右端：
        #   -nu * Δ_Γ u + alpha * u = f
        # 即 f = alpha * u - nu * Δ_Γ u
        f_expr = alpha * u_expr - nu * lap_expr  # <<< 关键一步

        # 2) 转为 FunctionExpr 的 XML 字符串（id 按你需要调）
        rhs_func_xml = sympy_expr_to_functionexpr(f_expr, func_id=1, dim=3)  # <<< 用 f_expr
        u_func_xml = sympy_expr_to_functionexpr(u_expr, func_id=3, dim=3)

        print("## XML FunctionExpr (rhs)")
        print(rhs_func_xml)
        print()
        print("## XML FunctionExpr (u)")
        print(u_func_xml)
        print()

        # 3) 输出 JSON 格式，包含 phi, u, f, alpha, nu
        print("## JSON (phi, u, f, alpha, nu)")
        json_block = make_json_block(
            phi_expr, u_expr, f_expr,
            alpha_sym=alpha, nu_sym=nu  # <<<
        )
        print(json_block)
        print("\n\n")


if __name__ == "__main__":
    main()
