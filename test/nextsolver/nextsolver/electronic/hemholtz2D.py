# -*- coding: utf-8 -*-
"""
Created on 2025/9/2

@author: Yifei Sun
"""

import torch
import pyrfm
import json
import math

json_str = """
{
  "material": {
    "relative_permittivity": 2.2,
    "relative_permeability": 1.0,
    "conductivity": 0.0
  },
  "source": {
    "type": "discrete_port",
    "position": 0.10,
    "amplitude": 1.0,
    "phase": 0.0,
    "unit": "V"
  },
  "boundary_conditions": {
    "left":  { "type": "dirichlet", "value": 0.0, "unit": "V"   },
    "right": { "type": "neumann",   "value": 0.0, "unit": "V/m" }
  },
  "frequency": 1.0e9
}
"""

"""
电磁二维 Helmholtz 问题（与所给 JSON 精确对应）


一、问题描述（Problem Statement）
- 物理场：时间谐波 TEM 近似下一维标量场 u(x) ∈ ℂ（与“V”单位对齐，可理解为电压包络）。
- 空间域：x ∈ (0, L)，其中 L > x₀ = 0.10 m（几何长度不在本设定 JSON 中固定）。
- 频率：f = 1.0 GHz，角频率 ω = 2πf。
- 材料：均匀各向同性无耗介质，ε = εᵣ ε₀，μ = μᵣ μ₀，σ = 0。
- 激励：在 x = x₀ 处的离散端口（点源），幅值 A = 1.0，相位 φ = 0（单位“V”）。
- 边界：左端 Dirichlet（u(0) = 0 V，等效短路）；右端 Neumann（u′(L) = 0 V/m，等效开路）。

二、常量与派生波参数（由材料与频率给出）
- 真空常数：ε₀ = 8.854 187 817×10⁻¹² F/m，μ₀ = 4π×10⁻⁷ H/m。
- 相对参数：εᵣ = 2.2，μᵣ = 1.0，σ = 0（无耗）。
- 角频率：ω = 2π × 10⁹ ≈ 6.283 185 307×10⁹ rad/s。
- 相速度：v = 1/√(με) = c₀/√(εᵣ μᵣ) ≈ 2.021 200 339×10⁸ m/s。
- 波长：λ = v/f ≈ 0.202 120 034 m。
- 波数：k = 2π/λ ≈ 31.086 405 362 rad/m。
- 介质本征阻抗：η = √(μ/ε) = η₀/√εᵣ ≈ 253.991 526 Ω（η₀ ≈ 376.730 314 Ω）。
  说明：η 为介质的平面波/TEM 本征阻抗，非几何定义的传输线特性阻抗 Z₀。

三、治理方程（Governing PDE，频域，e^{-iωt} 约定）
在 (0, L) 内，u(x) 满足一维 Helmholtz 方程含点源：
    u″(x) + k² u(x) = − Q δ(x − x₀),     0 < x < L,
其中 δ 为狄拉克 δ，Q ∈ ℂ 为点源强度（频域等效值）。与 JSON 中的
“amplitude = 1.0、phase = 0” 对应，可写
    Q = A·e^{iφ}·,
A = 1.0，φ = 0

四 点源跳跃条件（Jump Condition at Point Source）
由于点源的存在，u(x) 在 x = x₀ 处满足跳跃条件：
    u′(x₀⁺) − u′(x₀⁻) = − Q,
    u(x₀⁺) = u(x₀⁻),
其中 x₀⁺、x₀⁻ 分别表示 x₀ 处的右极限与左极限。


# 四、点源的高斯函数极限
# 为数值计算方便，点源可近似为高斯函数极限形式：
#     δ(x − x₀) = lim (σ → 0) 1/(σ√(2π)) exp(−(x − x₀)²/(2σ²)) 。
# 在实际计算中，σ 取一个小的正数（如网格尺寸的十分之一）即可.
# 并且保证采样点均值为 1 / I, 其中 I 为区间长度.
# 即 ∫ δ(x − x₀) dx = 1.

五、边界条件（与 JSON 一致）
- 左端 Dirichlet： u(0) = 0 V      （短路、强制电压为零）
- 右端 Neumann：  u′(L) = 0 V/m    （开路、零法向导数/零通量）


"""


def func_Dirac(x, x0, sigma, L):
    """
    高斯函数极限形式的狄拉克 δ 函数近似
    δ(x − x₀) = lim (σ → 0) 1/(σ√(2π)) exp(−(x − x₀)²/(2σ²))
    :param x: 输入张量
    :param x0: 点源位置
    :param sigma: 标准差，控制宽度（数值上取一个小的正数）
    :param I: 区间长度，用于归一化
    :return: 近似的 δ(x - x0)
    """
    coeff = 1.0 / (sigma * torch.sqrt(torch.tensor(2.0 * torch.pi)))  # 归一化系数
    r = (x - x0).norm(dim=1, p=2, keepdim=True)
    gauss = torch.exp(-0.5 * (r / sigma) ** 2) * coeff  # 高斯函数
    gauss /= gauss.mean()
    return gauss / L  # 保证积分为 1


def func_green_dirichlet(x, x0, k, L):
    x0 = torch.ones_like(x) * x0
    f_left = torch.sin(k * x) * torch.sin(k * (L - x0))
    f_right = torch.sin(k * x0) * torch.sin(k * (L - x))
    return torch.where(x < x0, f_left, f_right) / (k * math.sin(k * L))


if __name__ == "__main__":
    domain = pyrfm.Circle2D(center=(0, 0), radius=1)
    x_in = domain.in_sample(num_samples=10000, with_boundary=False)
    x0 = torch.tensor([[0.2, 0.0]])
    x_in = torch.cat([x_in, x0], dim=0)  # 确保采样点中包含点源位置
    x_on = domain.on_sample(num_samples=400)

    f_in = -func_Dirac(x_in, x0, 1e-5, 2 * torch.pi)
    k = 80

    model = pyrfm.RFMBase(dim=2, n_hidden=400, domain=domain, n_subdomains=1)

    A_in = model.features(x_in).cat(dim=1)
    A_in_xx = model.features_second_derivative(x_in, axis1=0, axis2=0).cat(dim=1)
    A_in_yy = model.features_second_derivative(x_in, axis1=1, axis2=1).cat(dim=1)
    A_on = model.features(x_on).cat(dim=1)

    A = pyrfm.concat_blocks([[A_in_xx + A_in_yy + k ** 2 * A_in], [A_on]])
    b = torch.cat([f_in, torch.zeros((x_on.shape[0], 1), device=A.device)], dim=0)

    model.compute(A, damp=1e-8).solve(b)

    viz = pyrfm.RFMVisualizer2D(model=model)

    viz.plot()
    viz.show()
