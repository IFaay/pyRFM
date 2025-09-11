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
电磁一维 Helmholtz 问题（与所给 JSON 精确对应）


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
    torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')
    input_points = """
0.00000000
0.00010064
0.00040256
0.00090565
0.00160988
0.00251521
0.00362162
0.00492905
0.00643747
0.00814682
0.01005708
0.01216819
0.01448009
0.01699273
0.01970605
0.02261999
0.02573449
0.02904947
0.03256486
0.03628057
0.04019652
0.04431262
0.04862878
0.05314490
0.05786088
0.06277660
0.06789197
0.07320685
0.07872113
0.08443467
0.09034734
0.09645900
0.10276949
0.10927867
0.11598638
0.12289246
0.12999675
0.13729906
0.14479924
0.15249710
0.16039245
0.16848510
0.17677487
0.18526154
0.19394491
0.20282477
0.21190089
0.22117306
0.23064106
0.24030466
0.25016362
0.26021771
0.27046668
0.28091028
0.29154825
0.30238033
0.31340626
0.32462577
0.33603857
0.34764438
0.35944291
0.37143386
0.38361694
0.39599184
0.40855825
0.42131584
0.43426429
0.44740326
0.46073241
0.47425140
0.48795987
0.50185748
0.51594385
0.53021863
0.54468144
0.55933191
0.57416966
0.58919429
0.60440541
0.61980262
0.63538551
0.65115369
0.66710672
0.68324420
0.69956568
0.71607075
0.73275895
0.74962984
0.76668295
0.78391784
0.80133402
0.81893102
0.83670836
0.85466557
0.87280214
0.89111758
0.90961138
0.92828303
0.94713201
0.96615779
0.98535984
1.00000000
    """

    points_list = input_points.strip().split("\n")
    # 转换为浮点数
    points_float = [float(x) for x in points_list]
    # 转换为 PyTorch 张量
    points = torch.tensor(points_float).reshape(-1, 1)

    print(points)

    data = json.loads(json_str)
    print(data)
    x_min = points[:, 0].min().item()
    x_max = points[:, 0].max().item()
    print(f"x_min: {x_min}, x_max: {x_max}")

    # x0 = 0.5
    #
    # domain1 = pyrfm.Line1D(x_min, x0)
    # domain2 = pyrfm.Line1D(x0, x_max)
    # model1 = pyrfm.RFMBase(dim=1, n_hidden=200, domain=domain1, n_subdomains=1)
    # model2 = pyrfm.RFMBase(dim=1, n_hidden=200, domain=domain2, n_subdomains=1)
    # x_in_1 = domain1.in_sample(num_samples=5000, with_boundary=False)
    # x_in_2 = domain2.in_sample(num_samples=5000, with_boundary=False)
    # x_on_1 = torch.tensor([[x_min]])
    # x_on_2 = torch.tensor([[x_max]])
    # x_c = torch.tensor([[x0]])
    #
    # A_in_1 = model1.features(x_in_1).cat(dim=1)
    # A_in_2 = model2.features(x_in_2).cat(dim=1)
    # A_in_xx_1 = model1.features_second_derivative(x_in_1, axis1=0, axis2=0).cat(dim=1)
    # A_in_xx_2 = model2.features_second_derivative(x_in_2, axis1=0, axis2=0).cat(dim=1)
    # A_on_1 = model1.features(x_on_1).cat(dim=1)
    # A_on_2 = model2.features(x_on_2).cat(dim=1)
    # A_c_1 = model1.features(x_c).cat(dim=1)
    # A_c_2 = model2.features(x_c).cat(dim=1)
    # A_c_x_1 = model1.features_derivative(x_c, axis=0).cat(dim=1)
    # A_c_x_2 = model2.features_derivative(x_c, axis=0).cat(dim=1)
    #
    # k = 31.086405362
    # A = pyrfm.concat_blocks([[A_in_xx_1 + k ** 2 * A_in_1, torch.zeros_like(A_in_1)],
    #                          [torch.zeros_like(A_in_2), A_in_xx_2 + k ** 2 * A_in_2],
    #                          [A_on_1, torch.zeros_like(A_on_1)],
    #                          [torch.zeros_like(A_on_2), A_on_2],
    #                          [A_c_1, -A_c_2],
    #                          [A_c_x_1, -A_c_x_2]])
    # b = torch.cat([torch.zeros((x_in_1.shape[0], 1), device=A.device),
    #                torch.zeros((x_in_2.shape[0], 1), device=A.device),
    #                torch.zeros((x_on_1.shape[0], 1), device=A.device),
    #                torch.zeros((x_on_2.shape[0], 1), device=A.device),
    #                torch.zeros((x_c.shape[0], 1), device=A.device),
    #                torch.tensor([[1.0]], device=A.device)], dim=0)
    #
    # A_normed = A.norm(dim=1, p=2, keepdim=True)
    # A = A / A_normed
    # b = b / A_normed
    # W = torch.linalg.lstsq(A, b)[0]
    # model1.W = W[:model1.n_hidden, :]
    # model2.W = W[model1.n_hidden:, :]
    #
    # x_in = torch.cat([x_in_1, x_in_2], dim=0)
    # f_in = torch.cat([model1(x_in_1), model2(x_in_2)], dim=0)
    #
    # import matplotlib.pyplot as plt
    #
    # plt.plot(x_in.cpu(), f_in.detach().cpu(), label='RFM')
    # plt.plot(x_in.cpu(), func_green_dirichlet(x_in, 0.5, k, x_max - x_min).cpu(), label='Exact')
    # plt.legend()
    # plt.show()

    domain = pyrfm.Line1D(x_min, x_max)
    x0 = 0.5
    x_in = domain.in_sample(num_samples=10000, with_boundary=False)
    x_in = torch.cat([x_in, torch.tensor([[x0]], device=x_in.device)], dim=0)
    x_in, _ = torch.sort(x_in, dim=0)
    x_on = domain.on_sample(num_samples=20)

    f_in = -func_Dirac(x_in, x0, 1e-6, x_max - x_min)
    k = 100

    model = pyrfm.RFMBase(dim=1, n_hidden=40, domain=domain, n_subdomains=k // 10)

    A_in = model.features(x_in).cat(dim=1)
    A_in_xx = model.features_second_derivative(x_in, axis1=0, axis2=0).cat(dim=1)
    A_on = model.features(x_on).cat(dim=1)

    A = pyrfm.concat_blocks([[A_in_xx + k ** 2 * A_in], [A_on]])
    b = torch.cat([f_in, torch.zeros((x_on.shape[0], 1), device=A.device)], dim=0)

    model.compute(A, damp=1e-8).solve(b)

    import matplotlib.pyplot as plt

    u_exact = func_green_dirichlet(x_in, x0, k, x_max - x_min)

    plt.plot(x_in.cpu(), u_exact.cpu())
    plt.plot(x_in.cpu(), model(x_in).detach().cpu())
    plt.show()

    error = torch.linalg.norm(model(x_in) - u_exact, dim=0, ord=2) / torch.linalg.norm(u_exact, dim=0, ord=2)
    print(f"Relative L2 Error: {error.item():.4e}")
