# -*- coding: utf-8 -*-
"""
Created on 2025/9/2

@author: Yifei Sun

说明：
- k 由 JSON 自动计算；sigma == 0 → 实数 (torch.float64)；sigma > 0 → 复数 (torch.complex128)。
- Green 函数与数值解均兼容两种 dtype。
"""

import torch
import pyrfm
import json
import math
import cmath
import matplotlib.pyplot as plt

# =========================
# JSON 输入
# =========================
json_str = """
{
  "material": {
    "relative_permittivity": 2.2,
    "relative_permeability": 1.0,
    "conductivity": 1.1e-4
  },
  "source": {
    "type": "discrete_port",
    "position": 0.2,
    "amplitude": 1.0,
    "phase": 0.1,
    "unit": "V"
  },
  "boundary_conditions": {
    "left":  { "type": "dirichlet", "value": 0.0, "unit": "V"   },
    "right": { "type": "neumann",   "value": 0.0, "unit": "V/m" }
  },
  "frequency": 1.0e9
}
"""


# =========================
# 常量 & 工具函数
# =========================
def compute_k_from_json(data):
    """根据 JSON 计算传播常数 k；sigma==0 返回 float64，sigma>0 返回 complex128（主值）"""
    eps0 = 8.854187817e-12  # F/m
    mu0 = 4.0 * math.pi * 1e-7  # H/m

    er = float(data["material"]["relative_permittivity"])
    ur = float(data["material"]["relative_permeability"])
    sigma = float(data["material"]["conductivity"])  # S/m
    f = float(data["frequency"])  # Hz
    omega = 2.0 * math.pi * f

    eps = er * eps0
    mu = ur * mu0

    if sigma == 0.0:
        # 无耗：k 实数
        k_val = omega * math.sqrt(mu * eps)
        return torch.tensor(k_val, dtype=torch.float64)
    else:
        # 有耗：k 复数（主值）
        k2 = (omega ** 2) * mu * eps - 1j * omega * mu * sigma
        k_val = cmath.sqrt(k2)
        return torch.tensor(k_val, dtype=torch.complex128)


def func_Dirac(x, x0, sigma, L):
    """
    高斯极限近似 δ(x-x0)，归一化到区间积分≈1/L（便于与离散采样匹配）
    返回与 x 相同 device；dtype 为 float64（源项为实）
    """
    device = x.device
    x0_t = torch.as_tensor(x0, dtype=torch.float64, device=device)
    sigma_t = torch.as_tensor(sigma, dtype=torch.float64, device=device)
    L_t = torch.as_tensor(L, dtype=torch.float64, device=device)

    coeff = 1.0 / (sigma_t * torch.sqrt(torch.tensor(2.0 * math.pi, dtype=torch.float64, device=device)))
    r = (x.to(torch.float64) - x0_t).norm(dim=1, p=2, keepdim=True)
    gauss = torch.exp(-0.5 * (r / sigma_t) ** 2) * coeff
    gauss /= gauss.mean()
    return gauss / L_t  # shape: (N,1), float64


def func_green_dirichlet(x, x0, k, L):
    """
    一维 Dirichlet(0) - Neumann(0) 的 Green 函数（点源在 x0），e^{-iωt} 约定。
    关键修复：比较操作在实数 dtype 上进行，避免 CUDA 复数比较不支持的问题。
    """
    device = x.device

    # 1) 先确定目标计算 dtype（复数或实数）
    if torch.is_tensor(k):
        k_t = k
    else:
        k_t = torch.tensor(k)

    cdtype = torch.complex128 if torch.is_complex(k_t) else torch.float64

    # 2) 掩码用实数来比较（CUDA 不支持复数比较）
    x_float = x.to(torch.float64)
    x0_float = torch.tensor(x0, dtype=torch.float64, device=device).expand_as(x_float)
    mask = (x_float < x0_float)  # bool，device 同 x

    # 3) 物理量转为目标计算 dtype（用于三角函数）
    x_c = x_float.to(cdtype)
    x0_c = x0_float.to(cdtype)
    L_c = torch.tensor(L, dtype=cdtype, device=device)
    k_c = k_t.to(cdtype).to(device)

    # 4) 计算左右分支与分母（统一 dtype）
    f_left = torch.sin(k_c * x_c) * torch.sin(k_c * (L_c - x0_c))
    f_right = torch.sin(k_c * x0_c) * torch.sin(k_c * (L_c - x_c))
    denom = k_c * torch.sin(k_c * L_c)

    # 5) 按掩码选择
    return torch.where(mask, f_left, f_right) / denom


# =========================
# 主程序
# =========================
if __name__ == "__main__":
    # 设备
    torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')

    # 采样点
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
    """.strip().split()

    points = torch.tensor([float(x) for x in input_points], dtype=torch.float64).reshape(-1, 1)
    x_min = points[:, 0].min().item()
    x_max = points[:, 0].max().item()
    L = x_max - x_min

    # 解析 JSON & k
    data = json.loads(json_str)
    k = compute_k_from_json(data)  # torch.float64 或 torch.complex128
    print("Computed k:", k)
    sigma = float(data["material"]["conductivity"])

    # 几何/源
    x0 = 0.5  # 点源位置（可改为 data["source"]["position"] 若要与 JSON 绑定）
    x0 = float(data["source"]["position"]) if "position" in data.get("source", {}) else x0

    # 域与采样
    domain = pyrfm.Line1D(x_min, x_max)
    num_samples = int((x_max - x_min) * k.abs().item() * 5) * 10  # 简单经验：每波长采 5 个点
    print("Number of interior samples (approx.):", int(num_samples))
    x_in = domain.in_sample(num_samples=num_samples, with_boundary=False)
    x_in = torch.cat([x_in, torch.tensor([[x0]], dtype=torch.float64, device=x_in.device)], dim=0)
    x_in, _ = torch.sort(x_in, dim=0)
    x_on = domain.on_sample(num_samples=2)

    # 源项（实数）
    f_in = func_Dirac(x_in, x0, 1e-14, L)  # (N,1) float64
    phase = data["source"]["phase"]
    if phase > 0:
        Q = data["source"]["amplitude"] * (math.cos(phase) + 1j * math.sin(phase))
    else:
        Q = data["source"]["amplitude"]

    f_in = -f_in * Q

    use_complex = True if torch.is_complex(k) or torch.is_complex(f_in) else False

    # —— 解析解（用于对比）——
    u_exact = func_green_dirichlet(x_in, x0, k, L) * Q  # (N,1) float/complex

    # =========================
    # RFM 数值解
    # =========================
    # if use_complex:
    #     # ---------- 无耗：单模型、实数 ----------
    #     n_sub = max(1, int(abs(float(k)) // 10))  # 简单经验
    #     model = pyrfm.RFMBase(dim=1, n_hidden=40, domain=domain, n_subdomains=n_sub)
    #
    #     A_in = model.features(x_in).cat(dim=1)  # (N, M)
    #     A_in_xx = model.features_second_derivative(x_in, axis1=0, axis2=0).cat(dim=1)  # (N, M)
    #     A_on = model.features(x_on).cat(dim=1)  # (Nb, M)
    #
    #     A = torch.cat([A_in_xx + (float(k) ** 2) * A_in, A_on], dim=0)  # (N+Nb, M), float64
    #     b = torch.cat([f_in, torch.zeros((x_on.shape[0], 1), dtype=torch.float64, device=A.device)], dim=0)
    #
    #     model.compute(A, damp=1e-8).solve(b)
    #
    #     u_num = model(x_in)  # (N,1) float64
    #
    #     # 误差（相对 L2）
    #     err = torch.linalg.norm(u_num - u_exact.to(torch.float64)) / torch.linalg.norm(u_exact.to(torch.float64))
    #
    #     # 绘图（实数）
    #     plt.figure()
    #     plt.plot(x_in.cpu(), u_exact.real.detach().cpu(), label='Exact (real)')
    #     plt.plot(x_in.cpu(), u_num.detach().cpu(), '--', label='RFM (real)')
    #     plt.legend();
    #     plt.xlabel('x [m]');
    #     plt.ylabel('u')
    #     plt.title('Helmholtz 1D (lossless)')
    #     plt.show()
    #
    #     print(f"Relative L2 Error (real): {err.item():.4e}")
    #
    # else:
    # ---------- 无耗：单模型、实数 ----------
    n_sub = max(1, int(k.abs().item() // 10))  # 简单经验
    print("Number of subdomains:", n_sub)
    model = pyrfm.RFMBase(dim=1, n_hidden=40, domain=domain, n_subdomains=n_sub)

    A_in = model.features(x_in).cat(dim=1)  # (N, M)
    A_in_xx = model.features_second_derivative(x_in, axis1=0, axis2=0).cat(dim=1)  # (N, M)
    A_on = model.features(x_on).cat(dim=1)  # (Nb, M)

    A = torch.cat([A_in_xx + k ** 2 * A_in, A_on], dim=0)  # (N+Nb, M), float64
    b = torch.cat([f_in, torch.zeros((x_on.shape[0], 1), dtype=torch.float64, device=A.device)], dim=0)
    model.compute(A, damp=1e-8, use_complex=True).solve(b)
    u_num = model(x_in)

    A = model.features_second_derivative(x_in, axis1=0, axis2=0).cat(dim=1) + k ** 2 * model.features(x_in).cat(
        dim=1)
    A /= A.norm(dim=1, p=2, keepdim=True)  # 列归一化，避免 k 很大时 PDE loss 过大
    res = torch.linalg.norm(A.to(device=model.device, dtype=model.dtype) @ model.W) / torch.linalg.norm(
        torch.ones(A.shape[0]))

    print("PDE Loss : {:.4e}".format(res))

    # # 误差：分别评估实部/虚部（或幅值）
    u_ex = u_exact.to(model.dtype)
    err_magnitude = torch.linalg.norm(torch.abs(u_num) - torch.abs(u_ex)) / torch.linalg.norm(torch.abs(u_ex))
    err_real = torch.linalg.norm(u_num.real - u_ex.real) / torch.linalg.norm(u_ex.real)
    # # 若解析虚部接近 0，分母可能很小；加个保护
    denom_im = torch.linalg.norm(u_ex.imag).clamp_min(1e-16)
    err_imag = torch.linalg.norm(u_num.imag - u_ex.imag) / denom_im

    # 绘图：幅值 & 实部 & 虚部

    plt.figure()
    plt.plot(x_in.cpu(), u_ex.real.detach().cpu(), label='Exact Re(u)')
    plt.plot(x_in.cpu(), u_num.real.detach().cpu(), '--', label='RFM Re(u)')
    plt.legend();
    plt.xlabel('x [m]');
    plt.ylabel('Re(u)')
    plt.title('Helmholtz 1D (lossy) - Real part')
    plt.show()

    plt.figure()
    plt.plot(x_in.cpu(), u_ex.imag.detach().cpu(), label='Exact Im(u)')
    plt.plot(x_in.cpu(), u_num.imag.detach().cpu(), '--', label='RFM Im(u)')
    plt.legend();
    plt.xlabel('x [m]');
    plt.ylabel('Im(u)')
    plt.title('Helmholtz 1D (lossy) - Imaginary part')
    plt.show()

    plt.figure()
    plt.plot(x_in.cpu(), torch.abs(u_ex).detach().cpu(), label='Exact |u|')
    plt.plot(x_in.cpu(), torch.abs(u_num).detach().cpu(), '--', label='RFM |u|')
    plt.legend();
    plt.xlabel('x [m]');
    plt.ylabel('|u|')
    plt.title('Helmholtz 1D (lossy) - Magnitude')
    plt.show()

    print(f"Relative L2 Error (magnitude): {err_magnitude.item():.4e}")
    print(f"Relative L2 Error (real): {err_real.item():.4e}")
    print(f"Relative L2 Error (imag): {err_imag.item():.4e}")
