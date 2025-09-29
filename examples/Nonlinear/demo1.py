# -*- coding: utf-8 -*-
"""
Nonlinear PDE solver benchmark with pyRFM
遍历 CPU/GPU 和不同非线性最小二乘方法
"""

import time
import math
import torch
import pyrfm


def func_u(x):
    # Exact solution
    return -0.5 * (1.5 * torch.cos(torch.pi * x[:, [0]] + 2 * torch.pi / 5) +
                   2 * torch.cos(2 * torch.pi * x[:, [0]] - torch.pi / 5)) * \
        (1.5 * torch.cos(torch.pi * x[:, [1]] + 2 * torch.pi / 5) +
         2 * torch.cos(2 * torch.pi * x[:, [1]] - torch.pi / 5)) - \
        0.5 * (1.5 * torch.cos(2 * torch.pi * x[:, [0]] + 2 * torch.pi / 5) +
               2 * torch.cos(4 * torch.pi * x[:, [0]] - torch.pi / 5)) * \
        (1.5 * torch.cos(2 * torch.pi * x[:, [1]] + 2 * torch.pi / 5) +
         2 * torch.cos(4 * torch.pi * x[:, [1]] - torch.pi / 5))


def func_f(x):
    u = func_u(x)
    return -(-0.5 * (-1.5 * torch.pi ** 2 * torch.cos(torch.pi * x[:, [0]] + 2 * torch.pi / 5) -
                     2 * (2 * torch.pi) ** 2 * torch.cos(2 * torch.pi * x[:, [0]] - torch.pi / 5)) *
             (1.5 * torch.cos(torch.pi * x[:, [1]] + 2 * torch.pi / 5) +
              2 * torch.cos(2 * torch.pi * x[:, [1]] - torch.pi / 5)) -
             0.5 * (1.5 * torch.cos(torch.pi * x[:, [0]] + 2 * torch.pi / 5) +
                    2 * torch.cos(2 * torch.pi * x[:, [0]] - torch.pi / 5)) *
             (-1.5 * torch.pi ** 2 * torch.cos(torch.pi * x[:, [1]] + 2 * torch.pi / 5) -
              2 * (2 * torch.pi) ** 2 * torch.cos(2 * torch.pi * x[:, [1]] - torch.pi / 5)) -
             0.5 * (-1.5 * (2 * torch.pi) ** 2 * torch.cos(2 * torch.pi * x[:, [0]] + 2 * torch.pi / 5) -
                    2 * (4 * torch.pi) ** 2 * torch.cos(4 * torch.pi * x[:, [0]] - torch.pi / 5)) *
             (1.5 * torch.cos(2 * torch.pi * x[:, [1]] + 2 * torch.pi / 5) +
              2 * torch.cos(4 * torch.pi * x[:, [1]] - torch.pi / 5)) -
             0.5 * (1.5 * torch.cos(2 * torch.pi * x[:, [0]] + 2 * torch.pi / 5) +
                    2 * torch.cos(4 * torch.pi * x[:, [0]] - torch.pi / 5)) *
             (-1.5 * (2 * torch.pi) ** 2 * torch.cos(2 * torch.pi * x[:, [1]] + 2 * torch.pi / 5) -
              2 * (4 * torch.pi) ** 2 * torch.cos(4 * torch.pi * x[:, [1]] - torch.pi / 5))) + u ** 2


def func_g(x):
    return func_u(x)


def run_once(method: str, device: str):
    """
    Run one solve with specified method and device.
    Return runtime, relative error, status.
    """
    torch.set_default_device(device)
    start_time = time.time()

    # Domain & model
    domain = pyrfm.Square2D(center=[0.5, 0.5], radius=[0.5, 0.5])
    model = pyrfm.RFMBase(dim=2, n_hidden=300, domain=domain, n_subdomains=2, pou=pyrfm.PsiB)

    # Sampling
    x_in = domain.in_sample(8000, with_boundary=False)
    x_on = domain.on_sample(2000)

    # Feature matrices
    A_in = model.features(x_in).cat(dim=1)
    A_in_xx = model.features_second_derivative(x_in, axis1=0, axis2=0).cat(dim=1)
    A_in_yy = model.features_second_derivative(x_in, axis1=1, axis2=1).cat(dim=1)
    A_on = model.features(x_on).cat(dim=1)

    # RHS
    f_in = func_f(x_in)
    f_on = func_g(x_on)

    # Residual and Jacobian
    def fcn(w):
        u = A_in @ w
        u_xx = A_in_xx @ w
        u_yy = A_in_yy @ w
        u_on = A_on @ w
        return torch.cat([(-u_xx - u_yy + u ** 2) - f_in, u_on - f_on])

    def jac(w):
        return torch.cat([-A_in_xx - A_in_yy + 2 * (A_in @ w) * A_in, A_on], dim=0)

    # Solve
    tol = 1e-8
    x0 = torch.zeros((A_in.shape[1], 1))

    result = pyrfm.nonlinear_least_square(
        fcn=fcn, x0=x0, jac=jac,
        ftol=tol, gtol=tol, xtol=tol,
        method=method, verbose=2
    )

    runtime = time.time() - start_time

    # Extract solution
    w_opt, status = result[0], result[1]
    model.W = w_opt

    # Test error
    x_test = domain.in_sample(2000, with_boundary=True)
    with torch.no_grad():
        u_test = func_u(x_test).view(-1, 1)
        u_pred = model(x_test)
        rel_err = (u_test - u_pred).norm() / u_test.norm()

    return runtime, float(rel_err.cpu()), int(status)


if __name__ == '__main__':
    methods = ['newton', 'trf', 'lm', 'dogbox']
    devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])

    results = []
    for device in devices:
        for method in methods:
            try:
                runtime, rel_err, status = run_once(method, device)
                results.append({
                    'device': device,
                    'method': method,
                    'time_s': runtime,
                    'rel_error': rel_err,
                    'status': status,
                })
            except Exception as e:
                results.append({
                    'device': device,
                    'method': method,
                    'time_s': math.nan,
                    'rel_error': math.nan,
                    'status': -999,
                })
                print(f"[WARN] {method} on {device} failed: {e}")

    # Print results
    header = f"{'Device':<8}  {'Method':<7}  {'Time (s)':>10}  {'Rel. Error':>12}  {'Status':>6}"
    print(header)
    print('-' * len(header))
    for r in results:
        print(f"{r['device']:<8}  {r['method']:<7}  {r['time_s']:>10.4f}  {r['rel_error']:>12.3e}  {r['status']:>6}")

    for device in devices:
        subset = [r for r in results if r['device'] == device and not math.isnan(r['rel_error'])]
        if subset:
            best = min(subset, key=lambda x: x['rel_error'])
            print(f"Best on {device}: {best['method']} "
                  f"(err={best['rel_error']:.3e}, time={best['time_s']:.3f}s, status={best['status']})")
