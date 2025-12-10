import time
import argparse

import pyrfm
import torch
import os
import sys
import scipy
import matplotlib.pyplot as plt

"""
    -Δu(x) + ∇p(x) + ξu(x) = f(x)        for x in Ω,
    ∇·u(x) - \tau ∇^2 p= 0                  for x in Ω,
    u(x) = u(x+L), p(x) = p(x+L) + C             for x on ∂Ω.

Ω is the square (0, 1) × (0, 1) with Supershape 

ξ = 1e6 if in the solid region, and ξ = 0 if in the fluid region.

"""


# def func_xi(image, x):
#     xi = torch.zeros((x.shape[0], 1), device=x.device)
#     # image:150 x 150
#     # x:[0,1], y:[0,1]
#     for i in range(x.shape[0]):
#         xi[i, 0] = 1e6 * (1 - image[int(x[i, 1] * 150), int(x[i, 0] * 150)])
#     return xi

def func_xi(image, x):
    N = image.shape[0]  # 150
    # scale coordinates [0,1] -> pixel index [0, N-1]
    ix = torch.clamp((x[:, 0] * (N - 1)).long(), 0, N - 1)
    iy = torch.clamp((x[:, 1] * (N - 1)).long(), 0, N - 1)

    mask = image[iy, ix]  # 1 = fluid, 0 = solid
    xi = (1 - mask).unsqueeze(1)  # convert to (batch,1)
    xi = xi * 1e6
    return xi


def run_rfm(image, Q=400, M=400, fx=1.0, fy=0.0):
    print("\n" + "=" * 40)
    print(f"Simulation Started with Parameters:")
    print(f"Q = {Q}, M = {M}")
    print(f"--------------------------")

    domain = pyrfm.Square2D(center=(0.5, 0.5), radius=(0.5, 0.5))

    nelx, nely = image.shape
    dx, dy = 1.0 / nelx, 1.0 / nely
    h2 = dx ** 2 + dy ** 2
    tau = h2 / 12

    model = pyrfm.RFMBase(dim=2, n_hidden=M, domain=domain, n_subdomains=1)
    x_in = domain.in_sample(Q, with_boundary=False)
    x_on = domain.on_sample(400)
    x_on_l = torch.flip(x_on, dims=[0])
    x_on_l = torch.cat([x_on_l[100:200], x_on_l[:100]], dim=0)
    x_on = x_on[:200]

    # # plot the scatter points x_on and x_on_l to verify
    # plt.scatter(x_on[:, 0].cpu(), x_on[:, 1].cpu(), color='red', label='x_on')
    # plt.scatter(x_on_l[:, 0].cpu(), x_on_l[:, 1].cpu(), color='blue', label='x_on_l')
    # plt.legend()
    # plt.show()

    x_corner = torch.tensor([[0.0, 0.0]])

    xi_x_in = func_xi(image, x_in)

    print(xi_x_in)

    u = model.features(x_in).cat(dim=1)
    u_in_x = model.features_derivative(x_in, axis=0).cat(dim=1)
    u_in_y = model.features_derivative(x_in, axis=1).cat(dim=1)
    u_in_xx = model.features_second_derivative(x_in, axis1=0, axis2=0).cat(dim=1)
    u_in_yy = model.features_second_derivative(x_in, axis1=1, axis2=1).cat(dim=1)

    u_on = model.features(x_on).cat(dim=1)
    u_on_l = model.features(x_on_l).cat(dim=1)
    u_corner = model.features(x_corner).cat(dim=1)

    # A1 = pyrfm.concat_blocks([[-(u_in_xx + u_in_yy) + xi_x_in * u, torch.zeros_like(u_in_xx), u_in_x],
    #                           [torch.zeros_like(u_in_xx), -(u_in_xx + u_in_yy) + xi_x_in * u, u_in_y],
    #                           [u_in_x, u_in_y, -tau * (u_in_xx + u_in_yy)]])

    A1 = pyrfm.concat_blocks([[-(u_in_xx + u_in_yy) + xi_x_in * u, torch.zeros_like(u_in_xx), u_in_x],
                              [torch.zeros_like(u_in_xx), -(u_in_xx + u_in_yy) + xi_x_in * u, u_in_y],
                              [u_in_x, u_in_y, torch.zeros_like(u)]])
    f1_in, f2_in = torch.ones_like(x_in[:, [0]]) * fx, torch.ones_like(x_in[:, [0]]) * fy
    b1 = torch.cat([f1_in, f2_in, torch.zeros_like(f1_in)], dim=0)
    # A2 = pyrfm.concat_blocks([[u_on - u_on_l, torch.zeros_like(u_on), torch.zeros_like(u_on)],
    #                           [torch.zeros_like(u_on), u_on - u_on_l, torch.zeros_like(u_on)]])

    # A2 = pyrfm.concat_blocks([[u_on, torch.zeros_like(u_on), torch.zeros_like(u_on)],
    #                           [torch.zeros_like(u_on), u_on, torch.zeros_like(u_on)]])

    #
    # A3 = pyrfm.concat_blocks([[torch.zeros_like(u_corner), torch.zeros_like(u_corner), u_corner]])
    # b3 = torch.tensor([[2]])

    A2 = pyrfm.concat_blocks([[u_on - u_on_l, torch.zeros_like(u_on), torch.zeros_like(u_on)],
                              [torch.zeros_like(u_on), u_on - u_on_l, torch.zeros_like(u_on)]])
    b2 = torch.cat([torch.zeros_like(u_on[:, 0:1]), torch.zeros_like(u_on[:, 0:1])], dim=0)

    A3 = pyrfm.concat_blocks([[torch.zeros_like(u_on), torch.zeros_like(u_on), u_on - u_on_l]])
    # b3 = torch.zeros_like(u_on[:, 0:1])
    b3 = torch.ones_like(u_on[:, 0:1]) * 1

    A = torch.cat([A1, A2, A3], dim=0)
    b = torch.cat([b1, b2, b3], dim=0)
    model.compute(A).solve(b)

    visualizer = pyrfm.RFMVisualizer2D(model, component_idx=0)
    visualizer.plot()
    visualizer.show()

    uvp = model.forward(x_in)
    u, v = uvp[:, [0]], uvp[:, [1]]
    C0 = torch.sum(u) / Q
    C1 = torch.sum(v) / Q

    print(C0, C1)
    return C0, C1


def get_C(image):
    C00, C10 = run_rfm(image, Q=400, M=400, fx=1.0, fy=0.0)
    C01, C11 = run_rfm(image, Q=400, M=400, fx=0.0, fy=1.0)
    return C00, C10, C01, C11


# C00, C10, C01, C11 = get_C(mstr_image_torch[test_image_id, :, :])
# print(f"C00: {C00}, C10: {C01}, C01: {C10}, C11: {C11}")
# print(c00[test_image_id], c10[test_image_id], c01[test_image_id], c11[test_image_id])

if __name__ == "__main__":
    mstr_image = scipy.io.loadmat(f'./mstr_images_1.mat')['mstr_images']
    mstr_homog_data = scipy.io.loadmat(f'./homogen_data_1.mat')
    c00, c10, c01, c11 = (mstr_homog_data['c00'], mstr_homog_data['c10'], mstr_homog_data['c01'],
                          mstr_homog_data['c11'])
    # torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')
    mstr_image_torch = torch.tensor(mstr_image, dtype=torch.long)
    test_image_id = 8
    plt.imshow(mstr_image[test_image_id, :, :], cmap='gray')
    C00, C10, C01, C11 = get_C(mstr_image_torch[test_image_id, :, :])
    print(f"C00: {C00}, C10: {C01}, C01: {C10}, C11: {C11}")
    print(c00[test_image_id], c10[test_image_id], c01[test_image_id], c11[test_image_id])
