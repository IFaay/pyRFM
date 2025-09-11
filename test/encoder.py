# -*- coding: utf-8 -*-
"""
Created on 6/6/25

@author: Yifei Sun
"""
from logging import critical

import pyrfm
import torch
import torch.nn as nn
import torch.optim as optim


def u(x):
    # $u(\mathbf{x})=$ $\left(\frac{1}{d} \sum_{i=1}^d x_i\right)^2+\sin \left(\frac{1}{d} \sum_{i=1}^d x_i\right), \forall \mathbf{x} \in \mathbf{R}^d$,
    return ((1.0 / x.shape[1] * x.sum(dim=1, keepdim=True)) ** 2
            + torch.sin(1.0 / x.shape[1] * x.sum(dim=1, keepdim=True)))


def f(x):
    return - (2.0 - torch.sin(1.0 / x.shape[1] * x.sum(dim=1, keepdim=True))) / x.shape[1]


def g(x):
    return u(x)


class AutoRegressiveNet(nn.Module):
    def __init__(self, d, M):
        super().__init__()
        self.encoder = nn.Linear(d, M)
        self.decoder = nn.Linear(M, d)
        self.net = nn.Sequential(
            self.encoder, nn.Tanh(), self.decoder
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":

    seed = 2025
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')
    dim = 30

    d = dim  # input/output dimension
    M = 6400  # hidden layer dimension
    lr = 1e-3  # learning rate
    epochs = 100
    batch_size = 32  # batch size

    torch.manual_seed(0)

    automodel = AutoRegressiveNet(d, M)
    # optimizer = optim.AdamW(automodel.parameters(), lr=lr)
    optimizer = optim.Adam(automodel.parameters(), lr=lr)
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    # criterion = nn.SmoothL1Loss()

    domain = pyrfm.HyperCube(dim=dim)
    x_in = domain.in_sample(10000, with_boundary=False)
    x_on = domain.on_sample(400 * 2 * dim)

    # Combine x_in and x_on, then randomly sample batch_size rows
    x_pool = torch.cat([x_in, x_on], dim=0)

    perm = torch.randperm(x_pool.size(0))  # 初始 perm
    pointer = 0  # 当前 perm 的位置

    for epoch in range(epochs):
        perm = torch.randperm(x_pool.size(0))
        for i in range(0, x_pool.size(0), batch_size):
            idx = perm[i:i + batch_size]
            x_batch = x_pool[idx]
            output = automodel(x_batch)
            loss = criterion(output, x_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.6f}")

        # === 可视化 encoder 的权重与偏置分布 ===
        import matplotlib.pyplot as plt

        with torch.no_grad():
            W = automodel.encoder.weight.detach().view(-1).cpu().numpy()
            b = automodel.encoder.bias.detach().view(-1).cpu().numpy()

        w_mean, w_std = W.mean(), W.std()
        b_mean, b_std = b.mean(), b.std()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=150)

        # weights 直方图
        axes[0].hist(W, bins=100, density=True)
        axes[0].axvline(w_mean, linestyle="--", linewidth=1)
        axes[0].set_title("Encoder Weights Histogram")
        axes[0].set_xlabel("Value")
        axes[0].set_ylabel("Density")
        axes[0].grid(alpha=0.3)
        axes[0].text(0.98, 0.98, f"mean={w_mean:.3e}\nstd={w_std:.3e}",
                     ha="right", va="top", transform=axes[0].transAxes)

        # bias 直方图
        axes[1].hist(b, bins=60, density=True)
        axes[1].axvline(b_mean, linestyle="--", linewidth=1)
        axes[1].set_title("Encoder Biases Histogram")
        axes[1].set_xlabel("Value")
        axes[1].set_ylabel("Density")
        axes[1].grid(alpha=0.3)
        axes[1].text(0.98, 0.98, f"mean={b_mean:.3e}\nstd={b_std:.3e}",
                     ha="right", va="top", transform=axes[1].transAxes)

        plt.tight_layout()
        # plt.savefig("encoder_params_hist.png", bbox_inches="tight", dpi=200)
        # print("Saved histogram to encoder_params_hist.png")
        plt.show()  # 若本地运行并希望弹窗显示，可取消注释


    # test_x = torch.rand(1, d) * 2.0 - 1.0
    # test_out = automodel(test_x)
    # print("Input:\n", test_x)
    # print("Output:\n", test_out)

    # print("Model parameters:")
    # for name, param in automodel.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name}: {param.data.shape}")
    #         print(param.data)

    class CustomRF(pyrfm.RFTanH):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # torch.nn.init.xavier_uniform_(self.weights, gain=torch.nn.init.calculate_gain('tanh'))
            # torch.nn.init.normal_(self.biases, mean=0.0, std=1 / self.n_hidden)

            self.weights = automodel.encoder.weight.t()
            self.biases = automodel.encoder.bias.view(1, self.n_hidden)
            # print("weights:\n", self.weights.shape)
            # print("biases:\n", self.biases.shape)

        def forward(self, x):
            return super(CustomRF, self).forward(x)


    model = pyrfm.RFMBase(rf=CustomRF, dim=dim, n_hidden=M, domain=domain, n_subdomains=1, seed=seed)

    x_in = domain.in_sample(1000, with_boundary=False)
    x_on = domain.on_sample(200 * 2 * dim)
    A_in_xx = model.features_second_derivative(x_in, axis1=0, axis2=0).cat(dim=1)
    for i in range(1, dim):
        A_in_xx += model.features_second_derivative(x_in, axis1=i, axis2=i).cat(dim=1)

    A_on = model.features(x_on).cat(dim=1)

    A = pyrfm.concat_blocks([[-A_in_xx], [A_on]])

    f_in = f(x_in).view(-1, 1)
    f_on = g(x_on).view(-1, 1)

    f = pyrfm.concat_blocks([[f_in], [f_on]])

    model.compute(A).solve(f)

    x_test = domain.in_sample(40, with_boundary=True)
    # x_test = x_in

    u_test = u(x_test).view(-1, 1)

    u_pred = model(x_test)

    print((u_test - u_pred).norm() / u_test.norm())
