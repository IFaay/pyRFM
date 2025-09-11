# -*- coding: utf-8 -*-
"""
Created on 6/19/25

@author: Yifei Sun
"""
import torch

if __name__ == "__main__":
    A = torch.rand((10, 5))
    A, tau = torch.geqrf(A)
    x = torch.randn(10, 1)
    y = torch.randn(5, 1)

    print(torch.ormqr(A, tau, x, left=True, transpose=False))
