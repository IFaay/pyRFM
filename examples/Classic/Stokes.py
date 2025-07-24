# -*- coding: utf-8 -*-
"""
Created on 2025/7/23

@author: Yifei Sun
"""
import time

import pyrfm
import torch
import os

"""
Consider Stokes flow defined by the system:

    -Δu(x) + ∇p(x) = f(x)        for x in Ω,
    ∇·u(x) = 0                  for x in Ω,
    u(x) = U(x)                 for x on ∂Ω.

In this problem, the pressure p is only determined up to a constant. 
To avoid difficulties, we fix the value of p at the left-bottom corner.
"""
