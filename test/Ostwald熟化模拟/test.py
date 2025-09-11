import torch
import pyrfm
import math
import random
import os, glob, shutil

try:
    import cv2
except Exception:
    cv2 = None

from typing import Tuple, List, Union

if __name__ == "__main__":
    base = pyrfm.Square2D(center=(0, 0), radius=(710.0, 710.0))

    (a, an), (b, bn), (c, cn), (d, dn) = base.on_sample(4000, with_normal=True, separate=True)
    c = c.flip(0)
    cn = cn.flip(0)
    d = d.flip(0)
    dn = dn.flip(0)
    print("a = ", a)
    print("c = ", c)
    print("b = ", b)
    print("d = ", d)
