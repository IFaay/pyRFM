import pyrfm
import numpy as np
import matplotlib.pyplot as plt
import torch
import moderngl
import moderngl_window as mglw
from moderngl_window import geometry
import math
from moderngl_window.integrations.imgui import ModernglWindowRenderer
import imgui
import sys

from pyrfm.core import *


class VisualizerBase:
    def __init__(self):
        pass

    def plot(self, *args, **kwargs):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def show(self, *args, **kwargs):
        raise NotImplementedError("This method should be implemented by subclasses.")


class RFMVisualizer2D(VisualizerBase):
    def __init__(self, model: RFMBase):
        super().__init__()


class RFMVisualizer(VisualizerBase):
    def __init__(self, model: RFMBase, resolution=(1920, 1080)):
        super().__init__()
        pass


if __name__ == "__main__":
    domain = pyrfm.ExtrudeBody(pyrfm.Circle2D(radius=0.5, center=(0, 0)), direction=[-10.0, -1.0, 0.0])
    print(domain.glsl_sdf())
