# -*- coding: utf-8 -*-
"""
Created on 2024/12/15

@author: Yifei Sun
"""

import math

import spdlog

logger = spdlog.ConsoleLogger('rfm_logger')
logger.set_level(spdlog.LogLevel.DEBUG)

import numpy as np

import torch
import torch.nn as nn

import time
from abc import ABC, abstractmethod
from typing import Optional, Any, Union, Tuple, List, Callable, Dict
from collections import Counter
from enum import Enum

from math import prod

# torch.backends.cuda.preferred_linalg_library('cusolver')  # or 'magma'
torch.set_default_dtype(torch.float64)
torch.set_default_device(torch.device("cpu"))


class Tensor:
    """
        A Tensor class that acts as a multi-dimensional container similar to a list.

        Note:
        This class is **not** `torch.Tensor`. It is a standalone implementation
        designed for handling multi-dimensional data using nested Python lists.

        This class supports:
          - Multi-dimensional indexing: Elements can be accessed and modified using indices for each dimension.
          - Shape definition: Allows defining the shape of the tensor explicitly or inferring it from the provided data.
          - Flattened and nested representations: Maintains both a nested structure (for easy access) and a flat list (for efficient operations).
    """

    def __init__(self, data=None, shape=None):
        """
        Initialize the Tensor object. By default, initializes an empty tensor.
        :param data: Data for the tensor (can be a nested list or other data types).
        :param shape: Optional shape of the tensor.
        """
        if data is None:  # Handle empty tensor case
            self.data = []
            self.shape = (0,)
            self.flat_data = []
        else:
            if shape is not None:
                if isinstance(data, list) and len(data) != self._prod(shape):
                    raise ValueError(f"Cannot reshape array of size {len(data)} into shape {shape}")
                self.shape = shape
                self.data = self._unflatten(data, shape)
                self.flat_data = list(data)
            else:
                self.data = data
                self.shape = self._infer_shape(data)
                self.flat_data = self._flatten(data)

    @classmethod
    def __class_getitem__(cls, item):
        """
        Support for Tensor[<any_type>] syntax.
        :param item: Type hint (could be any type).
        :return: A formatted string representing the type.
        """
        # If the item is a type object, retrieve its name
        if hasattr(item, "__name__"):
            type_name = item.__name__
        # For other type hints (e.g., Union, List), convert them to strings
        else:
            type_name = str(item)

        return f"{cls.__name__}[{type_name}]"

    def cat(self, dim: int = 1) -> torch.Tensor:
        """
        Concatenate the tensor along the specified dimension.

        :param dim: Dimension along which to concatenate.
        :return: Concatenated tensor as a torch.Tensor.
        :raises TypeError: If any element in self.data is not a torch.Tensor.
        :raises ValueError: If tensor shapes are incompatible for concatenation.
        """
        if not all(isinstance(sub, torch.Tensor) for sub in self.flat_data):
            raise TypeError("All elements in self.data must be torch.Tensor to use cat.")

        base_shape = list(self.flat_data[0].shape)
        base_shape[dim] = -1  # 忽略拼接维度
        for tensor in self.flat_data:
            current_shape = list(tensor.shape)
            if current_shape[:dim] + current_shape[dim + 1:] != base_shape[:dim] + base_shape[dim + 1:]:
                raise ValueError(f"Incompatible shapes for concatenation: {self.data[0].shape} and {tensor.shape}")

        return torch.cat(self.flat_data, dim=dim)

    def size(self):
        """
        Return the shape of the tensor (similar to torch.Tensor.size()).
        :return: Tuple representing the shape.
        """
        return self.shape

    def numel(self):
        """
        Return the total number of elements in the tensor (similar to torch.Tensor.numel()).
        :return: Integer representing the total number of elements.
        """
        return len(self.flat_data)

    def reshape(self, *new_shape):
        """
        Reshape the tensor to a new shape (similar to torch.Tensor.reshape()).
        :param new_shape: New shape as a tuple.
        :return: Tensor object with the new shape.
        """
        if self.numel() != self._prod(new_shape):
            raise ValueError("Cannot reshape array of size {} into shape {}".format(self.numel(), new_shape))
        self.shape = new_shape
        self.data = self._unflatten(self.flat_data, new_shape)
        return self

    def __neg__(self):
        """
        Unary negation of the tensor (element-wise).
        :return: New Tensor with negated values.
        """
        result_data = [-x for x in self.flat_data]
        return Tensor(result_data, shape=self.shape)

    def __add__(self, other):
        """
        Element-wise addition with another tensor or scalar.
        :param other: Tensor or scalar.
        :return: New Tensor with the result.
        """
        if isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError("Tensors must have the same shape for element-wise addition.")
            result_data = [a + b for a, b in zip(self.flat_data, other.flat_data)]
        else:  # Scalar addition
            result_data = [a + other for a in self.flat_data]
        return Tensor(result_data, shape=self.shape)

    def __sub__(self, other):
        """
        Element-wise subtraction with another tensor or scalar.
        :param other: Tensor or scalar.
        :return: New Tensor with the result.
        """
        if isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError("Tensors must have the same shape for element-wise subtraction.")
            result_data = [a - b for a, b in zip(self.flat_data, other.flat_data)]
        else:  # Scalar subtraction
            result_data = [a - other for a in self.flat_data]
        return Tensor(result_data, shape=self.shape)

    def __mul__(self, other):
        """
        Element-wise multiplication with another tensor or scalar.
        :param other: Tensor or scalar.
        :return: New Tensor with the result.
        """
        if isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError("Tensors must have the same shape for element-wise multiplication.")
            result_data = [a * b for a, b in zip(self.flat_data, other.flat_data)]
        else:  # Scalar multiplication
            result_data = [a * other for a in self.flat_data]
        return Tensor(result_data, shape=self.shape)

    def __truediv__(self, other):
        """
        Element-wise division with another tensor or scalar.
        :param other: Tensor or scalar.
        :return: New Tensor with the result.
        """
        if isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError("Tensors must have the same shape for element-wise division.")
            result_data = [a / b if b != 0 else float('inf') for a, b in zip(self.flat_data, other.flat_data)]
        else:  # Scalar division
            if other == 0:
                raise ZeroDivisionError("Division by zero is not allowed.")
            result_data = [a / other for a in self.flat_data]
        return Tensor(result_data, shape=self.shape)

    def _infer_shape(self, data):
        """
        Infer the shape of the tensor.
        :param data: Nested list.
        :return: Tuple representing the shape.
        """
        if isinstance(data, list):
            if len(data) == 0:
                return (0,)
            return (len(data), *self._infer_shape(data[0]))
        return ()

    def _flatten(self, data):
        """
        Flatten a nested list into a 1D list.
        :param data: Nested list.
        :return: Flattened 1D list.
        """
        if isinstance(data, list):
            return [item for sublist in data for item in self._flatten(sublist)]
        return [data]

    def _unflatten(self, flat_data, shape):
        """
        Reconstruct a nested list from a flattened list based on the given shape.
        :param flat_data: Flattened 1D list.
        :param shape: Target shape.
        :return: Nested list.
        """
        if len(shape) == 1:
            return flat_data[:shape[0]]
        size = shape[0]
        sub_shape = shape[1:]
        step = int(len(flat_data) / size)
        return [self._unflatten(flat_data[i * step:(i + 1) * step], sub_shape) for i in range(size)]

    def __getitem__(self, indices) -> torch.Tensor:
        """
        Override [] operator to support multidimensional and 1D indexing.
        :param indices: Indices for accessing elements.
        :return: Retrieved value.
        """
        if isinstance(indices, int):  # 1D indexing
            return self.flat_data[indices]
        if not isinstance(indices, tuple):
            indices = (indices,)
        return self._get_item(self.data, indices)

    def __setitem__(self, indices, value):
        """
        Override [] operator to support multidimensional and 1D assignments.
        :param indices: Indices for setting elements.
        :param value: Value to set.
        """
        if isinstance(indices, int):  # 1D assignment
            self.flat_data[indices] = value
            self.data = self._unflatten(self.flat_data, self.shape)
        else:
            if not isinstance(indices, tuple):
                indices = (indices,)
            self._set_item(self.data, indices, value)
            self.flat_data = self._flatten(self.data)

    def _get_item(self, data, indices):
        """
        Recursively retrieve the element at the specified indices.
        :param data: Current nested list.
        :param indices: Tuple of indices.
        :return: Retrieved element.
        """
        if len(indices) == 0:
            return data
        return self._get_item(data[indices[0]], indices[1:])

    def _set_item(self, data, indices, value):
        """
        Recursively set the value at the specified indices.
        :param data: Current nested list.
        :param indices: Tuple of indices.
        :param value: Value to set.
        """
        if len(indices) == 1:
            data[indices[0]] = value
        else:
            self._set_item(data[indices[0]], indices[1:], value)

    def _prod(self, iterable):
        """
        Compute the product of all elements in an iterable.
        """
        result = 1
        for x in iterable:
            result *= x
        return result

    def __iter__(self):
        """
        Enable iteration over the tensor using both 1D and multi-dimensional indices.
        """
        self._current_index = 0
        return self

    def __next__(self):
        """
        Returns the next element and its index during iteration.
        """
        if self._current_index >= self.numel():
            raise StopIteration
        multi_index = self._get_multi_index(self._current_index)
        value = self.flat_data[self._current_index]
        self._current_index += 1
        return multi_index, value

    def _get_multi_index(self, flat_index):
        """
        Convert a flat index to a multi-dimensional index.
        :param flat_index: 1D index.
        :return: Tuple of multi-dimensional indices.
        """
        indices = []
        for dim in reversed(self.shape):
            indices.append(flat_index % dim)
            flat_index //= dim
        return tuple(reversed(indices))

    def __repr__(self):
        """
        String representation of the Tensor object.
        :return: String representation with shape and data.
        """
        return f"Tensor(shape={self.shape}, data={self.data})"


def concat_blocks(blocks: List[List[torch.Tensor]]) -> Union[torch.Tensor, Tensor]:
    """
    Construct a block matrix from a 2D list of tensors.

    :param blocks: 2D list of tensors (e.g., [[a, b], [c, d]])
    :return: A single tensor representing the block matrix.
    """
    if isinstance(blocks[0][0], Tensor):
        rows = len(blocks)
        cols = len(blocks[0])
        tensor_shape = blocks[0][0].shape
        num_elements = blocks[0][0].numel()

        result_flat = []

        for i in range(num_elements):
            to_concat = []
            for r in range(rows):
                for c in range(cols):
                    to_concat.append(blocks[r][c].flat_data[i])
            result_flat.append(torch.cat(to_concat, dim=0))  # or dim=1 depending on use case

        return Tensor(result_flat, shape=tensor_shape)

    # 按行拼接每一行块
    rows = [torch.cat(row, dim=1) for row in blocks]
    # 再按列拼接所有行
    return torch.cat(rows, dim=0)


def spilit_blocks(matrix: torch.Tensor, dim: int = 1, n_blocks: int = 2, split_size=None) -> \
        Tuple[Tensor, ...]:
    """
    Split a matrix into a 2D list of blocks.

    :param matrix: Input matrix to split.
    :param dim: Dimension along which to split.
    :param n_blocks: Number of blocks to split into.
    :param split_size: Optional size of each block.
    :return: 2D list of blocks.
    """
    if split_size is None:
        return torch.split(matrix, matrix.shape[dim] // n_blocks, dim=dim)
    else:
        if isinstance(split_size, int):
            if matrix.shape[dim] % split_size != 0:
                raise ValueError("Matrix size must be divisible by split size.")
        elif isinstance(split_size, list):
            if sum(split_size) != matrix.shape[dim]:
                raise ValueError("Sum of split sizes must equal the matrix size.")
        return torch.split(matrix, split_size, dim=dim)


def ravel_multi_index(indices, shape):
    """
    Converts multi-dimensional indices (slice or tensor/list) into flat indices.

    Args:
        indices (list): List of slice objects, torch.Tensor, or lists representing multi-dimensional indices.
        shape (tuple): Shape of the multi-dimensional array.

    Returns:
        torch.Tensor: 1D indices.
    """
    grid = []
    for dim, idx in enumerate(indices):
        if isinstance(idx, slice):
            # Handle slice
            start, stop, step = idx.start or 0, idx.stop or shape[dim], idx.step or 1
            grid.append(torch.arange(start, stop, step))
        elif isinstance(idx, (torch.Tensor, list)):
            # Handle tensor or list
            grid.append(torch.tensor(idx))
        else:
            raise ValueError(f"Unsupported index type: {type(idx)} at dimension {dim}")

    # Use meshgrid to create multi-dimensional grids
    grids = torch.meshgrid(*grid, indexing='ij')

    # Compute flattened indices
    strides = torch.tensor(shape[1:]).flip(0).cumprod(0).flip(0)
    strides = torch.cat((strides, torch.tensor([1])))
    flat_index = sum(g.flatten() * stride for g, stride in zip(grids, strides))

    return flat_index
