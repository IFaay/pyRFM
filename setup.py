# -*- coding: utf-8 -*-
"""
Created on 2024/12/15

@author: Yifei Sun
"""
from setuptools import setup, find_packages

setup(
    name="pyrfm",
    version="0.1.0",
    description="A Python package for ... (add description)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/pyrfm",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "torch"
        "sympy"
        # 添加其他依赖包
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
