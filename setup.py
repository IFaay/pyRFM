# -*- coding: utf-8 -*-
"""
Created on 2024/12/15

@author: Yifei Sun
"""
from setuptools import setup, find_packages

setup(
    name="pyrfm",
    version="0.1.3",
    description="A Python package for Random Feature Method (RFM)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Yifei Sun",
    author_email="yfsun99@stu.suda.edu.cn",
    url="https://ifaay.github.io",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "spdlog"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
