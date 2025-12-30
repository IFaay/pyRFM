from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).parent

setup(
    name="pyrfm",
    version="0.3.0",
    description="Random Feature Method (RFM) tools in Python",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Yifei Sun",
    author_email="yfsun99@stu.suda.edu.cn",
    url="https://ifaay.github.io",
    packages=find_packages(exclude=["tests*", "docs*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0,<2.9",
        "numpy>=1.23",  # 建议加下限以确保与 torch 兼容
        "pandas>=1.5",
        "matplotlib>=3.5",
        "scipy>=1.9",
        "spdlog>=2.0.0"  # 若有版本要求可再加上限
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
