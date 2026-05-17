from os import name
import setuptools

NAME = "torch_fftconv"
VERSION = "1.0.0"
MAINTAINER = "Chin-Yun Yu"
EMAIL = "chin-yun.yu@qmul.ac.uk"


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name=NAME,
    version=VERSION,
    author=MAINTAINER,
    author_email=EMAIL,
    description="Implementation of 1D, 2D, and 3D FFT convolutions in PyTorch. Much faster than direct convolutions for large kernel sizes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yoyololicon/fft-conv-pytorch",
    packages=["torch_fftconv"],
    install_requires=["torch>=2.0"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
