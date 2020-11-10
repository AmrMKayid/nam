from setuptools import find_packages
from setuptools import setup

with open("README.md", "r") as f:
  long_description = f.read()

setup(
    name='nam',
    version='0.0.1',
    description="Neural Additive Models (Google Research)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Amr Kayid",
    url="https://github.com/AmrMKayid/nam",
    packages=find_packages(),
    install_requires=[
        "torch==1.7.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
