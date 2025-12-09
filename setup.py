from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="erd-spectrum-decomposer",
    version="1.0.0",
    author="Praise James",
    author_email="praisejames011@gmail.com",
    description="Elegant Recursive Discovery Engine for autonomous spectral decomposition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/praisejamesx/erd-spectrum-decomposer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
    ],
    keywords="spectroscopy, decomposition, AI, Raman, autonomous, scientific discovery",
)