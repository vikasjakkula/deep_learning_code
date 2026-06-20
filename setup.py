from setuptools import setup, find_packages

setup(
    name="nnscratch",
    version="0.1.0",
    description="Neural Network from Scratch with Numba CUDA acceleration "
                "(dict-of-weights architecture, RTX 3050 target)",
    author="vikasjakkula",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=["numpy>=1.23"],
    extras_require={
        "gpu": ["numba>=0.59"],
        "test": ["pytest>=7.0"],
    },
    include_package_data=True,
)
