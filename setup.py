from setuptools import setup, find_packages

setup(
    name='AlignCell',
    version='0.1.0',
    author="Zihuan Du",
    packages=find_packages(),  # 自动发现所有包，包括 performer_pytorch
    install_requires=[
        'torch',
    ],
)
