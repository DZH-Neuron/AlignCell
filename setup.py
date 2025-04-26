from setuptools import setup, find_packages

setup(
    name='AlignCell',
    version='0.1.0',
    author="Zihuan Du",
    packages=find_packages(), 
    install_requires=[
        'torch',
    ],
)
