#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='PointNet-modern.pytorch',
    version='0.0.1',
    description='PointNet impplemented in pytorch with better readability',
    author='PointCloudYC',
    author_email='chao.yin@connect.ust.hk',
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url='https://github.com/PointCloudYC/PointNet-modern.pytorch',
    install_requires=['torch'],
    packages=find_packages(),
)