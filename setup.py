#!/usr/bin/env python

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    
setup(
    name='question-answering-networks',
    version='0.0.1',
    description='Neural Networks for Question Answering',
    author='Muhammad Rifki Kurniawan',
    author_email='mrifkikurniawan17@gmail.com',
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url='',
    install_requires=requirements,
    packages=find_packages(),
)
