from setuptools import setup, find_packages
import blackbox
import os

def readme():
    with open('README.rst') as f:
        return f.read()

setup(
    name='iblackbox',
    version='0.0.1',
    description='Artificial Neural network potentials',
    long_description=readme(),
    classifiers=[
        "Development Status :: 1 - Planning",
        "Environment :: Console",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords=['Neural network potentials', 'Machine Learning'],
    author='Pu Du',
    author_email='pudugg@gmail.com',
    maintainer='Pu Du',
    maintainer_email='pudugg@gmail.com',
    url='https://github.com/ipudu/blackbox',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
)