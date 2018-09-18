# -*- coding: utf-8 -*-
"""
How to run it ? python setup.py build_ext --inplace
"""
import os
import numpy
from os import path
from setuptools import setup
from distutils.core import Extension

here = path.abspath(path.dirname(__file__))

src_files = ['c/main_wrapper.c', 'c/head_tail_proj.c', 'c/fast_pcst.c']
compile_args = ['-std=c11', '-lpython2.7', '-lm']
# calling the setup function
setup(
    # sparse_learning package.
    name='sparse_learning',
    # current version is 0.2.1
    version='0.2.3',
    # this is a wrapper of head and tail projection.
    description='A wrapper for sparse learning algorithms.',
    # a long description should be here.
    long_description='This package collects sparse learning algorithms.',
    # url of github projection.
    url='https://github.com/baojianzhou/sparse_learning.git',
    # number of authors.
    author='Baojian Zhou',
    # my email.
    author_email='bzhou6@albany.edu',
    include_dirs=[numpy.get_include()],
    license='MIT',
    packages=['sparse_learning'],
    classifiers=("Programming Language :: Python :: 2",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: POSIX :: Linux",),
    # specify requirements of your package here
    # will add openblas in later version.
    install_requires=['numpy'],
    headers=['c/head_tail_proj.h', 'c/fast_pcst.h'],
    # define the extension module
    ext_modules=[Extension('sparse_module',
                           sources=src_files,
                           language="C",
                           extra_compile_args=compile_args,
                           include_dirs=[numpy.get_include()])],
    keywords='sparse learning, structure sparsity, head/tail projection')