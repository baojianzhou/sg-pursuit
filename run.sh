#!/usr/bin/env bash
# this file is for uploading.
rm -rf dist
rm -rf build
rm -rf sparse_learning.egg-info
python setup.py sdist bdist_wheel
twine upload dist/*.tar.gz
rm -rf dist
rm -rf build
rm -rf sparse_learning.egg-info