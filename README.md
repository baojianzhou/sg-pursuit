# sg-pursuit

# python setup.py build_ext --inplace
This package collects sparse learning related algorithms. 
In this first initial version, it contains the following projection code:

head projection
tail projection
sparse-k projection
sudo python3 -m pip install --user --upgrade setuptools wheel python3 setup.py sdist bdist_wheel

twine upload dist/*

error: Upload failed (400): Binary wheel ‘ *-cp36-cp36m-linux_x86_64.whl’ has an unsupported platform tag ‘linux_x86_64’.

You can workaround this by not uploading the wheel, just the source code and your users will compile it on their machine.