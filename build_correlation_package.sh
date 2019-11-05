#!/bin/sh
git clone https://github.com/NVIDIA/flownet2-pytorch.git
cd flownet2-pytorch
git checkout 71034046166735a79a5b82df78de72d806e82842
cd ../
mv flownet2-pytorch/networks/correlation_package ./
cd ./correlation_package/
python setup.py build



