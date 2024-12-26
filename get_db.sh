#!/bin/sh

mkdir db
cd db
wget https://raw.githubusercontent.com/knamdar/data/master/MNIST/raw/train-images-idx3-ubyte.gz
wget https://raw.githubusercontent.com/knamdar/data/master/MNIST/raw/train-labels-idx1-ubyte.gz
wget https://raw.githubusercontent.com/knamdar/data/master/MNIST/raw/t10k-images-idx3-ubyte.gz
wget https://raw.githubusercontent.com/knamdar/data/master/MNIST/raw/t10k-labels-idx1-ubyte.gz
gzip -d *.gz
