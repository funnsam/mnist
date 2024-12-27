#!/bin/sh

mkdir db
cd db
wget https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip
unzip gzip.zip
mv gzip/* .
gzip -d emnist-balanced-*.gz
rm *.gz
