#!/bin/sh

if [ ! -f test.csv ]; then
    kaggle competitions download -c digit-recognizer -f test.csv \
        && unzip test.csv.zip
fi

cargo r -r -- kaggle model.bin > submission.csv \
    && kaggle competitions submit -c digit-recognizer -f submission.csv -m \
    "Multilayer perceptron 784 -> 196 -> 196 -> 64 -> 10 with LeakyReLU and softmax"
