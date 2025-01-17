# Digit recognition NN
This is an interactive neural network that runs on your browser through WebAssembly. It uses fully
connected feed forward neural networks with leaky ReLU and softmax to identify hand-written digits.

Training and testing data are from the MNIST dataset. Training inputs are translated, rotated, and
noise are added.

## Performance
It predicts 98.64% correct on the MNIST testing dataset in 0.746 seconds on a Tensor G4, including
reading the model and dataset.
