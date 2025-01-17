# Digit recognition NN
This is an interactive neural network that runs on your browser through WebAssembly. It uses fully
connected feed forward neural networks with leaky ReLU and softmax to identify hand-written digits.
Training data are from the MNIST dataset.

## Performance
It predicts 98.15% correct on the MNIST testing dataset. In 0.746 seconds on a Tensor G4, including
reading the model and dataset.
