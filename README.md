# Language-Modelling-with-RNNs
A simple series of programs to train gated recurrent neural networks with PyTorch and generate text based on them.

# The models
Currently supported are LSTM and GRU.

# Conditional Language Modelling
The program allows to train an RNN conditioning on the authors of the input text. 2 ways of doing this are available: by including "writer codes" or by initialising the hidden state with an author embedding. The writer codes option train together with the words/character embedding a writer code at each time step, the dimensionality of which can be specified as described below. Both methods can be used, even though the use of one or the other is suggested.

# Running the Program
## Setting up the workspace and download required libraries

To download the required libraries ... 
