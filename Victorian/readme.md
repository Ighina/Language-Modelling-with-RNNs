# Victorian Language Model
The present folder contains a pre-trained GRU language model built on characters from novels from from Charles Dickens and Oscar Wilde adapted from [project Gutenberg](http://www.gutenberg.org/). The model consists in 3 recurrent layers with the hidden state for each being initialised through the use of an author embedding. Parameters used for this model can be consulted in Hyperparameters.json.

# Generating from the model
To generate from this model, run the following code from the main directory of this repository
```
python generate.py --experiment_folder Victorian
```
Arguments can be added as described before
