# Victorian Language Model
The present folder contains a pre-trained GRU language model built on characters from novels from from Charles Dickens and Oscar Wilde adapted from [project Gutenberg](http://www.gutenberg.org/). The model consists in 3 recurrent layers with the hidden state for each being initialised through the use of an author embedding. Parameters used for this model can be consulted in Hyperparameters.json.

# Data Format
By looking into the data folder it can be noticed that the various text files are named as {AuthorName}{number}.txt 
Using this format allows the Preprocess_function to extract just the author name from the file name and, therefore, to create an appropriate number of author's embeddings (2 in this case).

# Generating from the model
To generate from this model, run the following code from the main directory of this repository
```
python generate.py --experiment_folder Victorian
```
Arguments can be added as described before
