
# Songs Language Model

The present folder contains a pre-trained GRU language model built on word tokens from songs from ABBA, Beach Boys, U2 and Bob Dylan. The model is conditional on the author, via the use of writer codes that were jointly trained with the word embeddings. Parameters used for this model can be consulted in Hyperparameters.json.

## Generating from the model

To generate from this model, run the following code from the main directory of this repository
```
python generate.py --experiment_folder songs
```
Arguments can be added as described [before](../)
