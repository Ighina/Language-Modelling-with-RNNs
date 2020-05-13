# Surnames Language Model
A classical example of conditional language generation with surnames from different nationalities. The concept of such an experiment is inspired and can be found [here](https://github.com/joosthub/PyTorchNLPBook/tree/master/chapters/chapter_7). The data are reproduced from a similar [tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html) available among the official PyTorch tutorials. The model itself consists of an LSTM with two recurrent layers and writer codes (even though here it would be more appropriate to call them nationality codes) learned along the characters' embeddings. More details about the parameters can be found in Hyperparameters.json file.

# Generate from the Model
To generate from the model, the following code can be run:
```
python generate.py --experiment_folder Surnames
```
With available options for the generate.py program are described in more details in the main page of this repository.
