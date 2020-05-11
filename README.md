# Language-Modelling-with-RNNs
A simple series of programs to train gated recurrent neural networks with PyTorch and generate text based on them.

# The models
Currently supported are LSTM and GRU.

# Conditional Language Modelling
The program allows to train an RNN conditioning on the authors of the input text. 2 ways of doing this are available: by including "writer codes" or by initialising the hidden state with an author embedding. The writer codes option train together with the words/character embedding a writer code at each time step, the dimensionality of which can be specified as described below. Both methods can be used, even though the use of one or the other is suggested.

# Running the Program
## Setting up the workspace and download required libraries

Download this repository, unzip it and move the working directory into it.

To download the required libraries, either pip can be used directly to install them on the python installation of the current environment, or a virtual environment can be created with conda. The current installation assumes no GPU available by default: to install pytorch with GPU, refer to [this page](https://pytorch.org/get-started/locally/).

In both cases, open your command prompt (anaconda prompt is preferred) and do one of the following.

Installing required libraries with pip:
'''
pip -install -r requirements.txt
'''

Creating a virtual-environment where to install required libraries with conda:
'''
conda create -n rnn --file ./requirements.txt -c pytorch
'''

After which the environment can be activated with:
'''
conda activate rnn
'''
And deactivated with:
'''
conda deactivate rnn
'''

## Setting up a workspace directory

Currently, the program works by gathering from and saving the data to user-defined folders having a pre-defined structure.
To create a new project, you will need to create such a new folder with a name of your choice and to do so, the Prepare_workspace.py program can be run, specifying the new folder name:
'''
python Prepare_workspace.py -name <your_folder_name>
'''

At this point a folder with the specified name should be created. You can copy your data to the data sub-directory in the newly created folder, as well as change the parameters of Hyperparameters.json inside the folder as required. Such a file is copied from the working directory (previously set to be this repository in the location you downloaded it on your pc) and defines the specific parameters for each separate experiment.

## Training a model

To train a model, the Hyperparameters.json file that has been copied into the newly created data folder is parsed by the run.py program, that in turn calls the Train.py program. The Hyperparameters.json inside the newly created folder specifies a number of options, such as number of hidden units/embeddings dimensions, whether to use a conditional language model and, in case, if to use writer codes or initialise the hidden layer (see above) and, crucially, wheter to model characters or entire words, how tokenization is done (by line or by logical sentence) and the source data format. To see all the available options and their use, the first lines of Train.py can be consulted (the ones defining the parser arguments) either by opening the file itself or by running the following command:
'''
python Train.py -h
'''
Be careful to use exactly the same formatting when changing the Hyperparameters.json file (e.g. write "TRUE" for including an option and "FALSE" to exclude it).
Once the Hyperparameters.json has been modified as appropriate (remember to check that the '--save_model' option is set to "TRUE" in order to store the trained model), run the following command to train the model:
'''
python run.py --experiment_folder <your_folder_name>
'''
According to the model size and if a GPU is used or not, the model could take many hours to train. By keeping the terminal open, the training process should go on until the number of desired epochs is reached or no improvement was observed.

## Generate from the trained model

Once the model is trained (and if the "--save_model" option was set to "TRUE"), it can be used to generate sample sentences as follow:
'''
python generate.py --experiment_folder <your_folder_name>
'''
Whereas additional options include:
-sample (default 3): the number of samples to generate.
-size (default 10): the maximum size of each sample (in words if the model was trained on words or in characters otherwise).
-cont (default False): a boolean. If included, the generation process won't stop when an end of sentence is encountered but it keeps generating until the maximum size is reached.
-save (default False): a boolean. If included, the output is saved to "output.txt" in the outputs sub-directory.

An example use could be as following:
'''
python generate.py -sample 2 -size 20 -cont -save
'''

## Generate from pre-trained model

In this repository, three pre-trained models are available in the songs, victorian and names folder respectively (more info in each folder). Such models can be used to generate sentences without the need to train a new one as described in each folder.

# Remarks

The present codes have been variously adapted from facebook's [fairseq repository](https://github.com/pytorch/fairseq), from [pytorch tutorials](https://pytorch.org/tutorials/) and from Delip Rao and Brain McMahan's [Natural Language Processing with PyTorch](https://github.com/joosthub/PyTorchNLPBook). Other pieces of codes might have been inspired by online blogs and/or tutorials: please, if I am missing to mention any one feel free to contact me.



