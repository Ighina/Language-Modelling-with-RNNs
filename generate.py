# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 20:50:04 2020

@author: Iacopo
"""

from __future__ import unicode_literals, print_function, division
import os
import pickle
import json
import re

import torch
from RNN import RNN
import torch.nn.functional as F
import argparse
import sys

class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

parser = MyParser(
        description = 'Run training with hyperparameters defined in the relative json file')

parser.add_argument('--experiment_folder', '-folder', default='experiment', type=str,
                    help='Folder storing results and containing data for the current experiment.')

parser.add_argument('--hyper_file', '-hyper', default='Hyperparameters.json', type=str, 
                    help='Configuration file defining the hyperparameters and options to be used in training.')

parser.add_argument('--model_path', '-model', default='saved_models', type=str,
                    help='Path to the trained model.')

parser.add_argument('--dictionary_path', '-dict', default='saved_dictionaries', type=str,
                    help='Path to vocabularies and writer dictionaries.')

parser.add_argument('--number_samples', '-sample', default=3, type=int,
                    help='Number of samples to be generated from the unconditional model or\
                    for each conditional value from the conditional model.')

parser.add_argument('--sample_size', '-size', default=20, type=int,
                    help='Maximun length of the generated samples (whereas\
                    the sample can be shorter if an end of sequence is generated).')

parser.add_argument('--temperature', '-temp', default=0.7, type=float,
                    help='Determine how peaked the distribution from which to sample\
                    wil be (see sampling functions below for more info).')

parser.add_argument('--continuous_gen', '-cont', action='store_true', default=False,
                    help='If included, the final hidden state from the previously generated sentence is used to generate the following sentence.')

parser.add_argument('--save_output', '-save', action='store_true', default=False,
                    help='Store the output of the generation.')

parser.add_argument('--output_directory', '-out', default='outputs', type=str,
                    help='Directory in which to store the generated texts.')

args = parser.parse_args()

os.chdir(args.experiment_folder)

def get_model_parameters(hyper_file=args.hyper_file):
    with open(hyper_file, encoding='utf-8') as f:
        temp = f.read()
    hyper = json.loads(temp)
    for key, value in hyper.items():
        if value=='TRUE':
            hyper[key]=True
        elif value=='FALSE':
            hyper[key]=False
    return hyper

def load_dictionaries(path):
    assert os.path.exists(path), 'directory for dictionaries {} could not be found'.format(path)
    with open(os.path.join(path,'voc_2_index.pkl'),'rb') as f:
        voc_2_index = pickle.load(f)
    with open(os.path.join(path,'index_2_voc.pkl'),'rb') as f:
        index_2_voc = pickle.load(f)
    with open(os.path.join(path,'writers.pkl'),'rb') as f:
        writer = pickle.load(f)
    return voc_2_index, index_2_voc, writer

def load_model(path, hyper, inference=True, dictionary_path=args.dictionary_path, LSTM=False):
    assert os.path.exists(path), 'directory for model {} could not be found'.format(path)
    voc_2_index, _ , writer = load_dictionaries(dictionary_path)
    model = RNN(hyper['--embed_size'], hyper['--hidden_size'], len(voc_2_index), hyper['--num_layers'], 
                    add_writer=hyper['--writer_codes'], 
                    writer_number=len(writer), 
                    writer_embed_size=hyper['--writers_embeddings'], 
                    add_writer_as_hidden=hyper['--initialise_hidden'], 
                    LSTM=LSTM)
#    lod = torch.load(os.path.join(path,'model.pt'))
    model.load_state_dict(torch.load(os.path.join(path,'model.pt')))
    if inference:
        model.eval()
    return model

def unconditional_sample_from_model(model, voc_2_index, num_samples=1, sample_size=20, 
                      temperature=1.0):
    """Sample a sequence of indices from the model
    
    Args:
        model (GenerationModel): the trained model
        voc_2_index (Dictionary): word to indices dictionary
        sample_size (int): the max length of the samples
        temperature (float): accentuates or flattens 
            the distribution. 
            0.0 < temperature < 1.0 will make it peakier. 
            temperature > 1.0 will make it more uniform
    Returns:
        indices (torch.Tensor): the matrix of indices; 
        shape = (num_samples, sample_size)
    """
    begin_seq_index = [voc_2_index['<s>'] 
                       for _ in range(num_samples)]
    begin_seq_index = torch.tensor(begin_seq_index, 
                                   dtype=torch.int64).unsqueeze(dim=1)
    indices = [begin_seq_index]
    h_t = None
    for time_step in range(sample_size):
        x_t = indices[time_step]
        x_emb_t = model.embedding(x_t)
        rnn_out_t, h_t = model.rnn(x_emb_t, h_t)
        prediction_vector = model.output(rnn_out_t.squeeze(dim=1))
        probability_vector = F.softmax(prediction_vector / temperature, dim=1)
        indices.append(torch.multinomial(probability_vector, num_samples=1))
    indices = torch.stack(indices).squeeze().permute(1, 0)
    return indices

def conditional_sample_from_model(model, voc_2_index, writers, sample_size=20, 
                      temperature=1.0, init_hidden=False, writer_codes=False):
    """Sample a sequence of indices from the model
    
    Args:
        model (GenerationModel): the trained model
        voc_2_index (Dictionary): word to indices dictionary
        writers (list): a list of integers representing writers
        init_hidden (Boolean): True if first hidden layer embeds the writer.
        writer_codes (Boolean): True if input is augmented with writer codes.
        sample_size (int): the max length of the samples
        temperature (float): accentuates or flattens 
            the distribution. 
            0.0 < temperature < 1.0 will make it peakier. 
            temperature > 1.0 will make it more uniform
    Returns:
        indices (torch.Tensor): the matrix of indices; 
        shape = (num_samples, sample_size)
    """
    num_samples = len(writers)
    begin_seq_index = [voc_2_index['<s>'] 
                       for _ in range(num_samples)]
    
    begin_seq_index = torch.tensor(begin_seq_index, 
                                   dtype=torch.int64).unsqueeze(dim=1)
    writer_seq_index = [writers[0] for _ in range(num_samples)]
    writer_seq_index = torch.tensor(writer_seq_index, 
                                   dtype=torch.int64).unsqueeze(dim=1)
    writer_embeddings_indices = [writer_seq_index]
    indices = [begin_seq_index]
    writer_indices = torch.tensor(writers, dtype=torch.int64).unsqueeze(dim=0)
    if init_hidden:
        h_t = model.init_embedding(writer_indices)
        h_t = h_t.repeat(model.num_layers,1,1).view(-1,*h_t.shape[1:])
    else:
        h_t = None
    for time_step in range(sample_size):
        x_t = indices[time_step]
        x_emb_t = model.embedding(x_t)
        if writer_codes:
            writ = writer_embeddings_indices[time_step]
            writer_emb = model.writer_embedding(writ)
            x_emb_t = torch.cat((x_emb_t, writer_emb),2)
        rnn_out_t, h_t = model.rnn(x_emb_t, h_t)
        prediction_vector = model.output(rnn_out_t.squeeze(dim=1))
        probability_vector = F.softmax(prediction_vector / temperature, dim=1)
        indices.append(torch.multinomial(probability_vector, num_samples=1))
        writer_embeddings_indices.append(writer_seq_index)
    indices = torch.stack(indices).squeeze().permute(1, 0)
    return indices

def decode_samples(sampled_indices, voc_2_index, index_2_voc,by_character=False, 
                   continuous_generation=False):
    """Transform indices into the string form
    
    Args:
        sampled_indices (torch.Tensor): the inidces from `sample_from_model`
        voc_2_index (Dictionary): the word to index vocabulary
        index_2_voc (Dictionary): the index to word vocabulary
        by_character (Boolean): True if not generating words.
        continuous_generation (Boolean): If true, the end of sequence tag do not break generation, but starts a new line.
    """
    decoded = []
    vocab = voc_2_index
    
    for sample_index in range(sampled_indices.shape[0]):
        output = ""
        for time_step in range(sampled_indices.shape[1]):
            sample_item = sampled_indices[sample_index, time_step].item()
            if sample_item == vocab['<s>']:
                continue
            elif sample_item == vocab['<\s>'] and not continuous_generation:
                break
            elif sample_item == vocab['<\s>']:
                if output[-1]=='\n':
                    pass
                else:
                    output += '\n'
            elif by_character:
                output += index_2_voc[sample_item]
            else:
                output += ' '+ index_2_voc[sample_item]
        decoded.append(output)
    return decoded
    
def main():
    voc_2_index, index_2_voc, writer = load_dictionaries(args.dictionary_path)
    hyper = get_model_parameters(args.hyper_file)
    model = load_model(args.model_path, hyper=hyper, LSTM=hyper['--LSTM'])
    output = []
    try:
        if hyper['--conditional_model']:
            try:
                if hyper['--writer_codes']:
                    cod_writer = True
                else:
                    cod_writer = False
            except KeyError:
                assert hyper['--initialise_hidden'], 'To use conditional model generation at least one of writer codes or hidden state initialisation must have been included in training.'
            try:
                if hyper['--initialise_hidden']:
                    h_init = True
                else:
                    h_init = False
            except KeyError:
                pass
            for index, el in enumerate(writer):
                output.append(el)
                if index==0:
                    continue
        #        h_init = writer[el]
                print("Sampled for {}: ".format(el))
                sampled_indices = conditional_sample_from_model(model, voc_2_index,  
                                                    writers=[index] * args.number_samples, 
                                                    sample_size=args.sample_size,
                                                    temperature=args.temperature,
                                                    writer_codes=cod_writer,
                                                    init_hidden=h_init)
                for sample in decode_samples(sampled_indices, voc_2_index, index_2_voc, 
                                             hyper['--tokenize_characters'],
                                             continuous_generation=args.continuous_gen):
                    print("-  " + sample)
                    output.append(sample)
        else:
            samples = unconditional_sample_from_model(model, voc_2_index)
            output = decode_samples(samples, voc_2_index, index_2_voc, 
                                    hyper['--tokenize_characters'],
                                    continuous_generation=args.continuous_gen)
            print(output)
    except KeyError:
        samples = unconditional_sample_from_model(model, voc_2_index)
        output = decode_samples(samples, voc_2_index, index_2_voc,
                                hyper['--tokenize_characters'],
                                continuous_generation=args.continuous_gen)
        print(output)
    if args.save_output:
        with open(os.path.join(args.output_directory,'output.txt'),'w') as f:
            for element in output:
                f.write(element+'\n')
if __name__ == '__main__':
    main()