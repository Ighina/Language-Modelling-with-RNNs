#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 13:07:24 2020

@author: Iacopo
"""

from __future__ import unicode_literals, print_function, division
from io import open

from tqdm import tqdm
from collections import OrderedDict
import numpy as np
import os
import pickle

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from RNN import *
from Preprocess_function import *
import argparse



parser = argparse.ArgumentParser(
        description = 'Toolkit for natural language processing with gated recurrent neural networks')

parser.add_argument('--experiment_folder', '-folder', default='experiment', type=str,
                    help='Folder storing results and containing data for the current experiment.')

parser.add_argument('--data_dir', '-dir', default='data', type=str, 
                    help='Directory for training data.')

parser.add_argument('--embed_size', '-es', default=32, type=int, 
                    help='Embeddings size')

parser.add_argument('--hidden_size', '-hs', default=32, type=int, 
                    help='Number of hidden units per layer')

parser.add_argument('--num_layers', '-nl', default=1, type=int, 
                    help='Number of layers')

parser.add_argument('--learning_rate', '-lr', default=0.001, type=float, 
                    help='Learning rate')

parser.add_argument('--batch_size', '-bs', default=10, type=int, 
                    help='Mini batch size')

parser.add_argument('--num_epochs', '-ne', default=100, type=int, 
                    help='Number of training epochs')

parser.add_argument('--shuffle', '-sh', action='store_true', default=False, 
                    help='If included, training data are shuffled before input\
                    to the system.')

parser.add_argument('--bidirectional', '-bi', action='store_true', default=False, 
                    help='Create a bidirectional network? Default false.')

parser.add_argument('--LSTM', action='store_true', default=False, 
                    help='Use LSTM? Default is false, therefore GRU is used: if\
                    the parameter is set to true LSTM will be used (currently LSTM\
                    do not support manual initialisation of hidden state)')

parser.add_argument('--use_development', '-devel', action='store_true',
                    default=False, 
                    help='Use development set in training? Default is false and\
                    the early stop mechanism will look at development set to decide if to stop training or not')

parser.add_argument('--validation_percentage', '-vp', default=10, type=int, 
                    help='Percentage of available data to be used as development\
                    set.')

parser.add_argument('--validation_file', '-vf', default=False, type=str,
                    help='Option to include the validation set as a separate file\
                    currently there is no option to limit the number of training\
                    or validation data to be used and the tokenizer will return\
                    all available data in the files: if a subset of such data\
                    needs to be used is advised to create new files with the required\
                    subsets before input them.')

parser.add_argument('--early_stop', '-stop', default=5, type=int, 
                    help='Number of bad epochs after which to stop training (early stop)')

parser.add_argument('--conditional_model', '-cond', action='store_true', default=False, 
                    help='If included, a conditional language model is generated,\
                    whereas the additional condition is placed on the file name or\
                    if a csv is used as source data format on the second column of the csv')

parser.add_argument('--writer_codes', '-wc', action='store_true',
                    default=False, 
                    help='If a conditional model is chosen, this option specifies\
                    to concatenate the embedding for the current conditional value\
                    to each source token embedding at each time step')

parser.add_argument('--writers_embeddings', '-we', default=32, type=int, 
                    help='Specify the embedding size for the writer codes if they\
                    are included in the model.')

parser.add_argument('--initialise_hidden', '-ih', action='store_true',
                    default=False, 
                    help='If a conditional model is chosen, this option specifies\
                    to initialise the first hidden state with an embedding for the\
                    current conditional value. Can be used as an alternative to\
                    writer codes (suggested) or in addition to it.')

parser.add_argument('--remove_hapaxes', '-rh', action = 'store_true', 
                    default=False, help='Boolean: substitute hapaxes with unknown tokens?')

parser.add_argument('--removal_threshold', '-rt', default=1, type=int, 
                    help='Threshold of words frequencies under which substitute them with unknown token')

parser.add_argument('--dropout_in', '-di', default=0.0, type=float, 
                    help='Apply dropout to embeddings layer: the default is 0, that practically disactivate dropout\
                    the value needs to be between 0 and 1')

parser.add_argument('--dropout_out', '-do', default=0.0, type=float, 
                    help='Apply dropout to hidden layer(s): the default is 0, that practically disactivate dropout\
                    the value needs to be between 0 and 1')

parser.add_argument('--clip', '-cl', default=4.0, type=float,
                    help='Define the value over which to clip the gradient for regolaring it.')

parser.add_argument('--tokenize_lines', '-tkl', action='store_true',
                    default=False, 
                    help='If set to true, the training sentence will be tokenized\
                    as single lines in the source file. Default is false and a tokenizer\
                    that consdier logical sentences (separated by punctuation marks)\
                    spanning more or less than single lines')

parser.add_argument('--tokenize_characters', '-tkc', action='store_true',
                    default=False, 
                    help='If included, the model will be built over character sequences\
                    whereas the source file are assumed to include a single word\
                    per line')

parser.add_argument('--separated_by_tab', '-sbt', action='store_true',
                    default=False, 
                    help='An alternative tokenization procedure in case the source\
                    file has tabs separating the contents of each line, where the\
                    first part of the line is assumed to be the sentence and the\
                    second is assumed to be the conditional value.')

parser.add_argument('--csv_source', '-csv', action='store_true',
                    default=False,
                    help='If included the source file is assumed to be a csv file\
                    having in the first column the preprocessed lines and, eventually,\
                    in the second column the additional conditioning values')

parser.add_argument('--use_NLTK', '-nltk', action = 'store_true', default=False,
                    help='If set to true, a single text document for each book/collection\
                    is assumed and nltk is used to tokenize sentences for each file content.\
                    In practice this can be set to true if the text is in English\
                    and has the above specified format, as the current nltk tokenizer supports just english.')

parser.add_argument('--preserve_new_lines', action='store_true', default=False,
                    help='If included, the preprocessor does not delete the special \n character for new line.')

parser.add_argument('--log_generation', '-log_g', action='store_true',
                    default=False,
                    help='If included, the terminal will output a generated sentence\
                    from the model every n batch during training, where n can be specified. Coud\
                    be useful to check that the model is learning as expected.')

parser.add_argument('--generate_log_each_n', '-gen_num', default=125, type=int,
                    help='Specify the n for generating from the model during training\
                    as explained above')

parser.add_argument('--log_stats', '-log_s', action='store_true', default=False,
                    help='If inclded, the terminal will output statistics and a\
                    completion bar for each batch during training. It is advisable\
                    not to use it in conjunction with --log_generation.')

parser.add_argument('--save_model', '-save', action='store_true', default=False,
                    help='If included, the learned model will be saved in the saved model directory under the name model.pt')

parser.add_argument('--save_dictionaries', '-save_d', action='store_false',
                    default= True, help='Save the dictionaries of the trained\
                    model, containing the vocabulary with words indices and\
                    the dictionary with writer indices (if included).')

parser.add_argument('--use_gpu', '-gpu', action='store_true', default=False,
                    help='If true, uses gpu as standard device (if available).')

args= parser.parse_args()

folder = args.experiment_folder
data_dir = os.path.join(folder, args.data_dir)
EMBEDDING_DIM = args.embed_size
HIDDEN_DIM = args.hidden_size
NUM_LAYERS = args.num_layers
learn = args.learning_rate
remove_hp = args.remove_hapaxes
threshold_remove = args.removal_threshold
drop = args.dropout_in
drop_out = args.dropout_out
tok_verse = args.tokenize_lines
epochs = args.num_epochs
shuffle = args.shuffle
bidirectional = args.bidirectional
use_lstm = args.LSTM
use_validation = args.use_development
include_writers = args.conditional_model
writer_codes = args.writer_codes
writer_embeds = args.writers_embeddings
writers_as_first_hidden = args.initialise_hidden
char = args.tokenize_characters
valid_perc = args.validation_percentage
validation_file = args.validation_file
early_stop = args.early_stop
log_generate = args.log_generation
generate_each_n_batches = args.generate_log_each_n
log_stats = args.log_stats
new_line = args.preserve_new_lines


if validation_file:
    preprocessor = Preprocess(data=data_dir)
    files_train = input("Enter training file(s) name(s), if more than one separated by white space:  ").split()
    files_valid = input("Enter validation file(s) name(s), if more than one separated by white space:  ").split()
    voc_2_index, word_counts, all_lines_gr, index_2_voc, writer =\
    preprocessor.preprocess_files(files_train, separated_by_tabs=True, 
                                  remove_hapaxes=remove_hp,
                                  remove_threshold=threshold_remove,
                                  include_writers=include_writers,
                                  use_NLTK=args.use_NLTK,
                                  preserve_new_line=new_line)
    if not remove_hp:
        voc_2_index['UKN']=len(voc_2_index)
        index_2_voc[voc_2_index['UKN']] = "UKN"
    sents = []
    for index, sent in enumerate(all_lines_gr.values()):
        sents.append(sent)
            
    
    sents_val = []       
    _,_,validation_lines,_ = preprocessor.preprocess_files(files_valid, separated_by_tabs=True)
    
    for index, sent in enumerate(validation_lines.values()):
        sents_val.append(sent)
            
    training_data = [(sentence_to_train(line, voc_2_index),sentence_to_target(line, voc_2_index)) for line in sents]
    validation_data = [(sentence_to_train(line, voc_2_index),sentence_to_target(line, voc_2_index)) for line in sents_val]
    valid_2_batch = LMDataset(validation_data, voc_2_index)
    dataset_valid = torch.utils.data.DataLoader(dataset=valid_2_batch, batch_size = BATCH_SIZE, collate_fn = training_2_batch.collater)
elif char:
    Preprocessor = Preprocess(data=data_dir)
    if args.csv_source:
        for file in Preprocessor.get_files():
            assert os.path.splitext(file)[1][1:]=='csv', 'You specified csv file(s) as input, but files with different extentions are present in the data directory: please change extension/move non-csv files'
    voc_2_index, word_counts, all_lines_gr, index_2_voc, writer =\
    Preprocessor.preprocess_files(by_character=True,
                                  remove_hapaxes=remove_hp,
                                  remove_threshold=threshold_remove,
                                  include_writers=include_writers,
                                  use_NLTK=args.use_NLTK,
                                  preserve_new_line=new_line)
    if include_writers:
        training_data = []
        for line in all_lines_gr.values():
            writer_tensor = torch.tensor([writer[line[1]] for i in range(len(line[0])+1)], dtype=torch.long)
            training_data.append((sentence_to_train(line[0], voc_2_index, by_character=True),sentence_to_target(line[0], voc_2_index, by_character=True),writer_tensor))
    else:
        training_data = [(sentence_to_train(line[0], voc_2_index, by_character=True),sentence_to_target(line[0], voc_2_index, by_character=0),sentence_to_target(line[0],voc_2_index)) for line in all_lines_gr.values()]
else:
    Preprocessor = Preprocess(data=data_dir)
    voc_2_index, word_counts, all_lines_gr, index_2_voc, writer =\
    Preprocessor.preprocess_files(remove_hapaxes=remove_hp,
                                  by_stanzas=tok_verse,
                                  remove_threshold=threshold_remove,
                                  include_writers=include_writers,
                                  use_NLTK=args.use_NLTK,
                                  preserve_new_line=new_line)
    if include_writers:
        training_data = []
        for line in all_lines_gr.values():
            writer_tensor = torch.tensor([writer[line[1]] for i in line[0].split()[:-1]], dtype=torch.long)
            training_data.append((sentence_to_train(line[0], voc_2_index),sentence_to_target(line[0], voc_2_index),writer_tensor))
    else:
        training_data = [(sentence_to_train(line[0], voc_2_index),sentence_to_target(line[0], voc_2_index),sentence_to_target(line[0],voc_2_index)) for line in all_lines_gr.values()]


os.chdir('../')

if args.save_dictionaries:
    with open('saved_dictionaries/voc_2_index.pkl','wb') as f:
        pickle.dump(voc_2_index,f)
    with open('saved_dictionaries/index_2_voc.pkl', 'wb') as f:
        pickle.dump(index_2_voc, f)
    with open('saved_dictionaries/word_counts.pkl', 'wb') as f:
        pickle.dump(word_counts, f)
    with open('saved_dictionaries/writers.pkl', 'wb') as f:
        pickle.dump(writer, f)
    
print("Total sentences: {}\n Total number of tokens: {}".format(len(all_lines_gr),sum(word_counts.values())))

BATCH_SIZE = args.batch_size

model_new = RNN(EMBEDDING_DIM, HIDDEN_DIM, len(voc_2_index), NUM_LAYERS, 
                    dropout_in=drop, dropout_out=drop_out,
                    add_writer=writer_codes, 
                    writer_number=len(writer), 
                    writer_embed_size=writer_embeds, 
                    add_writer_as_hidden=writers_as_first_hidden, 
                    LSTM=use_lstm, bidirectional=bidirectional)


criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
optimizer2 = torch.optim.Adam(model_new.parameters(), lr=learn)

training_2_batch = LMDataset(training_data, voc_2_index)
dataset = torch.utils.data.DataLoader(dataset=training_2_batch,
                                      batch_size = BATCH_SIZE,
                                      collate_fn = training_2_batch.collater,
                                      shuffle=shuffle)
if args.use_gpu:
    dataset = DeviceDataLoader(dataset, device)

losses = []
train_losses = []
embeddings = {}
bad_epochs = 0
best_validate = float('inf')

for epoch in range(epochs):
    model_new.train()
    acc_loss_train = 0
    acc_loss_valid = 0
    print('Starting epoch {}'.format(str(epoch)))
    counter = 0
    stats = OrderedDict()
    stats['loss'] = 0
    stats['lr'] = 0
    stats['num_tokens'] = 0
    stats['batch_size'] = 0
    stats['grad_norm'] = 0
    stats['clip'] = 0
    progress_bar = tqdm(dataset, desc='| Epoch {:03d}'.format(epoch), leave=False, disable=not(log_stats))
    validation = []
    train_count = 0

        # Iterate over the training set
    for counter, sentence in enumerate(progress_bar):
        
        if counter%valid_perc==0:
            if validation_file:
                pass
            elif use_validation:
                validation.append(sentence)
                continue
            else:
                pass
        
        train_count+=1

        if len(sentence['src_tokens']) == 0:
                continue
        
        h_n,tag_scores = model_new(sentence)
        
        
        loss = criterion(tag_scores.view(-1,len(voc_2_index)), sentence['tgt_tokens'].view(-1))/len(sentence['src_lengths'])
        
        
        if log_generate:
            
            if counter % generate_each_n_batches == 0:
            
                sentence_indexes = sentence['src_tokens'].view(-1).tolist()
                
                greedy_predictions = [torch.argmax(step).tolist() for step in tag_scores[0]]
                
                print('Original sentence: {}'.format(' '.join([index_2_voc[idx] for idx in sentence['src_tokens'][0].tolist()[1:] if idx!=0])))
                print('Predicted sentence: {}'.format(' '.join([index_2_voc[idx] for idx in greedy_predictions[:-1]])))
                print(loss.item()*len(sentence['src_lengths'])/sentence['num_tokens'])
        
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model_new.parameters(), args.clip)
        optimizer2.step()
        optimizer2.zero_grad()
        
        total_loss, num_tokens, batch_size = loss.item(), sentence['num_tokens'], len(sentence['src_tokens'])
        stats['loss'] += total_loss * len(sentence['src_lengths']) / sentence['num_tokens']
        stats['lr'] += optimizer2.param_groups[0]['lr']
        stats['num_tokens'] += num_tokens / len(sentence['src_tokens'])
        stats['batch_size'] += batch_size
        stats['grad_norm'] += grad_norm
        stats['clip'] += 1 if grad_norm > args.clip else 0
        progress_bar.set_postfix({key: '{:.4g}'.format(value / (counter + 1)) for key, value in stats.items()},
                                         refresh=True)

    model_new.eval()
    stats['valid_loss'] = 0
    stats['num_tokens_val'] = 0
    stats['batch_size_val'] = 0
    if use_validation:
        if validation_file:
            for sentence in dataset_valid:
                validation.append(sentence)
            
        for i, sample in enumerate(validation):
            if len(sample) == 0:
                continue
            with torch.no_grad():
                # Compute loss
                _,output = model_new(sample)
                loss = criterion(output.view(-1, output.size(-1)), sample['tgt_tokens'].view(-1))
            # Update tracked statistics
            stats['valid_loss'] += loss.item()
            stats['num_tokens_val'] += sample['num_tokens']
            stats['batch_size_val'] += len(sample['src_tokens'])
    
        # Calculate validation perplexity
        stats['valid_loss'] = stats['valid_loss'] / stats['num_tokens_val']
        perplexity = np.exp(stats['valid_loss'])
        stats['num_tokens_val'] = stats['num_tokens_val'] / stats['batch_size_val']
        stats['loss']=stats['loss']/train_count
        losses.append(stats['valid_loss'])
        print('average loss for epoch {}: {}'.format(str(epoch+1),str(stats['valid_loss'])))
    else:
        stats['loss']=stats['loss']/counter
        perplexity = np.exp(stats['loss'])
    print('Epoch {:03d}: {}'.format(epoch, ' | '.join(key + ' {:.3g}'.format(value) for key, value in stats.items())) +
        ' | valid_perplexity {:.3g}'.format(perplexity))
    
    if perplexity < best_validate:
            best_validate = perplexity
            bad_epochs = 0
            state_dict = model_new.state_dict()
    else:
            bad_epochs += 1
    if bad_epochs >= early_stop:
            print('No validation set improvements observed for {:d} epochs. Early stop!'.format(early_stop))
            break
    train_losses.append(stats['loss'])
    
if args.save_model:
    torch.save(state_dict, 'saved_models/model.pt')