# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 17:53:55 2020

@author: Iacopo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence as PACK
from torch.utils.data import Dataset

class RNN(nn.Module):
    def __init__(self, embed_size, hidden_size, voc_size,num_layers,
                 decode=False, bidirectional=False, dropout_in=False,
                 dropout_out=False,padding_idx=0, batch_first=True,
                 LSTM=True, add_writer=False,
                 writer_embed_size=32, writer_number=None, 
                 add_writer_as_hidden=False):
        super(RNN,self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.voc_size = voc_size
        self.num_layers = num_layers
        self.decode = decode
        self.bidirectional=bidirectional
        self.add_writer = add_writer
        self.writer_as_hidden = add_writer_as_hidden
        
        self.embedding = nn.Embedding(num_embeddings=voc_size, 
                                      embedding_dim=embed_size, 
                                      padding_idx=padding_idx)
        if add_writer:
            assert writer_number!=None, "Remember to add writer number (+1 for padding) if you want to include writers' embeddings!"
            self.writer_embedding = nn.Embedding(num_embeddings=writer_number,
                                                 embedding_dim=writer_embed_size,
                                                 padding_idx=padding_idx)
        if add_writer_as_hidden:
            assert writer_number!=None, "Remember to add writer number (+1 for padding) if you want to include writers' embeddings!"
            self.init_embedding = nn.Embedding(num_embeddings=writer_number,
                                               embedding_dim=self.hidden_size,
                                               padding_idx=padding_idx)
        if LSTM:
            if add_writer:
                self.rnn = nn.LSTM(input_size=self.embed_size+writer_embed_size,
                                hidden_size=self.hidden_size,
                                batch_first=batch_first,
                                num_layers=self.num_layers,
                                bidirectional=self.bidirectional)
            else:
                self.rnn = nn.LSTM(input_size=self.embed_size,
                                hidden_size=self.hidden_size,
                                batch_first=batch_first,
                                num_layers=self.num_layers,
                                bidirectional=self.bidirectional)
            
        else:
            if add_writer:
                self.rnn = nn.GRU(input_size=self.embed_size+writer_embed_size,
                                hidden_size=self.hidden_size,
                                batch_first=batch_first,
                                num_layers=self.num_layers,
                                bidirectional=self.bidirectional)
            else:
                self.rnn = nn.GRU(input_size=self.embed_size,
                                hidden_size=self.hidden_size,
                                batch_first=batch_first,
                                num_layers=self.num_layers,
                                bidirectional=self.bidirectional)
        
        self.dropout_in = dropout_in
        
        self.dropout_out = dropout_out
        
        if self.bidirectional:
            self.output = nn.Linear(hidden_size*2,voc_size)
        else:
            self.output = nn.Linear(hidden_size, voc_size)
        
        
    def forward(self, line, apply_softmax=False):
        try:
            line, line_len, writer = line['src_tokens'], line['src_lengths'], line['writer']        
        except TypeError:
            line = line[0]
        embedded = self.embedding(line)
        if self.add_writer:
            writer_embedding = self.writer_embedding(writer)
            embedded = torch.cat((embedded, writer_embedding),2)
        
        
        if self.bidirectional:
            
            batch_size, src_time_steps = line.size()
            
            state_size = 2 * self.num_layers, batch_size, self.hidden_size
            
            embedded_pack = PACK(embedded, line_len, batch_first=True)
        
            hidden_initial = embedded.new_zeros(*state_size)
            
            cells_initial = embedded.new_zeros(*state_size)
            
            packed_outputs, (hidden, cells) = self.rnn(embedded_pack,(hidden_initial,cells_initial))
            
            rnn_output, (final_hidden_states, final_cell_states) = nn.utils.rnn.pad_packed_sequence(packed_outputs, padding_value=0.)
            
            def combine_directions(outs):
                return torch.cat([outs[0: outs.size(0): 2], outs[1: outs.size(0): 2]], dim=2)
            final_hidden_states = combine_directions(final_hidden_states)
            final_cell_states = combine_directions(final_cell_states)
            
            src_mask = line.eq(self.padding_idx)

            return {'src_embeddings': embedded.transpose(0, 1),
                    'src_out': (rnn_output, final_hidden_states, final_cell_states),
                    'src_mask': src_mask if src_mask.any() else None}
        
        else:
            
            if self.dropout_in:
                _embedded = F.dropout(embedded, p=self.dropout_in)
                if self.writer_as_hidden:
                    assert type(self.rnn)==torch.nn.modules.rnn.GRU, "Just GRU currently supported for additional conditioning factor as first hidden state"
                    batch_size,length = writer.shape
                    writer = writer.view(-1)[::length]
                    writer_hidden_embedding = self.init_embedding(writer)
                    writer_hidden_embedding = writer_hidden_embedding.unsqueeze(0)
                    writer_hidden_embedding = writer_hidden_embedding.repeat(self.num_layers,1,1).view(-1,
                                                                            *writer_hidden_embedding.shape[1:])
                    rnn_out, h_n = self.rnn(_embedded, writer_hidden_embedding)
                else:
                    rnn_out, h_n = self.rnn(_embedded)
            else:
                if self.writer_as_hidden:
                    assert type(self.rnn)==torch.nn.modules.rnn.GRU, "Just GRU currently supported for additional conditioning factor as first hidden state"
                    batch_size,length = writer.shape
                    writer = writer.view(-1)[::length]
                    writer_hidden_embedding = self.init_embedding(writer)#put writer in right dimension
                    writer_hidden_embedding = writer_hidden_embedding.unsqueeze(0)
                    writer_hidden_embedding = writer_hidden_embedding.repeat(self.num_layers,1,1).view(-1,
                                                                            *writer_hidden_embedding.shape[1:])
                    rnn_out,h_n = self.rnn(embedded, writer_hidden_embedding)
                else:
                    rnn_out,h_n = self.rnn(embedded)
            batch_size, seq_size, feat_size = rnn_out.shape
            rnn_out = rnn_out.contiguous().view(batch_size*seq_size,feat_size)
            if self.dropout_out:
                rnn_out = F.dropout(rnn_out, p=self.dropout_out)
            out = self.output(rnn_out)
            if apply_softmax:
                out = F.softmax(out,dim=1)
            new_feat_size = out.shape[-1]
            out = out.view(batch_size, seq_size, new_feat_size)
            return h_n, out



def sentence_to_target(line, voc_2_index, by_character=False):
    if by_character:
        idx = [voc_2_index[token] for token in line] + [voc_2_index['<\s>']]
    else:
        if 'UKN' in voc_2_index:
            idx = [voc_2_index[token] if token in voc_2_index else voc_2_index['UKN'] for token in line.split()[1:]]
        else:
            idx = [voc_2_index[token] for token in line.split()[1:]]
    return torch.tensor(idx, dtype=torch.long)


def sentence_to_train(line, voc_2_index, by_character=False):
    if by_character:
        idx = [voc_2_index['<s>']]+[voc_2_index[token] for token in line]
    else:
        if 'UKN' in voc_2_index:
            idx = [voc_2_index[token] if token in voc_2_index else voc_2_index['UKN'] for token in line.split()[:-1]]
        else:
            idx = [voc_2_index[token] for token in line.split()[:-1]]
    return torch.tensor(idx, dtype=torch.long)



class LMDataset(Dataset):
    def __init__(self, lines, vocabulary):
        self.vocabulary = vocabulary
        self.src_dataset = [line[0] for line in lines]
        self.src_sizes = np.array([len(tokens) for tokens in self.src_dataset])

        self.tgt_dataset = [line[1] for line in lines]
        self.tgt_sizes = np.array([len(tokens) for tokens in self.tgt_dataset])
        self.writer = [line[2] for line in lines]
    
    def __getitem__(self, index):
        return {
            'id': index,
            'source': torch.LongTensor(self.src_dataset[index]),
            'target': torch.LongTensor(self.tgt_dataset[index]),
            'writer': torch.LongTensor(self.writer[index])
        }
        
    def __len__(self):
        return len(self.src_dataset)
    
    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        if len(samples) == 0:
            return {}
        def merge(values, move_eos_to_beginning=False):
            max_length = max(v.size(0) for v in values)
            result = values[0].new(len(values), max_length).fill_(self.vocabulary['<pad>'])
            for i, v in enumerate(values):
#                if move_eos_to_beginning:
#                    assert v[-1] == self.src_dict.eos_idx
#                    result[i, 0] = self.src_dict.eos_idx
#                    result[i, 1:len(v)] = v[:-1]
#                else:
                result[i, :len(v)].copy_(v)
            return result

        id = torch.LongTensor([s['id'] for s in samples])
        src_tokens = merge([s['source'] for s in samples])
        writer_tokens = merge([s['writer'] for s in samples])
        tgt_tokens = merge([s['target'] for s in samples])
        tgt_inputs = merge([s['target'] for s in samples], move_eos_to_beginning=True)

        # Sort by descending source length (not necessarily needed)
        src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
        src_lengths, sort_order = src_lengths.sort(descending=True)
        id = id.index_select(0, sort_order)
        src_tokens = src_tokens.index_select(0, sort_order)
        writer_tokens = writer_tokens.index_select(0, sort_order)
        tgt_tokens = tgt_tokens.index_select(0, sort_order)
        tgt_inputs = tgt_inputs.index_select(0, sort_order)

        return {
            'id': id,
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'tgt_tokens': tgt_tokens,
            'tgt_inputs': tgt_inputs,
            'num_tokens': sum(len(s['target']) for s in samples),
            'writer': writer_tokens
        }
        

def to_device(data, device):
    if isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
