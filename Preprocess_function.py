# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 20:05:05 2020

@author: Iacopo
"""
from __future__ import unicode_literals, print_function, division
from io import open
import os
import unicodedata
import re
import warnings
import pandas as pd
import argparse
import nltk

# Read into file, get the maximum length of sentence and tokenize

class Preprocess():
    def __init__(self, data='data'):
        self.apostrophe = re.compile("([a-zA-Z]')([a-zA-Z])")
        self.punctuation = re.compile("[\.,\!\?:\<\>\(\);\-\"]")
        self.end_punctuation = re.compile("[\.\!\?:;]")
        self.data_dir = os.path.join(os.getcwd(),data)
        
        
    # Turn a Unicode string to plain ASCII, thanks to
    # https://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )
    
    def get_writers(self):
        self.writers = {'<pad>':0}
        count=1
        if self.csv:
            writers = list(set(self.dataset.iloc[:,1]))
            for writer in writers:
                if writer in self.writers:
                    pass
                else:
                    self.writers[writer]=count
                    count+=1
        else:
            for root, directories, files in os.walk(self.data_dir):
                for file in files:
                    writer = re.findall("([^0-9]*)[0-9]*\.",file)[0]
                    if writer in self.writers:
                        pass
                    else:
                        self.writers[writer]=count
                        count+=1
    
    def get_files(self):
        return os.listdir(self.data_dir)
        
    
    # Lowercase, trim, and remove non-letter characters
    
    
    def normalizeString(self, s):
        s = self.unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?'`]+", r" ", s)
        s = re.sub(r"([a-zA-Z])'",r"\g<1>' ", s)
        s = re.sub(r"\s\s",r" ", s)
        return s
    
    def RemoveChaptersIndent(self, s):
        s = re.sub('[A-Z]+[\s]?([\s][A-Z]+)*[0-9]*\n', '', s)
        s = re.sub('[\n]+', ' ', s)
        s = re.sub('^ ', ' ', s)
        s = re.sub(r"\s\s",r" ", s)
        return s
    
    def RemoveChapters(self, s):
        s = re.sub('[A-Z]+[\s]?([\s][A-Z]+)*[0-9]*\n', '', s)
        s = re.sub('[\n]+', ' \n', s)
        s = re.sub('^ ', ' ', s)
        s = re.sub(r"\s[\s]+",r" ", s)
        s = re.sub('^[\s]+\n',' ', s)
        return s
    
    def Tokenize_sentences(self, s):
        s = re.sub('([\.\?\!]) ([A-Z])','\g<1>\~\g<2>', s)
        sentences = s.split('\~')
        return sentences

    def preprocess(self, line):
#        line= re.sub(self.apostrophe,'\g<1> \g<2>', line)
        
        return self.normalizeString(line)


    def preprocess_files(self, files=None, remove_hapaxes=False, 
                         by_stanzas=False, remove_threshold=1, 
                         separated_by_tabs=False, include_writers=False, 
                         by_character=False, use_NLTK=False, preserve_new_line=False):
        
        if not files:
            files = self.get_files()
        
        self.files=files
        
        self.extensions = set()
        
        for file in files:
            self.extensions.add(os.path.splitext(file)[1][1:])
        
        if len(self.extensions)>1:
            warnings.warn("Provided files have different extensions!")
        
        voc_2_index = {'<pad>':0,'<s>':1,'<\s>':2}
    
        word_counts = {}
        
        counter = 3
        
        max_len = 0
        
        os.chdir(self.data_dir)
        
        all_lines_gr = {}
        
        if self.extensions.pop()=='csv':
            self.csv = True
        else:
            self.csv = False
        
        line_count = 0
        
        
        if self.csv:
                for file in files:
                    self.dataset = pd.read_csv(file)
                    for row in range(len(self.dataset)):
                        if by_stanzas:
                            for line in self.dataset.iloc[row][0].split('\n'):
                                if line.lstrip().rstrip():
                                    if by_character:
                                        if include_writers:
                                            all_lines_gr[line_count] = [line.lstrip().rstrip(),self.dataset.iloc[row][1]]
                                        else:
                                            all_lines_gr[line_count] = [line.lstrip().rstrip()]
                                        line_count+=1
                                        for letter in line:
                                            if letter not in voc_2_index:
                                                word_counts[letter]=0
                                                voc_2_index[letter]=counter
                                                counter+=1
                                            word_counts[letter]+=1
                                    else:
                                        line = self.preprocess(line)
                                        if include_writers:
                                            all_lines_gr[line_count] = ['<s> '+line.lstrip().rstrip()+' <\s>',self.dataset.iloc[row][1]]
                                        else:
                                            all_lines_gr[line_count] = ['<s> '+line.lstrip().rstrip()+' <\s>']
                                        line_count+=1
                                        for token in line.split():
                                            if token not in voc_2_index:
                                                word_counts[token]=0
                                                voc_2_index[token]=counter
                                                counter+=1
                                            word_counts[token]+=1
                        else:
                            line = self.dataset.iloc[row][0].lstrip().rstrip()
                            if line:
                                if include_writers:
                                    all_lines_gr[line_count] = [line,self.dataset.iloc[row][1]]
                                else:
                                    all_lines_gr[line_count] = [line]
                                line_count+=1
                                if by_character:
                                    for letter in line:
                                        if letter not in voc_2_index:
                                            word_counts[letter]=0
                                            voc_2_index[letter]=counter
                                            counter+=1
                                        word_counts[letter]+=1
                                else:
                                    for token in line.split():
                                        if token not in voc_2_index:
                                            word_counts[token]=0
                                            voc_2_index[token]=counter
                                            counter+=1
                                        word_counts[token]+=1
        elif separated_by_tabs:
          for file in files:
            with open(file, encoding='utf-8') as f:
                    for line in f:
                        len_line = len(line.split())
                        if len_line==0:
                            continue
                        line = self.preprocess(line.split('\t')[0])
                        if include_writers:
                            all_lines_gr[line_count] = ['<s> '+line.lstrip().rstrip()+' <\s>',re.findall("([^0-9]*)[0-9]*\.",file)[0]]
                        else:
                            all_lines_gr[line_count] = ['<s> '+line.lstrip().rstrip()+' <\s>']
                        line_count += 1
                        if len_line>max_len:
                            max_len = len_line
                        if by_character:
                            for letter in line:
                                if letter not in voc_2_index:
                                    word_counts[letter]=0
                                    voc_2_index[letter]=counter
                                    counter+=1
                                word_counts[letter]+=1
                        else:
                            for token in line.split():
                                if token not in voc_2_index:
                                    word_counts[token]=0
                                    voc_2_index[token]=counter
                                    counter+=1
                                word_counts[token]+=1
        elif by_stanzas:
            for file in files:
                with open(file, encoding='utf-8') as f:
                    for line in f:
                        len_line = len(line.split())
                        if len_line==0:
                            continue
                        if by_character:
                            if include_writers:
                                all_lines_gr[line_count] = [line.lstrip().rstrip(),re.findall("([^0-9]*)[0-9]*\.",file)[0]]
                            else:
                                all_lines_gr[line_count] = [line.lstrip().rstrip()]
                        else:
                            line = self.preprocess(line)
                            if include_writers:
                                all_lines_gr[line_count] = ['<s> '+line.lstrip().rstrip()+' <\s>',re.findall("([^0-9]*)[0-9]*\.",file)[0]]
                            else:
                                all_lines_gr[line_count] = ['<s> '+line.lstrip().rstrip()+' <\s>']
                        line_count += 1
                        if len_line>max_len:
                            max_len = len_line
                        if by_character:
                            for letter in line:
                                if letter not in voc_2_index:
                                    word_counts[letter]=0
                                    voc_2_index[letter]=counter
                                    counter+=1
                                word_counts[letter]+=1
                        else:
                            for token in line.split():
                                if token not in voc_2_index:
                                    word_counts[token]=0
                                    voc_2_index[token]=counter
                                    counter+=1
                                word_counts[token]+=1
        elif use_NLTK:
            try:
                tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
            except:
                nltk.download()
                tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
            for file in files:
                temp = open(file, encoding='utf-8').read()
                if preserve_new_line:
                    temp = self.RemoveChapters(temp)
                else:
                    temp = self.RemoveChaptersIndent(temp)
                sentences = tokenizer.tokenize(temp)
                for line in sentences:
                    if by_character:
                        if include_writers:
                            all_lines_gr[line_count] = [line.lstrip().rstrip(),re.findall("([^0-9]*)[0-9]*\.",file)[0]]
                        else:
                            all_lines_gr[line_count] = [line.lstrip().rstrip()]
                    else:
                        line = self.preprocess(line)
                        if include_writers:
                            all_lines_gr[line_count] = ['<s> '+line.lstrip().rstrip()+' <\s>',re.findall("([^0-9]*)[0-9]*\.",file)[0]]
                        else:
                            all_lines_gr[line_count] = ['<s> '+line.lstrip().rstrip()+' <\s>']
                    line_count += 1
                    if by_character:
                        for letter in line:
                            if letter not in voc_2_index:
                                word_counts[letter]=0
                                voc_2_index[letter]=counter
                                counter+=1
                            word_counts[letter]+=1
                    else:
                        for token in line.split():
                            if token not in voc_2_index:
                                word_counts[token]=0
                                voc_2_index[token]=counter
                                counter+=1
                            word_counts[token]+=1
        else:
            for file in files:
                temp = open(file).read()
                if preserve_new_line:
                    temp = self.RemoveChapters(temp)
                else:
                    temp = self.RemoveChaptersIndent(temp)
                sentences = self.Tokenize_sentences(temp)
                for line in sentences:
                    if by_character:
                        if include_writers:
                            all_lines_gr[line_count] = ['<s>'+line.lstrip().rstrip()+'<\s>',re.findall("([^0-9]*)[0-9]*\.",file)[0]]
                        else:
                            all_lines_gr[line_count] = ['<s>'+line.lstrip().rstrip()+'<\s>']
                    else:
                        line = self.preprocess(line)
                        if include_writers:
                            all_lines_gr[line_count] = ['<s> '+line.lstrip().rstrip()+' <\s>',re.findall("([^0-9]*)[0-9]*\.",file)[0]]
                        else:
                            all_lines_gr[line_count] = ['<s> '+line.lstrip().rstrip()+' <\s>']
                    line_count += 1
                    if by_character:
                        for letter in line:
                            if letter not in voc_2_index:
                                word_counts[letter]=0
                                voc_2_index[letter]=counter
                                counter+=1
                            word_counts[letter]+=1
                    else:
                        for token in line.split():
                            if token not in voc_2_index:
                                word_counts[token]=0
                                voc_2_index[token]=counter
                                counter+=1
                            word_counts[token]+=1
#                    with open(file, encoding='utf-8') as f:
#                        line_gr = ''
#                        for line in f:
#                            len_line = len(line.split())
#                            if len_line==0:
#                                continue
#                            match = re.findall(self.end_punctuation, line)
#                            if match:
#                                line_post = []
#                                idx = []
#                                for index,pun in enumerate(match):
#                                    if index!=0:
#                                        line_post.append(line[sum(idx)+len(idx):])
#                                        
#                                        idx.append(line_post[-1].find(pun))
#                                        
#                    #                    print(idx)
#                                        
#                                        line_post[-1] = line_post[-1][:idx[-1]]
#                                    else:
#                                        idx.append(line.find(pun))
#                                        line_post.append(line[:idx[-1]])
#                                    
#                                while line_post:
#                                    line_post_element = line_post.pop(0)
#                                    if len(self.preprocess(line_post_element).split())==0:
#                                        continue
#                                    line_gr = line_gr + ' ' + self.preprocess(line_post_element).lstrip().rstrip()
#                    #                print(line_gr)
#                                    if include_writers:
#                                        all_lines_gr[line_count] = ['<s> '+line_gr.lstrip().rstrip()+' <\s>',re.findall("([^0-9]*)[0-9]*\.",file)[0]]
#                                    else:
#                                        all_lines_gr[line_count] = ['<s> '+line_gr.lstrip().rstrip()+' <\s>']
#                                    line_count += 1
#                                    line_gr = ''
#                            else:
#                                line_gr = line_gr + ' ' + self.preprocess(line).lstrip().rstrip()
#                            line = self.preprocess(line)
                            
        #                    if len_line>max_len:
        #                        max_len = len_line
#                            if by_character:
#                                for letter in line:
#                                    if letter not in voc_2_index:
#                                        word_counts[letter]=0
#                                        voc_2_index[letter]=counter
#                                        counter+=1
#                                    word_counts[letter]+=1
#                            else:
#                                for token in line.split():
#                                    if token not in voc_2_index:
#                                        word_counts[token]=0
#                                        voc_2_index[token]=counter
#                                        counter+=1
#                                    word_counts[token]+=1
        if remove_hapaxes:
            print('Total words: {}. Total hapaxes: {}'.format(len(voc_2_index),len([v for v in word_counts.values() if v==1])))
    
            counter = 3
            
            voc_2_index = {}
            
            for k,v in word_counts.items():
                if v>remove_threshold:
                    voc_2_index[k]=counter
                    counter+=1
            
            voc_2_index['pad']=0
            
            voc_2_index['<s>'] = 1
            
            voc_2_index['<\s>'] = 2
            
            voc_2_index['UKN'] = len(voc_2_index)
            
            print('Total words after substituting hapaxes: {}'.format(len(voc_2_index)))
        
        word_counts = {k:v for k,v in reversed(sorted(word_counts.items(), key=lambda item: item[1]))}
    
        index_2_voc = {v:k for k,v in voc_2_index.items()}  
        
        self.get_writers()
        
        
        return voc_2_index, word_counts, all_lines_gr, index_2_voc, self.writers


if __name__ == '__main__':
    Preprocessor = Preprocess()
    voc_2_index, word_counts, all_lines_gr, index_2_voc, writer = Preprocessor.preprocess_files(files, by_stanzas=True)