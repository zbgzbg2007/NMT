import torch
import torch.nn as nn
import random

import torch.autograd as ag
from torch import optim
import torch.nn.functional as F
from model import *


class Language:
    # information about the language
    def __init__(self, name):
        # name (string): language name
        self.name = name
        self.word2index = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.word2count = {}
        self.num_words = 2 # total number of words
        self.SOS_token = 0
        self.EOS_token = 1


    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1
    def addSentence(self, sentence):
        for word in sentence.split(" "):
            self.addWord(word)

        
def sortFiles(file1, file2, newfile1, newfile2): # from orig/ to data/
    # sort the sentence pairs in file1 and file2 in decreasing order 
    # by sentence length in file1 
    # write sorted sentences into two new files: newfile1, newfile2 
    # all parameters are strings representing paths of the files 

    # read from two files
    lines1 = open(file1, encoding='utf-8').read().strip().split('\n')
    lines2 = open(file2, encoding='utf-8').read().strip().split('\n')
    pairs = [(lines1[i], lines2[i]) for i in range(len(lines1))]
    pairs.sort(key=lambda x: len(x[0].split()), reverse=True)
    
    with open(newfile1, 'w') as target_file1:
        with open(newfile2, 'w') as target_file2:
            for i in range(len(pairs)):
                target_file1.write(pairs[i][0]+'\n')
                target_file2.write(pairs[i][1]+'\n')


def readWords(file1, file2, lang1, lang2):
    # read words from two file lists into two language classes
    # file1 (list of strings): list of paths for language1 files including 
    # training file, test file and validation file.
    # file2 (list of strings): list of paths for language2 files corresponding those in file1 
    for j in range(3):
        lines1 = open(file1[j], encoding='utf-8').read().strip().split('\n')
        lines2 = open(file1[j], encoding='utf-8').read().strip().split('\n')
        for i in range(len(lines1)):
            lang1.addSentence(lines1[i])
            lang2.addSentence(lines2[i])


def VariablesFromPairs(pair, lang1, lang2):
    # generate Variables from pair of sentences
    # first sentence in language1, second in language2
    # pair (tuple of two strings): two sentences in two languages
    # lang1 (Language class): first language
    # lang2 (Language class): second language
    # return a pair of Variables representing word indices for the pair of sentences 
    id1 = [lang1.word2index[i] for i in pair[0].split(' ')]
    id2 = [lang2.word2index[i] for i in pair[1].split(' ')]
    id1.append(lang1.EOS_token)
    id2.append(lang2.EOS_token)
    v1 = ag.Variable(torch.LongTensor(id1)) # need additional dimension?
    v2 = ag.Variable(torch.LongTensor(id2))
    return (v1, v2)
    

def train(encoder, decoder, args):
    encoder_optim = optim.Adam(encoder.parameters(), lr=args.lr)
    loss = nn.NLLLOSS()
    training_batches = []
    validation_batchs = []



parser = argparse.ArgumentParser(description='NMT')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate') 
parser.add_argument('--epoch', type=int, default=30, help='total number of epochs to train; in each epoch we train all sentences once' )
parser.add_argument('--batch', type=int, default=64, help='batch size for training')
parser.add_argument('--gpu', type=bool, default=True, help='if use gpu')
parser.add_argument('--source-training-file', default='/data/train.de-en.de', help='path for source training file')
parser.add_argument('--target-training-file', default='/data/train.de-en.en', help='path for target training file')
parser.add_argument('--source-validation-file', default='/data/valid.de-en.de', help='path for source validation file')
parser.add_argument('--target-validation-file', default='/data/valid.de-en.en', help='path for target validation file')
parser.add_argument('--source-testing-file', default='/data/test.de-en.de', help='path for source testing file')
parser.add_argument('--target-testing-file', default='/data/test.de-en.en', help='path for target testing file')
