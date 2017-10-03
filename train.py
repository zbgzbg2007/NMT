import torch
import torch.nn as nn
import random

import torch.autograd as autograd
from torch import optim
import torch.nn.functional as F
from model import *

SOS_token = 0
EOS_token = 1

class Language:
    # information about the language
    def __init__(self, name):
        # name (string): language name
        self.name = name
        self.word2index = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.word2count = {}
        self.num_words = 2 # total number of words

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

        


def train(encoder, decoder, lr=0.0002):
    encoder_optim = optim.Adam(encoder.parameters(), lr=lr)
    loss = nn.NLLLOSS()
    
    
