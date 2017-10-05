import torch
import torch.nn as nn
import random

import torch.autograd as autograd
from torch import optim
import torch.nn.functional as F


class Encoder(nn.Module):
    # encoder using GRU 
    def __init__(self, embed_size, input_size, hidden_size, num_layers, bidirectional, gpu):
        '''
        embed_size (int): number of features after embedding
        input_size (int): number of total words in input language
        hidden_size (int): number of features in a hidden state
        num_layers (int): number of layers of rnn
        bidirectional (bool): if rnn is bidirectional
        gpu (bool): if using gpu
        '''
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.input_size = input_size
        self.num_directions = 2 if bidrectional else 1
        self.hidden_size = hidden_size // self.num_directions
        self.num_layers = num_layers
        self.gpu = gpu
        
        self.embedding = nn.Embedding(input_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=True)
        
    def init_hidden(self, batch_size):
        # initialize a hidden state with all zeros
        # batch_size (int): size of batch 
        hidden = autograd.Variable(torch.zeros(self.num_layers * self.num_drections, batch_size, self.hidden_size))
        return hidden.cuda() if self.gpu else hidden

    def forward(self, inputs, inputs_len):
        # return all outputs, and last hidden state for entire sequence
        # inputs (Variable of shape (batch_size, seq_len, input_size)): padded input batch sequences for encoder
        # inputs_len (list of int): list of lengths for sequences in input batch
   
        batch_size = inputs.size()[0] 
        outputs = self.embedding(inputs)
        pack = nn.utils.rnn.pack_padded_sequence(outputs, inputs_len, batch_first=True)
        hidden = init_hidden(batch_size)
        outputs, hidden = self.rnn(pack, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(o,batch_first=True)
        return outputs, hidden


class Decoder(nn.Module):
    # decoder using lstm with soft attention
    def __init__(self, embed_size, output_size, hidden_size, attention_size, maxout_size, gpu):
        '''
        see https://arxiv.org/pdf/1409.0473.pdf
        embed_size (int): number of features after embedding
        output_size (int): number of total words in output language
        hidden_size (int): number of features in a hidden state, assumed to be the same as in encoder
        attention_size (int): number of features in a hidden state in the attention model
        maxout_size (int): number of features in a hidden state in the maxout layer
        gpu (bool): if using gpu
        '''
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = number_layers
        self.gpu = gpu
 
        self.embedding = nn.Embedding(output_size, embed_size)
        # assume encoder and decoder have the same hidden_size, this is used to preprocess the hidden state from encoder
        self.init_hidden = nn.Linear(hidden_size*2, hidden_size) 
        # combine context vector and input vector as input for rnn
        self.rnn = nn.GRU(embed_size+attention_size, hidden_size, batch_first=True)
        self.maxout = nn.Linear(hidden_size+embed_size+attention_size, maxout_size*2)
        self.output = nn.Linear(maxout_size, output_size) 
        # self.softmax = nn.Softmax()
        self.attention = Alignment(attention_size, hidden_size*2, hidden_size, gpu)

    def forward(self, inputs, hidden, encoder_hiddens):
        # return only one step output and hidden state
        # inputs (Variable of shape (batch_size, 1, output_size)): input batch of words 
        # hidden (Variable of shape (batch_size, 1, hidden_size)): the hidden state from previous step
        # encoder_hiddens (Variable of shape (batch_size, seq_len, output_size)): all hidden states from encoder
        embed = self.embedding(inputs)
        context = self.attention(encoder_hiddens, hidden)
        _, hidden = self.rnn(torch.cat((embed, context), 1), hidden)
        outputs = self.maxout(torch.cat((embed, context, hidden), 1).view(inputs.size()[0], -1))
        outputs = nn.MaxPool1d(outputs, 2, stride=2)
        outputs = self.output(outputs)
        return outputs, hidden
        
class Alignment(nn.Module):
    # attention layer
    def __init__(self, hidden_size, encoder_hidden_size, decoder_hidden_size, gpu):
        '''
        hidden_size (int): number of features in a hidden state 
        encoder_hidden_size (int): number of features in a hidden state in encoder
        decoder_hidden_size (int): number of features in a hidden state in decoder
        gpu (bool): if use gpu
        '''
        self.encoder_size = encoder_hidden_size
        self.decoder_size = decoder_hidden_size
        self.hidden_size = hidden_size
        self.gpu = gpu
        self.encoder_linear = nn.Linear(encoder_hidden_size, hidden_size)
        self.decoder_linear = nn.Linear(decoder_hidden_size, hidden_size)
        self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

        
    def forward(self, encoder_hiddens, decoder_hidden):
        # return the context vector c 
        # encoder_hiddens (Variable of shape (batch, seq_len, encoder_size)): all hidden states from encoder
        # decoder_hidden (Variable of shape (batch, 1, decoder_size)): previous hidden state from decoder

        results = []
        batch_size, seq_len, _ = encoder_hiddens.size() 
        encoder_hs = self.encoder_linear(encoder_hiddens.view(-1, self.encoder_size))
        encoder_hs = encoder_hs.view(batch_size, seq_len, -1) # of shape (batch, seq_len, self.hidden_size)
        decoder_h = self.decoder_linear(decoder_hidden.view(batch_size, -1)) 
        scores = ag.Variable(torch.zeros(batch_size, seq_len, 1))
        if self.gpu: scores = scores.cuda() 
        for i in range(batch_size): # compute context for each sequence 
            score = [] 
            for j in range(seq_len): # sequences may have different lengths
                if (encoder_hs[i,j,:] == 0).all(): # all zeros mean hidden state stops here
                    break
                score.append(self.v.dot(nn.Tanh(encoder_hs[i,j,:] + decoder_h[i])))
            score = torch.cat((nn.Softmax(torch.cat(scores), 1)
            scores[i,:score.size()[0]] = score

        return torch.bmm(encoder_hiddens, scores)
        

       
class NMT(nn.Module):
    def __init__(self, embed_size, input_size, output_size, hidden_size, attention_size, maxout_size, num_layers, bidirectional, gpu):
        self.encoder = Encoder(embed_size, input_size, hidden_size, num_layers, bidirectional, gpu)
        self.decoder = Decoder(embed_size, output_size, hidden_size, attention_size, maxout_size, gpu)

