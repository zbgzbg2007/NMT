import torch
import torch.nn as nn
import random
import numpy as np
import argparse

import torch.autograd as ag
from torch import optim
import torch.nn.functional as F
from model import *


class Language:
    # information about the language
    # contain all words from files, doesn't remove rare words
    def __init__(self, name, threshold):
        # name (string): language name
        # threshold (int): threshold for rare words, which will be mapped to sepcial token
        self.name = name
        self.threshold = threshold
        self.word2index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.index2count = {0: 1, 1: 1, 2: 1, 3: 1}
        self.num_words = 4 # total number of words
        self.PAD_token = 0
        self.SOS_token = 1
        self.EOS_token = 2
        self.UNK_token = 3


    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.index2count[self.num_words] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.index2count[self.word2index[word]] += 1

    def addSentence(self, sentence):
        for word in sentence.split(" "):
            self.addWord(word)

    def removeRare(self):
        c = 3
        for i in range(4, self.num_words):
            w = self.index2word[i]
            if self.index2count[i] < self.threshold:
                self.index2count[self.UNK_token] += self.index2count[i]
                self.word2index[w] = self.UNK_token
            else:
                c += 1
                self.word2index[w] = c
                self.index2word[c] = w
                self.index2count[c] = self.index2count[i]
        self.num_words = c + 1
        
def sortFiles(file1, file2, newfile1, newfile2): 
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
    for j in range(len(file1)):
        lines1 = open(file1[j], encoding='utf-8').read().strip().split('\n')
        for i in range(len(lines1)):
            lang1.addSentence(lines1[i])
    for j in range(len(file2)):
        lines2 = open(file2[j], encoding='utf-8').read().strip().split('\n')
        for i in range(len(lines2)):
            lang2.addSentence(lines2[i])


def IndicesFromPairs(pair, lang1, lang2):
    # generate indices from pair of sentences
    # first sentence in language1, second in language2
    # pair (tuple of two strings): two sentences in two languages
    # lang1 (Language class): first language
    # lang2 (Language class): second language
    # return a pair of Variables representing word indices for the pair of sentences 

    id1 = [lang1.word2index[i] for i in pair[0].split(' ')]
    id2 = [lang2.word2index[i] for i in pair[1].split(' ')]
    id1.append(lang1.EOS_token)
    id2.append(lang2.EOS_token)
    return (id1, id2)
    

def readPairs(file1, file2):
    # read pairs from two files
    # return a list of pairs of sentences
    lines1 = open(file1, encoding='utf-8').read().strip().split('\n')
    lines2 = open(file2, encoding='utf-8').read().strip().split('\n')
    return [(lines1[i], lines2[i]) for i in range(len(lines1))]




def train(myNMT, args, lang1, lang2):
    # train model
    # myNMT (NMT model): model to train
    # args (a set of parameters): from parser
    # lang1 (Language class): source language
    # lang2 (Language class): target language

    myoptim = optim.Adam(myNMT.parameters(), lr=args.lr)
    
    training_data = [ IndicesFromPairs(p, lang1, lang2) for p in readPairs(args.source_training_file, args.target_training_file) ]

    # generate batches
    def generateBatches(data, batch_size):
        batches = []
        batch = []
        for i in range(len(data)): 
            batch.append(data[i])
            if len(batch) >= batch_size:
                batches.append(batch)
                batch = []
        if batch != []:
           batches.append(batch)
           batch = []
        return batches

    training_batches_pairs = generateBatches(training_data, args.batch_size)
    
    # transfer batches to padded Variables
    training_batches = []
    source_len, target_len = [], []
    for b in training_batches_pairs: 
        source_batch = [ sentence[0]  for sentence in b] 
        target_batch = [ sentence[1]  for sentence in b] 
        source_len.append([len(s) for s in source_batch])
        target_len.append([len(s) for s in target_batch])
        max_len = source_len[-1][0]
        source_batch = [ s + [lang1.PAD_token] * (max_len - len(s)) for s in source_batch] 
        max_len = max(target_len[-1])
        target_batch = [ s + [lang2.PAD_token] * (max_len - len(s)) for s in target_batch] 

        # mask for target sentence
        source_variable = ag.Variable(torch.LongTensor(source_batch))
        target_variable = ag.Variable(torch.LongTensor(target_batch))
        if args.gpu:
            source_variable = source_variable.cuda()
            target_variable = target_variable.cuda()
        training_batches.append((source_variable, target_variable))
       
    for e in range(args.num_epoch):
        for i in range(len(training_batches)):
            source, target = training_batches[i]
            myoptim.zero_grad()
            loss = 0
            criterion = nn.CrossEntropyLoss()
     
            # train network
            encoder_outputs, encoder_hidden = myNMT.encoder(source, source_len[i])

            # encoder has bidirectional rnn, dimensions are different 
            decoder_hidden = myNMT.decoder.init_hidden(encoder_hidden) 
            batch_size, length = target.size()
            decoder_input = ag.Variable(torch.LongTensor([lang2.SOS_token]  * target.size()[0]))
            if args.gpu:
                decoder_input = decoder_input.cuda()
            for j in range(length):
                decoder_output, decoder_hidden = myNMT.decoder(decoder_input, decoder_hidden, encoder_outputs)

                # compute loss with mask 
                mask_tensor = torch.from_numpy((np.array(target_len[i]) > j).astype(np.int32)).byte()
                masked_index = ag.Variable(torch.masked_select(torch.arange(0, batch_size), mask_tensor).long())
                if args.gpu:
                    masked_index = masked_index.cuda()
                masked_outputs = torch.index_select(decoder_output, 0, masked_index)
                masked_targets = torch.index_select(target[:, j], 0, masked_index)
                loss += criterion(masked_outputs, masked_targets)

                decoder_input = target[:,j]

            loss = loss.div(sum(target_len[i]))
            loss.backward()
            torch.nn.utils.clip_grad_norm(myNMT.parameters(), args.clip)
            myoptim.step()
            print (time.strftime('%Hh %Mm %Ss', time.localtime()), " batch ", i)

        test = evaluate(myNMT, args.source_validation_file, args.target_validation_file, args, lang1, lang2)
        print (time.strftime('%Hh %Mm %Ss', time.localtime()), " epoch ", e, " evaluate accuracy ", test)
        print (time.strftime('%Hh %Mm %Ss', time.localtime()), " epoch ", e, " evaluate accuracy ", test, file=open(args.process_file, 'a'))
        torch.save(myNMT.state_dict(), args.weights_file+str(e))
            


def evaluate(myNMT, source_file, target_file, args, lang1, lang2, predict=False, max_len=25):
    # evaluate model for data in source file, write outputs into target file
    # myNMT (NMT model): model to evaluate
    # source_file(string): path to the source file
    # target_file(string): if predict is True, we write outputs into this path, else this is the target file containing ground truth
    # args (a set of parameters): from parser
    # lang1 (Language class): source language
    # lang2 (Language class): target language
    # max_len (int): maximum length for the generated sequence 
    # predict (bool): if True, this function is for predict only without ground truth; otherwise it is for evaluation by simply comparing outputs with ground truth 
    
    # read and transfer data from file
    lines = open(source_file, encoding='utf-8').read().strip().split('\n')
    data = [[lang1.word2index[i] for i in l.split(' ')] + [lang1.EOS_token] for l in lines]
    outputs = []

    w = 0
    for seq in data:
        inputs = ag.Variable(torch.LongTensor(seq), volatile=True).unsqueeze(0)
        if args.gpu: inputs = inputs.cuda()
        encoder_outputs, encoder_hidden = myNMT.encoder(inputs, [len(seq)])


        # encoder has bidirectional rnn, dimensions are different 
        decoder_hidden = myNMT.decoder.init_hidden(encoder_hidden) 
        decoder_input = ag.Variable(torch.LongTensor([lang2.SOS_token]))
        if args.gpu:
            decoder_input = decoder_input.cuda()
        decoder_outputs = []
        for i in range(max_len):
            decoder_output, decoder_hidden = myNMT.decoder(decoder_input, decoder_hidden, encoder_outputs)
            top_v, top_i = decoder_output.data.topk(1)
            index = top_i[0][0]
            if index == lang2.EOS_token: 
                break
            else:
                decoder_outputs.append(lang2.index2word[index])
            decoder_input = ag.Variable(torch.LongTensor([index]))
            if args.gpu: decoder_input = decoder_input.cuda()
        outputs.append(decoder_outputs)

    if predict:
        with open(target_file, 'w') as t:
            for l in outputs:
                line = " ". join(l)
                t.write(line+'\n')
    else:
        correct, total = 0, 0
        lines = open(source_file, encoding='utf-8').read().strip().split('\n')
        for i in range(len(outputs)):
            ans = lines[i].split(' ')
            for j in range(min(len(ans), len(outputs[i]))):
                if ans[j] == outputs[i][j]:
                    correct += 1
                total += 1
        return correct / total
                
            
            
        

parser = argparse.ArgumentParser(description='NMT')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate') 
parser.add_argument('--num-epoch', type=int, default=25, help='total number of epochs to train; in each epoch we train all sentences once' )
parser.add_argument('--batch_size', type=int, default=256, help='batch size for training')
parser.add_argument('--gpu', type=bool, default=True, help='if use gpu')
parser.add_argument('--source-training-file', default='data/train.de-en.de', help='path for source training file')
parser.add_argument('--target-training-file', default='data/train.de-en.en', help='path for target training file')
parser.add_argument('--source-validation-file', default='orig/valid.de-en.de', help='path for source validation file')
parser.add_argument('--target-validation-file', default='orig/valid.de-en.en', help='path for target validation file')
parser.add_argument('--source-testing-file', default='orig/test.de-en.de', help='path for source testing file')
parser.add_argument('--target-testing-file', default='orig/test.de-en.en', help='path for target testing file')
parser.add_argument('--predict-file', default='data/predict', help='path for file of predicted outputs')
parser.add_argument('--source-lang', default='German', help='name for the source language')
parser.add_argument('--target-lang', default='English', help='name for the target language')
parser.add_argument('--embed-dim', type=int, default=500, help='number of features in an embedded word vector')
parser.add_argument('--input-size', type=int, default=50000, help='max number of total words in source language')
parser.add_argument('--output-size', type=int, default=50000, help='max number of total words in target language')
parser.add_argument('--encoder-hidden-size', type=int, default=256, help='the number of features in a hidden state in the encoder')
parser.add_argument('--decoder-hidden-size', type=int, default=256, help='the number of features in a hidden state in the decoder')
parser.add_argument('--attention-size', type=int, default=1000, help='the number of features in a hidden state in the attention model')
parser.add_argument('--maxout-size', type=int, default=500, help='the number of features in a hidden state in the maxout layer in the decoder')
parser.add_argument('--encoder-num-layers', type=int, default=1, help='the number of layers in the encoder')
parser.add_argument('--decoder-num-layers', type=int, default=1, help='the number of layers in the decoder')
parser.add_argument('--clip', type=float, default=40, help='clip gradient norm')
parser.add_argument('--weights-file', default='model-parameter', help='path to file to save model parameters')
parser.add_argument('--process-file', default='training-process', help='path to file recording training process')
parser.add_argument('--threshold', type=int, default=3, help='threshold for language dictionary to remove rare words')


def main():
    args = parser.parse_args()
    args.gpu = args.gpu and torch.cuda.is_available()
    
    #sortFiles('orig/train.de-en.de', 'orig/train.de-en.en', args.source_training_file, args.target_training_file)

    lang1 = Language(args.source_lang, args.threshold)
    lang2 = Language(args.target_lang, args.threshold)
    readWords([args.source_training_file, args.source_validation_file, args.source_testing_file], [args.target_training_file], lang1, lang2)
    lang1.removeRare()
    lang2.removeRare()
    args.input_size, args.output_size = lang1.num_words, lang2.num_words
    print (lang1.num_words, lang2.num_words)
    
    myNMT = NMT(args.embed_dim, args.input_size, args.output_size, args.encoder_hidden_size, args.attention_size, args.maxout_size, num_layers=1, bidirectional=True, gpu=args.gpu) 
    
    if args.gpu:
        myNMT.cuda()
    print (myNMT) 
    print ("start training")

    train(myNMT, args, lang1, lang2)

    evaluate(myNMT, args.source_training_file, args.predict_file, args, lang1, lang2, True)

if __name__ == '__main__':
    main()
