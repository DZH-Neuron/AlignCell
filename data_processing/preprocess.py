import os
import torch
import argparse
import numpy as np
from torch import nn
from gensim.models import Word2Vec

class Preprocess_gene():
    def __init__(self, sentences, sen_len, w2v_path=None):
        self.sentences = sentences
        self.sen_len = sen_len
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = []
        if w2v_path is None:
            # 获取当前文件路径（假设当前文件在 AlignCell 目录下）
            self.base_dir = os.path.dirname(__file__)
            self.w2v_path = os.path.join(self.base_dir, 'hum_dic_gene2vec.model')
          # 确保路径存在
        if not os.path.exists(self.w2v_path):
            raise FileNotFoundError(f"!!!not found：{self.w2v_path}")
    def get_w2v_model(self):
        self.embedding = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size
    def add_embedding(self, word):
        vector = torch.empty(1, self.embedding_dim)
        torch.nn.init.uniform_(vector)# torch.nn.init.uniform_(tensor, a=0, b=1)服从均匀分布U（0,1）
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)
    def make_embedding(self, load=True):
        print("Get embedding ...")
        if load:
            print("loading word to vec model ...")
            self.get_w2v_model()
        else:
            raise NotImplementedError
        for i, word in enumerate(self.embedding.wv.key_to_index):
            print('get words #{}'.format(i+1), end='\r')
            #e.g. self.word2index['he'] = 1 
            #e.g. self.index2word[1] = 'he'
            #e.g. self.vectors[1] = 'he' vector
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(self.embedding.wv[word])
        print('')
        self.embedding_matrix = torch.tensor(np.array(self.embedding_matrix))
        self.add_embedding("<SLC>")
        self.add_embedding("<PAD>")
        self.add_embedding("<UNK>")
        print("total words: {}".format(len(self.embedding_matrix)))
        return self.embedding_matrix
    def add_slc(self, sentence):
        sentence.insert(0,self.word2idx["<SLC>"])
        return sentence
    def pad_sequence(self, sentence):
        if len(sentence) > self.sen_len:
            sentence = sentence[:self.sen_len]
        else:
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx["<PAD>"])
        assert len(sentence) == self.sen_len
        return sentence
    def sentence_word2idx(self):
        sentence_list = []
        for i, sen in enumerate(self.sentences):
            print('sentence count #{}'.format(i+1), end='\r')
            sentence_idx = []
            for word in sen:
                if (word in self.word2idx.keys()):
                    sentence_idx.append(self.word2idx[word])
                else:
                    sentence_idx.append(self.word2idx["<UNK>"])
            sentence_idx = self.pad_sequence(sentence_idx)
            sentence_idx = self.add_slc(sentence_idx)
            sentence_list.append(sentence_idx)
        return torch.LongTensor(sentence_list)
    def labels_to_tensor(self, z):
        return torch.LongTensor(z)
    def exp_to_tensor(self, y):
        y = [[1] + exp for exp in y]
        y = [exp[:2000] + [0] * (2000 - len(exp[:2000])) for exp in y]
        return torch.LongTensor(y)
