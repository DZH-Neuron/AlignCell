import sys
import os

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)

import math
import torch
import torch.nn as nn
from performer_pytorch.performer_pytorch import FastAttention
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0, fast_attention_config=None):
        super(EncoderLayer, self).__init__()
        dim_heads = d_model // num_heads
        self.self_attn = FastAttention(dim_heads, **fast_attention_config)
        self.num_heads = num_heads
        self.dim_heads = dim_heads
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0)
        self.dropout2 = nn.Dropout(0)

    def forward(self, x, mask=None, output_attentions=False):
        batch_size, seq_len, _ = x.size()
        q = self.linear_q(x).view(batch_size, seq_len, self.num_heads, self.dim_heads).transpose(1, 2)
        k = self.linear_k(x).view(batch_size, seq_len, self.num_heads, self.dim_heads).transpose(1, 2)
        v = self.linear_v(x).view(batch_size, seq_len, self.num_heads, self.dim_heads).transpose(1, 2)
        if output_attentions:
            # If self.self_attn returns two values (attn_output, attn_weight)
            attn_output, attn_weight = self.self_attn(q, k, v, output_attentions=True)
        else:
            # If self.self_attn returns one value (attn_output)
            attn_output = self.self_attn(q, k, v, output_attentions=False)
            attn_weight = None  # 不返回 attention 权重
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.linear_out(attn_output)
        x = self.dropout1(attn_output)
        x = self.norm1(x)
        ff_output = self.feed_forward(x)
        x = self.dropout2(ff_output)
        x = self.norm2(x)
        
        if output_attentions:
            return x, attn_weight
        else:
            return x

class PositionalEncoding(nn.Module):
    def __init__(self, embedding, d_model, vocab_size, dropout=0):
        self.vocab_size = embedding.size(0) #Parameters from the embedding
        self.d_model = embedding.size(1)
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(vocab_size, d_model)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # print(x.shape)
        y = self.pe[:, : x.size(1), :]
        # print(y.shape)
        x = x + y
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embedding, input_dim, d_model, num_layers, num_heads, d_ff, max_len, dropout, fix_embedding=False, fast_attention_config=None):
        super(TransformerEncoder, self).__init__()
        #input_dim ,d_model = embedding.size()
        #self.embedding = nn.Linear(input_dim, d_model)
        self.embedding = torch.nn.Embedding(input_dim, d_model) #Parameters: number of words, word vector dimension, derived from the input.
        self.embedding.weight = torch.nn.Parameter(embedding) #Initialize the embedding layer to accept the w2v embedding layer as input.
        # Whether to fix the embedding. If fix_embedding is False, the embedding will also be trained during the training process.
        self.embedding.weight.requires_grad = False if fix_embedding else True
        self.embedding_dim = embedding.size(1)
        self.pos_encoder = PositionalEncoding(embedding, d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout, fast_attention_config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask=None, output_attentions=False):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        
        attentions = [] 
        for layer in self.layers:
            if output_attentions:
                src, attn_weight = layer(src, mask, output_attentions=True)
                attentions.append(attn_weight)
            else:
                src = layer(src, mask, output_attentions=False)
        
        src = src[:, 0, :]
        
        if output_attentions:
            return src, attentions
        else:
            return src
