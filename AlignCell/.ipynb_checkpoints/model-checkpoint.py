import math
import torch
import torch.nn as nn
from performer_pytorch import FastAttention
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

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        q = self.linear_q(x).view(batch_size, seq_len, self.num_heads, self.dim_heads).transpose(1, 2)
        k = self.linear_k(x).view(batch_size, seq_len, self.num_heads, self.dim_heads).transpose(1, 2)
        v = self.linear_v(x).view(batch_size, seq_len, self.num_heads, self.dim_heads).transpose(1, 2)
        attn_output, attn_weight  = self.self_attn(q, k, v, output_attentions = True)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.linear_out(attn_output)
        x = self.dropout1(attn_output)
        x = self.norm1(x)
        ff_output = self.feed_forward(x)
        x = self.dropout2(ff_output)
        x = self.norm2(x)
        return x, attn_weight  # 返回最终输出和注意力输出

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, vocab_size, dropout=0):
        self.vocab_size = embedding.size(0)#来自embedding的参数
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
        print(x.shape)
        y = self.pe[:, : x.size(1), :]
        print(y.shape)
        x = x + y
        #return self.dropout(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, num_layers, num_heads, d_ff, max_len, dropout, fix_embedding=False, fast_attention_config=None):
        super(TransformerEncoder, self).__init__()
        input_dim ,d_model = embedding.size()
        #self.embedding = nn.Linear(input_dim, d_model)
        self.embedding = torch.nn.Embedding(input_dim, d_model)#参数：单词个数，词向量维度，来自于输入
        self.embedding.weight = torch.nn.Parameter(embedding)#初始化embedding层，让w2v的embedding层输入
        # 是否将embedding fix住，如果fix_embedding为False，在训练过程中，embedding也会跟着被训练
        self.embedding.weight.requires_grad = False if fix_embedding else True
        self.embedding_dim = embedding.size(1)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout, fast_attention_config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask=None):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        
        attentions = []  # 存储每一层的注意力
        for layer in self.layers:
            src, attn_weight = layer(src, mask)
            attentions.append(attn_weight)
        src = src[:, 0, :]
        return src, attentions
