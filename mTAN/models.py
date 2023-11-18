#pylint: disable=E1101
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class multiTimeAttention(nn.Module):

    def __init__(self, input_dim, nhidden=16,
                 embed_time=16, num_heads=1):
        super(multiTimeAttention, self).__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = nhidden
        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time),
                                      nn.Linear(embed_time, embed_time),
                                      nn.Linear(input_dim*num_heads, nhidden)])

    def attention(self, query, key, value, mask=None, dropout=None):
        #query(B,N, h, num_ref, embed_time_k)
        #key(B,N, h, SEQ_LEN, embed_time_k)
        #value(B,N, h, SEQ_LEN, 2*F)
        #mask(B,N, h, SEQ_LEN, 2*F)
        "Compute 'Scaled Dot Product Attention'"
        dim = value.size(-1)#2*F
        d_k = query.size(-1)

        scores = torch.matmul(query, key.transpose(-2, -1)) \
            / math.sqrt(d_k)# scores(B,N, h, num_ref, SEQ_LEN)
        
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)# scores(B,N, h, num_ref, SEQ_LEN,2*F)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -1e9)#mask.unsqueeze(-3): (B,N, h,1, SEQ_LEN, 2*F)

        p_attn = F.softmax(scores, dim=-2)#(B,N, h, num_ref, SEQ_LEN,2*F)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.sum(p_attn*value.unsqueeze(-3), -2), p_attn#(B,N, h, SEQ_LEN,2*F)

    def forward(self, query, key, value, mask=None, dropout=None):
        #query:(B,N,num_ref,embed_time)
        #key:(B,N,seq_len,embed_time)
        
        "Compute 'Scaled Dot Product Attention'"
        batch, N, seq_len, dim = value.size()
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(2) #(B,N,1,SEQ_LEN,2*F)

        value = value.unsqueeze(2)  #(B,N,1,SEQ_LEN,F)

        #query和key经过一层linear后再转置成： (B,N, SEQ_LEN, h, embed_time_k) -> (B,N, h, SEQ_LEN, embed_time_k)
        query, key = [l(x).view(x.size(0), N, -1, self.h, self.embed_time_k).transpose(2, 3)
                      for l, x in zip(self.linears, (query, key))]
        
        x, _ = self.attention(query, key, value, mask, dropout)
        x = x.transpose(2, 3).contiguous() \
             .view(batch, N, -1, self.h * dim)#(B,N, SEQ_LEN, h*2*F)
        return self.linears[-1](x)#(B,N, SEQ_LEN, nhidden)


class enc_mtan_rnn(nn.Module):
    def __init__(self, input_dim, latent_dim=2, nhidden=16,
                 embed_time=16, num_heads=1, learn_emb=False, device='cuda'):
        super(enc_mtan_rnn, self).__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.learn_emb = learn_emb
        self.att = multiTimeAttention(
            2*input_dim, nhidden, embed_time, num_heads)
        self.gru_rnn = nn.GRU(
            nhidden, nhidden, bidirectional=True, batch_first=True)
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(2*nhidden, 50),
            nn.ReLU(),
            nn.Linear(50, latent_dim * 2))
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time-1)
            self.linear = nn.Linear(1, 1)

    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def fixed_time_embedding(self, pos):
        d_model = self.embed_time
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(10.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x, time_steps, query):
        # x(B,N,SEQ_LEN,F+F) 由原始输入和mask矩阵cat组成
        # print(query.shape)
        batch, N, _, _ = x.shape
        # time_steps = time_steps.cpu()
        mask = x[:, :, :, self.dim:]
        mask = torch.cat((mask, mask), 3)
        if self.learn_emb:  #可学习时间嵌入
            key = self.learn_time_embedding(time_steps).to(self.device)#(B,N,SEQ_LEN,embed_time)
            query = self.learn_time_embedding(query).to(self.device)#(B,N,num_ref,embed_time)
            # print(query.shape)
        else:   #固定时间嵌入
            key = self.fixed_time_embedding(time_steps).to(self.device)
            query = self.fixed_time_embedding(query).to(self.device)
        # print(query.shape)
        # print(key.shape)
        out = self.att(query, key, x, mask)#(B,N, SEQ_LEN, nhidden)
        out, _ = self.gru_rnn(out.view(-1, out.shape[2], out.shape[3]))#双向GRU
        out = out.view(batch, N, out.shape[1], out.shape[2])#(B,N,SEQ_LEN,2*nhidden)
        out = self.hiddens_to_z0(out)#(B,N,SEQ_LEN,2*latent_dim)
        return out


class dec_mtan_rnn(nn.Module):

    def __init__(self, input_dim, latent_dim=2, nhidden=16,
                 embed_time=16, num_heads=1, learn_emb=False, device='cuda'):
        super(dec_mtan_rnn, self).__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.learn_emb = learn_emb
        self.att = multiTimeAttention(
            2*nhidden, 2*nhidden, embed_time, num_heads)
        self.gru_rnn = nn.GRU(latent_dim, nhidden,
                              bidirectional=True, batch_first=True)
        self.z0_to_obs = nn.Sequential(
            nn.Linear(2*nhidden, 50),
            nn.ReLU(),
            nn.Linear(50, input_dim))
        if learn_emb:
            self.periodic = nn.Linear(1, embed_time-1)
            self.linear = nn.Linear(1, 1)

    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def fixed_time_embedding(self, pos):
        d_model = self.embed_time
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = 48.*pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(10.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, z, time_steps, query):
        #z(B*k_iwae,N,sql_len,latent_dim)
        batch, N, _, _ = z.shape
        out, _ = self.gru_rnn(z.view(-1, z.shape[2], z.shape[3]))
        out = out.view(batch, N, out.shape[1], out.shape[2])#(B*k_iwae,N,sql_len,2*latent_dim)
        # time_steps = time_steps.cpu()
        if self.learn_emb:
            key = self.learn_time_embedding(query).to(self.device)
            query = self.learn_time_embedding(time_steps).to(self.device)
        else:
            key = self.fixed_time_embedding(query).to(self.device)
            query = self.fixed_time_embedding(time_steps).to(self.device)
        out = self.att(query, key, out)#(B*k_iwae,N,sql_len,2*hidden)
        out = self.z0_to_obs(out)#(B*k_iwae,N,sql_len,input_dim)
        return out


class enc_dec_mtan(nn.Module):
    def __init__(self, input_dim, latent_dim=2, rec_hidden=16, gen_hidden=16, embed_time=16,
                 enc_num_heads=1, dec_num_heads=1, k_iwae=5, learn_emb=False, device='cuda'):
        super(enc_dec_mtan, self).__init__()
        self.enc = enc_mtan_rnn(
            input_dim, latent_dim, rec_hidden, embed_time, enc_num_heads, learn_emb, device)
        self.dec = dec_mtan_rnn(
            input_dim, latent_dim, gen_hidden, embed_time, dec_num_heads, learn_emb, device)
        self.k_iwae = k_iwae
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, x, mask, timestamp, query):
        #X,mask:(B,N,SEQ_LEN,F)
        #timestamp:(B,N,SEQ_LEN)
        #query:(B,N,num_ref)
        batch_len = x.shape[0]
        out = self.enc(torch.cat((x, mask), 3), timestamp, query) #编码层，将输入编程成一个分布(B,N,SEQ_LEN,2*latent_dim)
        qz0_mean = out[:, :, :, :self.latent_dim]
        qz0_logvar = out[:, :, :, self.latent_dim:]

        epsilon = torch.randn(
            self.k_iwae, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2], qz0_mean.shape[3]
        ).to(self.device)#生成k_iwae组高斯噪音

        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean#根据噪音生成隐变量
        z0 = z0.view(-1, qz0_mean.shape[1],
                     qz0_mean.shape[2], qz0_mean.shape[3])#(B*k_iwae,N,sql_len,latent_dim)
        
        out = self.dec(
            z0,
            timestamp[None, :, :, :].repeat(
                self.k_iwae, 1, 1, 1).view(-1, timestamp.shape[1], timestamp.shape[2]),
            query[None, :, :, :].repeat(
                self.k_iwae, 1, 1, 1).view(-1, query.shape[1], query.shape[2])
        ) #解码层，将隐变量解码(B*k_iwae,N,sql_len,F)

        out = out.view(self.k_iwae, batch_len,
                       out.shape[1], out.shape[2], out.shape[3])#(k_iwae,B,N,sql_len,F)
        return out, qz0_mean, qz0_logvar
