import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math
import utils
import argparse
import data_loader

# from ipdb import set_trace
from sklearn import metrics


class FeatureRegression(nn.Module):
    def __init__(self, input_size):
        super(FeatureRegression, self).__init__()
        self.build(input_size)

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)  # 对角线为0，其他位置为1的矩阵
        self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # print("before z_h: ", x, self.W, self.m, self.b)
        z_h = F.linear(x, self.W * Variable(self.m), self.b)  # 自定义线性变换
        return z_h

class TemporalDecay(nn.Module):  # represent missing patterns
    def __init__(self, input_size, output_size, diag = False):
        super(TemporalDecay, self).__init__()
        self.diag = diag

        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = Parameter(torch.Tensor(output_size, input_size))  # 默认求梯度
        self.b = Parameter(torch.Tensor(output_size)) # 相加时触发了广播机制

        if self.diag == True:
            assert(input_size == output_size)
            m = torch.eye(input_size, input_size)  # 对角线为1的矩阵
            self.register_buffer('m', m)  # 这行代码可以在内存中定一个常量，同时，模型保存和加载的时候可以写入和读出，即这个变量不会参与反向传播。（不加上这行会报错吗？）

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        if self.diag == True:
            gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))  # self.W * Variable(self.m)意义何在？-> 只考虑自身维度的缺失模式！
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma) # 延迟因子
        return gamma

class Model(nn.Module):
    def __init__(self, rnn_hid_size, attributes=3):
        super(Model, self).__init__()

        self.rnn_hid_size = rnn_hid_size
        self.attributes = attributes

        self.build()

    def build(self):
        self.rnn_cell = nn.LSTMCell(self.attributes * 2, self.rnn_hid_size) # hid_size太大容易过拟合！

        self.temp_decay_h = TemporalDecay(input_size = self.attributes, output_size = self.rnn_hid_size, diag = False)
        self.temp_decay_x = TemporalDecay(input_size = self.attributes, output_size = self.attributes, diag = True)

        self.hist_reg = nn.Linear(self.rnn_hid_size, self.attributes)
        self.feat_reg = FeatureRegression(self.attributes)

        self.weight_combine = nn.Linear(self.attributes * 2, self.attributes)

    def forward(self, data, direct):
        values = data[direct]['values']
        values = torch.where(torch.isnan(values), torch.full_like(values, 0), values)
        masks = data[direct]['masks']
        deltas = data[direct]['deltas']  # (batch, time_step, attributes)
        # print(values.shape, masks.shape, deltas.shape)

        h = Variable(torch.zeros((values.size()[0], self.rnn_hid_size))) # 默认不求导，即requires_grad=False
        c = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))

        if torch.cuda.is_available():
            h, c = h.cuda(), c.cuda()

        x_loss = 0.0

        imputations = []

        time_steps = values.shape[1]
        for t in range(time_steps):  # 核心代码  不同样本的缺失模式是不同的，训练到的参数也不同！
            x = values[:, t, :]
            #x = np.nan_to_num(x)
            #print("x:",x)
            m = masks[:, t, :]
            #print("m:",m)
            d = deltas[:, t, :]
            #print("d:",d)
            #print("d_test;:",d)

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)
            #print("gamma_x_test:",gamma_x)

            h = h * gamma_h # 乘以延迟因子！

            x_h = self.hist_reg(h)
            if torch.sum(m) != 0:
                x_loss += torch.sum(torch.abs(x - x_h) * m) / torch.sum(m)

            x_c =  m * x +  (1 - m) * x_h

            z_h = self.feat_reg(x_c) # 特征维度唯一，z_h就没有意义了！ 只有bias了
            #print("z_h: ", z_h.shape, z_h)
            if torch.sum(m) != 0:
                x_loss += torch.sum(torch.abs(x - z_h) * m) / torch.sum(m)

            alpha = self.weight_combine(torch.cat([gamma_x, m], dim = 1))
            #print("alpha_test:",alpha)

            c_h = alpha * z_h + (1 - alpha) * x_h
            #print("c_h_test:",c_h)
            if torch.sum(m):
                x_loss += torch.sum(torch.abs(x - c_h) * m) / torch.sum(m)

            c_c = m * x + (1 - m) * c_h  # 未缺失值不做处理！
            #print("c_c_test:",c_c)

            inputs = torch.cat([c_c, m], dim = 1)

            h, c = self.rnn_cell(inputs, (h, c))

            imputations.append(c_c.unsqueeze(dim = 1))

        imputations = torch.cat(imputations, dim = 1)

        x_loss = x_loss / time_steps

        return {'loss': x_loss, \
                'imputations': imputations}

    def run_on_batch(self, data, optimizer, epoch = None):
        ret = self(data, direct = 'forward')

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret
