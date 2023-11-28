import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math
import utils
import argparse
import data_loader

import models.rits_new as rits_new
from sklearn import metrics

# from ipdb import set_trace


class Model(nn.Module):
    def __init__(self, rnn_hid_size, attributes=3):
        super(Model, self).__init__()

        self.rnn_hid_size = rnn_hid_size
        self.attributes = attributes

        self.build()

    def build(self):
        #forward model and backward model
        self.rits_new_f = rits_new.Model(self.rnn_hid_size, self.attributes)
        self.rits_new_b = rits_new.Model(self.rnn_hid_size, self.attributes)

    def forward(self, data):
        ret_f = self.rits_new_f(data, 'forward')
        ret_b = self.reverse(self.rits_new_b(data, 'backward')) #reverse the result of backward model

        ret = self.merge_ret(ret_f, ret_b)

        return ret

    def merge_ret(self, ret_f, ret_b):
        loss_f = ret_f['loss']
        loss_b = ret_b['loss']
        loss_c = self.get_consistency_loss(ret_f['imputations'], ret_b['imputations'])

        loss = loss_f + loss_b + loss_c

        imputations = (ret_f['imputations'] + ret_b['imputations']) / 2 #take the mean of forward result and backward result

        ret_f['loss'] = loss
        ret_f['imputations'] = imputations

        return ret_f

    def get_consistency_loss(self, pred_f, pred_b):
        #Consistency between forward and backward results
        loss = torch.abs(pred_f - pred_b).mean() * 1e-1
        return loss

    def reverse(self, ret):
        #param ret:(B,seq_len,N*F)
        def reverse_tensor(tensor_):
            if tensor_.dim() <= 1:
                return tensor_
            indices = range(tensor_.size()[1])[::-1] # range(seq_len)[::-1] reverse indices
            indices = Variable(torch.LongTensor(indices), requires_grad = False)

            if torch.cuda.is_available():
                indices = indices.cuda()

            return tensor_.index_select(1, indices)

        for key in ret:
            ret[key] = reverse_tensor(ret[key])

        return ret

    def run_on_batch(self, data, optimizer, epoch=None):
        ret = self(data)

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret

