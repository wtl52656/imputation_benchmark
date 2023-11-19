import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

class GRUI(nn.Module):
    def __init__(self,input_size,h_dim): 
        super(GRUI, self).__init__()

        self.fc1 = nn.Linear(in_features=input_size+h_dim,\
                             out_features=h_dim)
        
        self.fc2 = nn.Linear(in_features=input_size+h_dim,\
                             out_features=h_dim)

        self.fc3 = nn.Linear(in_features=input_size+h_dim,\
                             out_features=h_dim)

 
    def forward(self,belta,x,state=None):
        """
        :param x: Input data of shape (batch_size,  N*F)
        :belta: Time decay coefficient (batch_size,  h_dim)
        :state: Previous step state (batch_size,  h_dim)
        :return: new_state (batch_size,  h_dim)
        """
        if state is None:
            state = torch.zeros_like(belta)
        state = belta * state

        u_t = torch.sigmoid(self.fc1(torch.cat([state,x],dim=-1)))
        r_t = torch.sigmoid(self.fc2(torch.cat([state,x],dim=-1)))

        new_state = torch.tanh(self.fc3(torch.cat([r_t*state,x],dim=-1)))
        new_state = (1-u_t)*state + u_t*new_state

        return new_state

class Generator_seq2z(nn.Module):
    def __init__(self,input_size,h_dim,z_dim): 
        super(Generator_seq2z, self).__init__()
        
        self.belta_fc = nn.Linear(in_features=input_size,\
                             out_features=h_dim)
        
        self.out_fc = nn.Linear(in_features=h_dim,\
                             out_features=z_dim)

        self.grui_cell = GRUI(input_size,h_dim)
 
    def forward(self,x, noise, time_delta):
        """
        :param x: observable data of shape (batch_size, sample_len, N*F)
        :noise: (batch_size, sample_len, N*F)
        :time_delta:  (batch_size, sample_len, N*F)
        :return: hidden state z (batch_size, z_dim)
        """
        B,T,N = x.shape

        x = x + noise

        belta = self.belta_fc(time_delta)
        belta = torch.exp(-torch.maximum(torch.zeros_like(belta),belta))

        state = None

        for i in range(T):
            state = self.grui_cell(belta[:,i],x[:,i],state)
        
        out = self.out_fc(state)
        return out

class Generator_z2seq(nn.Module):
    def __init__(self,input_size,h_dim,z_dim,seq_len=12): 
        super(Generator_z2seq, self).__init__()
        
        self.belta_fc = nn.Linear(in_features=input_size,\
                             out_features=h_dim)
        
        self.z_fc = nn.Linear(in_features=z_dim,\
                             out_features=input_size)

        self.grui_cell = GRUI(input_size,h_dim)

        self.out_fc = nn.Linear(in_features=h_dim,\
                             out_features=input_size)
        
        self.seq_len = seq_len
        self.input_size = input_size
        self.h_dim = h_dim
        self.z_dim = z_dim
 
    def forward(self,z,time_delta):
        """
        :param hidden state z (batch_size, z_dim)
        :time_delta:  (batch_size, sample_len, N*F)
        :return: Imputated results (batch_size, sample_len, N*F)
        """
        x = self.z_fc(z)

        belta = self.belta_fc(time_delta)
        belta = torch.exp(-torch.maximum(torch.zeros_like(belta),belta))

        state = self.grui_cell(belta[:,0],x)

        out = self.out_fc(state)

        tot_result = out.view(-1,1,self.input_size)

        for i in range(1,self.seq_len):
            state = self.grui_cell(belta[:,i],out,state)

            out = self.out_fc(state)
            
            tot_result = torch.cat([tot_result,out.view(-1,1,self.input_size)],dim=1)

        return tot_result    
        



class Generator(nn.Module):
    def __init__(self,input_size,h_dim,z_dim, seq_len): 
        super(Generator, self).__init__()
        self.seq2z = Generator_seq2z(input_size,h_dim,z_dim)
        self.z2seq = Generator_z2seq(input_size,h_dim,z_dim,seq_len)
 
    def forward(self,x,m, noise, time_delta):
        """
        :param x: observable data of shape (batch_size, sample_len, N*F)
        :m: mask matrix (batch_size, sample_len, N*F)
        :noise: (batch_size, sample_len, N*F)
        :time_delta:  (batch_size, sample_len, N*F)
        :return: Imputated results (batch_size, sample_len, N*F)
        """
        z = self.seq2z(x,noise,time_delta)
        imputated = self.z2seq(z,torch.ones_like(time_delta))

        return imputated


class Discriminator(nn.Module):
    def __init__(self,input_size,h_dim): 
        super(Discriminator, self).__init__()

        self.belta_fc = nn.Linear(in_features=input_size,\
                             out_features=h_dim)

        self.out_fc = nn.Linear(in_features=h_dim,\
                             out_features=1)

        self.grui_cell = GRUI(input_size,h_dim)

    def forward(self,x,time_delta):
        """
        :param x: Input data of shape (batch_size, sample_len, N*F)
        :time_delta:  (batch_size, sample_len, N*F)
        :return: Probability of data being true (batch_size, 1)
        """
        B,T,N = x.shape

        belta = self.belta_fc(time_delta)
        belta = torch.exp(-torch.maximum(torch.zeros_like(belta),belta)) # Time decay coefficient

        state = None
        for i in range(T):
            state = self.grui_cell(belta[:,i],x[:,i],state)

        out = self.out_fc(state)
        out = torch.sigmoid(out)
        return out