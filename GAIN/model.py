import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from utils import normalization, renormalization, rounding
from utils import xavier_init
from utils import binary_sampler, uniform_sampler, sample_batch_index

class Generator(nn.Module):
    def __init__(self,dim,h_dim): 
        super(Generator, self).__init__()
        #Generator variables
        # Data + Mask as inputs (Random noise is in missing components)
        self.G_W1 = nn.Parameter(xavier_init([dim*2, h_dim]))  
        self.G_b1 = nn.Parameter(torch.zeros(size = [h_dim]))
    
        self.G_W2 = nn.Parameter(xavier_init([h_dim, h_dim]))
        self.G_b2 = nn.Parameter(torch.zeros(size = [h_dim]))
        
        self.G_W3 = nn.Parameter(xavier_init([h_dim, dim]))
        self.G_b3 = nn.Parameter(torch.zeros(size = [dim]))
    def forward(self,x,m):
        # Concatenate Mask and Data
        inputs = torch.cat([x, m], dim = 1) 
        G_h1 = F.relu(torch.matmul(inputs, self.G_W1) + self.G_b1)
        G_h2 = F.relu(torch.matmul(G_h1, self.G_W2) + self.G_b2)   
        # MinMax normalized output
        G_prob = torch.sigmoid(torch.matmul(G_h2, self.G_W3) + self.G_b3) 
        return G_prob

class Discriminator(nn.Module):
    def __init__(self,dim,h_dim): 
        super(Discriminator, self).__init__()
        # Discriminator variables
        self.D_W1 = nn.Parameter(xavier_init([dim*2, h_dim])) # Data + Hint as inputs
        self.D_b1 = nn.Parameter(torch.zeros(size = [h_dim]))
        
        self.D_W2 = nn.Parameter(xavier_init([h_dim, h_dim]))
        self.D_b2 = nn.Parameter(torch.zeros(size = [h_dim]))
        
        self.D_W3 = nn.Parameter(xavier_init([h_dim, dim]))
        self.D_b3 = nn.Parameter(torch.zeros(size = [dim]))  # Multi-variate outputs
    def forward(self,x,h):
        # Concatenate Data and Hint
        inputs = torch.cat([x, h], dim = 1) 
        D_h1 = F.relu(torch.matmul(inputs, self.D_W1) + self.D_b1)  
        D_h2 = F.relu(torch.matmul(D_h1, self.D_W2) + self.D_b2)
        D_logit = torch.matmul(D_h2, self.D_W3) + self.D_b3
        D_prob = torch.sigmoid(D_logit)
        return D_prob