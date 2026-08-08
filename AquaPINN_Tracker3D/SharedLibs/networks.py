# -*- coding: utf-8 -*-
import torch
import numpy as np

#modified forward neural network
class MFNN(torch.nn.Module):
    def __init__(self, layers):
        '''
        Parameters
        ----------
        layers : list
            The number of neurons for each layer        
        '''        
        super(MFNN, self).__init__()
        self.layers=layers
        self.activation = torch.relu #torch.nn.LeakyReLU() #torch.relu #tanh
        self.Znet = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        for i in range(len(layers)-1):
            self.Znet.append(torch.nn.Linear(layers[i],layers[i+1]))
            self.dropouts.append(torch.nn.Identity())
        self.Unet = torch.nn.Linear(layers[0],layers[1])
        self.Vnet = torch.nn.Linear(layers[0],layers[1])
        
        #self.init_params()
        
    def forward(self, x):
        '''
        Forward propagation of network
        '''
        U=self.activation(self.Unet(x))
        V=self.activation(self.Vnet(x))
        for linear,dropout in zip(self.Znet[:-1], self.dropouts[:-1]):
            x = dropout(linear(x))
            Z=self.activation( x ) 
            x=Z*U + (1-Z)*V
        y=self.Znet[-1](x)
        return y
    
    def init_params(self):
        for net in self.Znet+[self.Unet, self.Vnet]:
            stdv = 1. / np.sqrt(net.in_features)
            net.weight.data.uniform_(-stdv, stdv)
            if net.bias is not None:
                net.bias.data.uniform_(-stdv, stdv) 
        return

class MFNN_feature(torch.nn.Module):
    """Fully-connected neural network."""
    def __init__(self, layers, distance=0.5):
        '''
        Parameters
        ----------
        layers : list
            The number of neurons for each layer        
        '''        
        super(MFNN_feature, self).__init__()
        self.layers=layers
        self.activation = torch.relu #torch.nn.LeakyReLU() #torch.relu #tanh
        self.Znet = torch.nn.ModuleList()
        for i in range(len(layers)-1):
            self.Znet.append(torch.nn.Linear(layers[i],layers[i+1]))
        self.Unet = torch.nn.Linear(layers[0],layers[1])
        self.Vnet = torch.nn.Linear(layers[0],layers[1])
        self.filterRatio = torch.nn.Parameter( torch.zeros( 1,layers[0],
                                                           dtype = torch.float32)
                                            )
        self.distance = 0.2
        self.init_params()

    def init_params(self):
        '''
        initialize network parameters useing Glorot normalization
        '''        
        initializer     = torch.nn.init.xavier_normal_ #iniFun_dict.get("Glorot normal")
        initializer_zero= torch.nn.init.zeros_ #siniFun_dict.get("zeros")
        for net in self.Znet+[self.Unet, self.Vnet]:
            initializer(net.weight)
            if net.bias is not None:
                initializer_zero(net.bias)
        return

    def forward_feature(self, x, ratio=0):
        '''
        Forward propagation of network
        '''
        x *= (1+ratio*self.distance)        
        U=self.activation(self.Unet(x))
        V=self.activation(self.Vnet(x))
        for linear in self.Znet[:-1]:
            Z=self.activation(linear(x))
            x=Z*U + (1-Z)*V
        y=self.Znet[-1](x)
        return y

    def forward_LF(self, x):
        return self.forward_feature(x, ratio=0)

    def forward(self,x):
        return self.forward_feature(x, self.filterRatio)
    
   
class BSplineModel3D(torch.nn.Module):
    def __init__(self, n_ctrl_pts=6, dims=3, k=4):
        super().__init__()
        self.n_ctrl_pts = n_ctrl_pts
        self.dims = dims
        self.k = k
        self.ctrl_pts = torch.nn.Parameter(torch.zeros(n_ctrl_pts, dims))        
        knot_head = torch.zeros(k)
        knot_tail = torch.ones(k)+1E-8
        middle_count = n_ctrl_pts+2 - k + 1
        knot_middle = torch.linspace(0, 1, middle_count + 1)[1:-1]  # 去掉首尾
        knot_vec = torch.cat([knot_head, knot_middle[1:-1], knot_tail], dim=0)
        self.knot_vec = knot_vec
    def init_params(self):
        return 
    def basis(self, u, i, k):
        knot = self.knot_vec
        if k == 1:    
            return ((u >= knot[i]) & (u < knot[i+1])).float()
        else:    
            denom1 = knot[i + k - 1] - knot[i]    
            denom2 = knot[i + k]     - knot[i + 1]
            term1, term2 = 0.0, 0.0
            if denom1 != 0:
                term1 = (u - knot[i]) / denom1 * self.basis(u, i, k-1)
            if denom2 != 0:
                term2 = (knot[i + k] - u) / denom2 * self.basis(u, i+1, k-1)
            return term1 + term2

    def build_basis_matrix(self, t):
        basis_matrix = torch.zeros(t.shape[0], self.n_ctrl_pts)
        for i in range(self.n_ctrl_pts):
            basis_matrix[:,i] = self.basis(t, i, self.k) .squeeze() 
        return basis_matrix

