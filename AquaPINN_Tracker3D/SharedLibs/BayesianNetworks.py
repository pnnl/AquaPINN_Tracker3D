# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 10:07:55 2025

@author: chen096
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

# --------------------------
# Bayesian Linear Layer
# --------------------------

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_std=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        # Parameters for the weight posterior: mean and log-scale (rho)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        # Parameters for the bias posterior: mean and log-scale (rho)
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_rho.data.fill_(-10)  # small initial standard deviation
        self.bias_mu.data.uniform_(-stdv, stdv)
        self.bias_rho.data.fill_(-10)
    
    def forward(self, input):
        # Compute standard deviations via softplus to ensure positivity.
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)
        
        # Sample from the posterior using the reparameterization trick.
        weight_eps = torch.randn_like(self.weight_mu)
        bias_eps = torch.randn_like(self.bias_mu)
        weight = self.weight_mu + weight_sigma * weight_eps
        bias = self.bias_mu + bias_sigma * bias_eps
        
        return F.linear(input, weight, bias)
    
    def kl_divergence(self):
        # Compute KL divergence between posterior N(mu, sigma) and prior N(0, prior_std)
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)
        
        # For a normal distribution, the KL divergence is:
        # KL(N(mu, sigma) || N(0, prior_std)) = log(prior_std/sigma) + (sigma^2 + mu^2)/(2*prior_std^2) - 1/2
        kl_weight = (torch.log(self.prior_std / weight_sigma) +
                     (weight_sigma**2 + self.weight_mu**2) / (2 * self.prior_std**2) - 0.5)
        kl_bias = (torch.log(self.prior_std / bias_sigma) +
                   (bias_sigma**2 + self.bias_mu**2) / (2 * self.prior_std**2) - 0.5)
        return kl_weight.sum() + kl_bias.sum()


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
        for i in range(len(layers)-1):
            self.Znet.append(BayesianLinear(layers[i],layers[i+1]))
        self.Unet =BayesianLinear(layers[0],layers[1])
        self.Vnet = BayesianLinear(layers[0],layers[1])
        
        #self.init_params()
        
    def forward(self, x):
        '''
        Forward propagation of network
        '''
        U=self.activation(self.Unet(x))
        V=self.activation(self.Vnet(x))
        for linear in self.Znet[:-1]:
            Z=self.activation( linear(x) ) 
            x=Z*U + (1-Z)*V
        y=self.Znet[-1](x)
        return y
    
    def init_params(self):
        '''
        initialize network parameters useing Glorot normalization
        '''    
        return 
        initializer     = torch.nn.init.xavier_normal_ #iniFun_dict.get("Glorot normal")
        initializer_zero= torch.nn.init.zeros_ #siniFun_dict.get("zeros")
        for net in self.Znet+[self.Unet, self.Vnet]:
            initializer(net.weight)
            if net.bias is not None:
                initializer_zero(net.bias)
        return
    
    def kl_divergence(self):
        kl = 0 
        for net in self.Znet+[self.Unet, self.Vnet]:
            kl += net.kl_divergence()
        return kl
    def print(self):
        names=[f'layer{d}' for d in range(len(self.Znet))] + ['Unet', 'Vnet']
        for name, net in zip(names, self.Znet+[self.Unet, self.Vnet]):
            weight_sigma = F.softplus(net.weight_rho)
            bias_sigma = F.softplus(net.bias_rho)
            print(name, 'weight:', net.weight_mu.mean().item(), weight_sigma.mean().item(), 
                            'bias:',  net.bias_mu.mean().item(), bias_sigma.mean().item())
        return 
    