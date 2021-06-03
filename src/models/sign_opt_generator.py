import numpy as np
from scipy import fftpack
import torch
from numpy.linalg import norm as norm
import torch.nn as nn
class SignOptGenerator():
    def __init__(self,alpha = 0.2, beta = 0.001):
        self.alpha = alpha
        self.beta = beta

    def generate_ps(self, sample, target, N, level=None,step=-1):
        theta = sample - target
        lbd = norm(theta)
        theta /= lbd
        theta = np.expand_dims(theta,0).repeat(N,axis=0)
        
        u = np.random.randn(*theta.shape)
        u /= np.sqrt(np.sum(u**2, axis=(1,2,3), keepdims=True))
        new_theta = theta + self.beta * u
        new_theta /= np.sqrt(np.sum(new_theta**2, axis=(1,2,3), keepdims=True))
        return new_theta
    
    def calc_rho(self, gt, inp):
        return np.array([0.01])[0]