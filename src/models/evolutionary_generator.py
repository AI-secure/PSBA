import numpy as np
from scipy import fftpack
import torch
import sys
import collections
import cv2
from numpy.linalg import norm as norm
import torch.nn as nn

class EAGenerator():
    def __init__(self, dimension_reduction = None, x_shape=(224,224,3), mu = 1, sigma = 0.01, decay_factor = 0.01, c = 0.001, dtype = np.float32):
        self.dimension_reduction = dimension_reduction
        self.x_shape = x_shape
        if self.dimension_reduction:
            self.pert_shape = (*dimension_reduction, x_shape[2])
        else:
            self.pert_shape = x_shape
        self.dtype = dtype
        self.sigma = sigma
        self.mu = mu
        self.c = c
        self.evolution_path = np.zeros(self.pert_shape, dtype=self.dtype)
        self.diagonal_covariance = np.ones(self.pert_shape, dtype=self.dtype)
        self.stats_adversarial = collections.deque(maxlen=30)
        self.perturbation = None
        self.decay_factor = decay_factor
        self.N = np.prod(self.pert_shape)
        self.K = int(self.N / 20)
        
    def generate_ps(self, sample, target, N = 1, level=None,step=-1):
        
        sample, target = sample.transpose(1,2,0), target.transpose(1,2,0)
        unnormalized_source_direction = target - sample
        source_norm = np.linalg.norm(unnormalized_source_direction)
        
        selection_probability = self.diagonal_covariance.reshape(-1) / np.sum(self.diagonal_covariance)
        selected_indices = np.random.choice(self.N, self.K, replace=False, p=selection_probability)
        
        perturbation = np.random.normal(0.0, 1.0, self.pert_shape).astype(self.dtype)
        factor = np.zeros([self.N], dtype=self.dtype)
        factor[selected_indices] = 1
        perturbation *= factor.reshape(self.pert_shape) * np.sqrt(self.diagonal_covariance)
        self.perturbation = perturbation
        
        if self.dimension_reduction:
            perturbation_large = cv2.resize(perturbation, self.x_shape[:2])
            if len(perturbation_large.shape)<3:
                perturbation_large = perturbation_large[:,:,None]
        else:
            perturbation_large = perturbation
            
        biased = sample + self.mu * unnormalized_source_direction
        candidate = biased + self.sigma * source_norm * perturbation_large / np.linalg.norm(perturbation_large)
        candidate = target - (target - candidate) / np.linalg.norm(target - candidate) * np.linalg.norm(target - biased)
        gradient = (candidate - sample)[np.newaxis,:].transpose(0,3,1,2)
        return gradient
    
    def update(self, is_adversarial):
        is_adversarial = (is_adversarial+1)/2
        self.stats_adversarial.appendleft(is_adversarial)
        if is_adversarial:
            self.evolution_path = self.decay_factor * self.evolution_path + np.sqrt(1 - self.decay_factor ** 2) * self.perturbation
            self.diagonal_covariance = (1 - self.c) * self.diagonal_covariance + self.c * (self.evolution_path ** 2)
            
        if len(self.stats_adversarial) == self.stats_adversarial.maxlen:
            p_step = np.mean(self.stats_adversarial)
            self.mu *= np.exp(p_step - 0.2)
            self.stats_adversarial.clear()
    
    def calc_rho(self, gt, inp):
        return np.array([0.01])[0]