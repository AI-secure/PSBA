# reference : https://github.com/facebookresearch/pytorch_GAN_zoo/blob/master/models/progressive_gan.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import EqualizedConv2d, EqualizedLinear,\
    NormalizationLayer, Upscale2d
from .utils import num_flat_features
from .utils import miniBatchStdDev
import numpy as np
from skimage import transform
from scipy import fftpack
import torch
from sklearn.decomposition import PCA

def get_2d_dct(x):
    return fftpack.dct(fftpack.dct(x.T, norm='ortho').T, norm='ortho')

def get_2d_idct(x):
    return fftpack.idct(fftpack.idct(x.T, norm='ortho').T, norm='ortho')

def RGB_img_dct(img):
    assert len(img.shape) == 3 #and img.shape[0] == 3
    c_img = img.shape[0] # support 1-channel mnist data
    signal = np.zeros_like(img)
    for c in range(c_img):
        signal[c] = get_2d_dct(img[c])
    return signal

def RGB_signal_idct(signal):
    assert len(signal.shape) == 3 #and signal.shape[0] == 3
    c_signal = signal.shape[0] # support 1-channel mnist data
    img = np.zeros_like(signal)
    for c in range(c_signal):
        img[c] = get_2d_idct(signal[c])
    return img

class PGenerator(nn.Module):

    def __init__(self,
             dimLatent=9408,
             depthScale0=512,
             n_channels=3,
             preprocess=None,
             device='cuda',
             initBiasToZero=True,
             leakyReluLeak=0.2,
             normalization=True,
             generationActivation=None,
             equalizedlR=True,
             dct_factor = 1,
             topk = 1):
        r"""
        Build a generator for a progressive GAN model
        Args:
            - dimLatent (int): dimension of the latent vector
            - depthScale0 (int): depth of the lowest resolution scales
            - initBiasToZero (bool): should we set the bias to zero when a
                                    new scale is added
            - leakyReluLeak (float): leakyness of the leaky relu activation
                                    function
            - normalization (bool): normalize the input latent vector
            - generationActivation (function): activation function of the last
                                               layer (RGB layer). If None, then
                                               the identity is used
            - dimOutput (int): dimension of the output image. 3 -> RGB, 1 ->
                               grey levels
            - equalizedlR (bool): set to true to initiualize the layers with
                                  N(0,1) and apply He's constant at runtime
            - factor: the factor of DCT
        """
        super(PGenerator, self).__init__()
        self.preprocess = preprocess
        self.device = device
        self.equalizedlR = equalizedlR
        self.initBiasToZero = initBiasToZero
        self.n_channels = n_channels
        self.topk = topk
        self.dct_factor = dct_factor
        self.basis = None
        self.X_shape = None
        # Initalize the scales
        self.scalesDepth = [depthScale0]
        self.scaleLayers = nn.ModuleList()
        self.toRGBLayers = nn.ModuleList()
        # Initialize the scale 0
        self.initFormatLayer(dimLatent)
        self.dimOutput = n_channels
        self.groupScale0 = nn.ModuleList()
        self.groupScale0.append(EqualizedConv2d(depthScale0, depthScale0, 3,
                        equalized=equalizedlR,
                        initBiasToZero=initBiasToZero,
                        padding=1))
        self.toRGBLayers.append(EqualizedConv2d(depthScale0, self.dimOutput, 1,
                        equalized=equalizedlR,
                        initBiasToZero=initBiasToZero))
        # Initalize the upscaling parameters
        # alpha : when a new scale is added to the network, the previous
        # layer is smoothly merged with the output in the first stages of
        # the training
        self.alpha = 0

        # Leaky relu activation
        self.leakyRelu = torch.nn.LeakyReLU(leakyReluLeak)

        # normalization
        self.normalizationLayer = None
        if normalization:
            self.normalizationLayer = NormalizationLayer()

        # Last layer activation function
        self.generationActivation = nn.Tanh()
        self.depthScale0 = depthScale0
        self.to(self.device)
        
    def loadScale(self, depthScale=[256,256,128,64,32,16]):
        for scale in depthScale:
            self.addScale(scale)
        self.setNewAlpha(1)
        
    def initFormatLayer(self, dimLatentVector):
        r"""
        The format layer represents the first weights applied to the latent
        vector. It converts a 1xdimLatent input into a 4 x 4 xscalesDepth[0]
        layer.
        """

        self.dimLatent = dimLatentVector
        self.formatLayer = EqualizedLinear(self.dimLatent,
                       16 * self.scalesDepth[0],
                       equalized=self.equalizedlR,
                       initBiasToZero=self.initBiasToZero).to(self.device)
        
    def load(self, path):
        self.basis = np.load(path)
        self.basis = self.basis[:self.topk]
        self.X_shape = self.basis.shape[1:]
        self.basis = self.basis.reshape(self.basis.shape[0], -1)
            
    def getOutputSize(self):
        r"""
        Get the size of the generated image.
        """
        if len(self.toRGBLayers) == 1:
            return (4,4)
        else:
            side = 7 * (2**(len(self.toRGBLayers) - 2))
        return (side, side)

    def addScale(self, depthNewScale):
        r"""
        Add a new scale to the model. Increasing the output resolution by
        a factor 2
        Args:
            - depthNewScale (int): depth of each conv layer of the new scale
        """
        depthLastScale = self.scalesDepth[-1]

        self.scalesDepth.append(depthNewScale)
        self.scaleLayers.append(nn.ModuleList().to(self.device))

        self.scaleLayers[-1].append(EqualizedConv2d(depthLastScale,
                            depthNewScale,
                            3,
                            padding=1,
                            equalized=self.equalizedlR,
                            initBiasToZero=self.initBiasToZero).to(self.device))
        self.scaleLayers[-1].append(EqualizedConv2d(depthNewScale, depthNewScale,
                            3, padding=1,
                            equalized=self.equalizedlR,
                            initBiasToZero=self.initBiasToZero).to(self.device))

        self.toRGBLayers.append(EqualizedConv2d(depthNewScale,
                            self.dimOutput,
                            1,
                            equalized=self.equalizedlR,
                            initBiasToZero=self.initBiasToZero).to(self.device))

    def setNewAlpha(self, alpha):
        r"""
        Update the value of the merging factor alpha
        Args:
            - alpha (float): merging factor, must be in [0, 1]
        """

        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be in [0,1]")

        if not self.toRGBLayers:
            raise AttributeError("Can't set an alpha layer if only the scale 0"
                                 "is defined")

        self.alpha = alpha

    def forward(self, x):
        if not torch.is_tensor(x):
            # print("GAN generator convert input to FloatTensor")
            x = torch.FloatTensor(x)
        if not x.is_cuda:
            x = x.to(self.device)
        ## Normalize the input ?
        if self.normalizationLayer is not None:
            x = self.normalizationLayer(x)
        x = x.view(-1, num_flat_features(x))
        # format layer
        x = self.leakyRelu(self.formatLayer(x))
        x = x.view(x.size()[0], -1, 4, 4)

        x = self.normalizationLayer(x)

        # Scale 0 (no upsampling)
        for convLayer in self.groupScale0:
            x = self.leakyRelu(convLayer(x))
            if self.normalizationLayer is not None:
                x = self.normalizationLayer(x)

        # Dirty, find a better way
        if self.alpha != 1 and len(self.scaleLayers) == 1:
            y = self.toRGBLayers[-2](x)
            y = Upscale2d(y,7/4)

        # Upper scales
        for scale, layerGroup in enumerate(self.scaleLayers, 0):
            if scale == 0:
                x = Upscale2d(x,7/4)
            else:
                x = Upscale2d(x)
            for convLayer in layerGroup:
                x = self.leakyRelu(convLayer(x))
                if self.normalizationLayer is not None:
                    x = self.normalizationLayer(x)

            if self.alpha != 1 and scale == (len(self.scaleLayers) - 2):
                y = self.toRGBLayers[-2](x)
                y = Upscale2d(y)

        # To RGB (no alpha parameter for now)
        x = self.toRGBLayers[-1](x)

        # Blending with the lower resolution output when alpha > 0
        if self.alpha != 1 and len(self.scaleLayers):
            x = self.alpha * x + (1.0-self.alpha) * y

        if self.generationActivation is not None:
            x = self.generationActivation(x)
        return x

    def generate_ps(self, sample,target, N, level=None):
        inp = sample
        if self.preprocess is not None:
            transp, mean, std = self.preprocess
            inp = inp.transpose(*transp)
            inp = (inp - mean) / std
        
        # generate the 9408 main components for pgan224
#         ps = []
#         Z = np.random.randn(40000,self.dimLatent)
#         for i in range(400):
#             inm = self.forward(Z[100*i:100*(i+1)]).cpu().detach().numpy()
#             ps.append(inm)
#         ps = np.concatenate(ps, axis=0)
#         ps = ps.reshape(40000,-1)
#         from sklearn.decomposition import PCA
#         model = PCA(9408)
#         model.fit(ps)
#         components = model.components_
#         np.save("./gen_models/pca_pgan_gen_imagenet_9408_.npy",components.reshape(9408,3,224,224))

        with torch.no_grad():
            ## Spectrum_PCA ###    
            if self.topk != 0:
                rv = np.random.randn(N,self.topk)
                ps = rv.dot(self.basis).reshape(N, *self.X_shape)
#             if self.topk != 0:
#                 dot = []
#                 for i in range(N):
#                     interm = np.sum(np.tile(ps[i].reshape(1,-1),(self.topk,1)) * self.basis, axis = 1)
#                     dot.append(interm)
#                 ps = np.stack(dot, saxis=0)
#                 ps = ps.dot(self.basis).reshape(N,*inp.shape)
            
            else:
                Z = np.random.randn(N,self.dimLatent)
                ps_old = self.forward(Z).cpu().detach().numpy()
                ### comment off the annotation when testing the sensitivity of PGAN ###
                # bias = self.forward(np.zeros((1,self.dimLatent))).cpu().detach().numpy()
				# bias = transform.resize(bias[0].transpose(1,2,0), inp.transpose(1,2,0).shape).transpose(2,0,1)
				# ps_old = ps_old - bias
                ps = ps_old
                
                ### resize to 224*224 ###
                if ps.shape[-1] == 224:
                    pass
                else:
                    p = []
                    for i in range(n):
                        temp = transform.resize(ps[i].transpose(1,2,0), inp.transpose(1,2,0).shape).transpose(2,0,1)
                        p.append(temp)
                    ps = np.stack(p, axis=0) 
                ps_old = ps
                
                ### DCT: take low frequence ###
                if self.dct_factor == 1. :
                    pass 
                else:
                    ps = []
                    C, H, W = inp.shape
                    h_use, w_use = int(H/self.dct_factor), int(W/self.dct_factor)
                    for _ in range(N):
                        ori_signal = RGB_img_dct(ps_old[_])
                        p_signal = np.zeros_like(ps_old[_])
                        for c in range(self.n_channels):
                            p_signal[c,:h_use,:w_use] = ori_signal[c,:h_use,:w_use] 
                        p_img = RGB_signal_idct(p_signal)
                        ps.append(p_img)
                    ps = np.stack(ps, axis=0)

                ### DCT: take high frequence ###
#                 ps = []
#                 C, H, W = inp.shape
#                 h_use, w_use = int(H/self.dct_factor), int(W/self.dct_factor)
#                 for _ in range(N):
#                     ori_signal = RGB_img_dct(ps_old[_])
#                     p_signal = np.zeros_like(ps_old[_])
#                     for c in range(self.n_channels):
#                         ori_signal[c,:h_use,:w_use] = 0 #ori_signal[c,:h_use,:w_use] 
#                     p_img = RGB_signal_idct(ori_signal)
#                     ps.append(p_img)
#                 ps = np.stack(ps, axis=0)


                ### Orient all vectors toward tgt img ###
                ### a samll trick which is beneficial for GAN-related attack. ###
                ### you can also annotate it. ###
                diff = sample-target
                cos = nn.CosineSimilarity()
                cos_value = cos(torch.FloatTensor(diff).reshape(1,-1).expand(N,diff.size),torch.FloatTensor(ps).reshape(N,-1))
                cos_value[cos_value>=0]=1
                cos_value[cos_value<0]=-1
                cos_value = -cos_value.numpy().reshape(N,1,1,1)
                ps = cos_value * ps

        if self.preprocess is not None:
            rev_transp = np.argsort(transp)
            ps = ps * std
            ps = ps.transpose(0, *(rev_transp+1))
        return ps
    
    def calc_rho(self, gt, inp):
        return np.array([0.01])[0]
    
    def project(self, latent_Z):
        if self.preprocess is not None:
            transp, mean, std = self.preprocess
        ps = self.forward(latent_Z)

        if self.preprocess is not None:
            rev_transp = np.argsort(transp)
            ps = ps * std
            ps = ps.transpose(0, *(rev_transp+1))
        return ps
    
    
class PDiscriminator(nn.Module):

    def __init__(self,
                 depthScale0=512,
                 device='cuda',
                 initBiasToZero=True,
                 leakyReluLeak=0.2,
                 sizeDecisionLayer=1,
                 miniBatchNormalization=True,
                 dimInput=3,
                 equalizedlR=True):
        r"""
        Build a discriminator for a progressive GAN model
        Args:
            - depthScale0 (int): depth of the lowest resolution scales
            - initBiasToZero (bool): should we set the bias to zero when a
                                    new scale is added
            - leakyReluLeak (float): leakyness of the leaky relu activation
                                    function
            - decisionActivation: activation function of the decision layer. If
                                  None it will be the identity function.
                                  For the training stage, it's advised to set
                                  this parameter to None and handle the
                                  activation function in the loss criterion.
            - sizeDecisionLayer: size of the decision layer. Will typically be
                                 greater than 2 when ACGAN is involved
            - miniBatchNormalization: do we apply the mini-batch normalization
                                      at the last scale ?
            - dimInput (int): 3 (RGB input), 1 (grey-scale input)
        """
        super(PDiscriminator, self).__init__()
        self.device = device
        # Initialization paramneters
        self.initBiasToZero = initBiasToZero
        self.equalizedlR = equalizedlR
        self.dimInput = dimInput

        # Initalize the scales
        self.scalesDepth = [depthScale0]
        self.scaleLayers = nn.ModuleList()
        self.fromRGBLayers = nn.ModuleList()

        self.mergeLayers = nn.ModuleList()

        # Initialize the last layer
        self.initDecisionLayer(sizeDecisionLayer)

        # Layer 0
        self.groupScaleZero = nn.ModuleList()
        self.fromRGBLayers.append(EqualizedConv2d(dimInput, depthScale0, 1,
                          equalized=equalizedlR,
                          initBiasToZero=initBiasToZero))

        # Minibatch standard deviation
        dimEntryScale0 = depthScale0
        if miniBatchNormalization:
            dimEntryScale0 += 1

        self.miniBatchNormalization = miniBatchNormalization
        self.groupScaleZero.append(EqualizedConv2d(dimEntryScale0, depthScale0,
                           3, padding=1,
                           equalized=equalizedlR,
                           initBiasToZero=initBiasToZero))

        self.groupScaleZero.append(EqualizedLinear(depthScale0 * 16,
                           depthScale0,
                           equalized=equalizedlR,
                           initBiasToZero=initBiasToZero))

        # Initalize the upscaling parameters
        self.alpha = 0

        # Leaky relu activation
        self.leakyRelu = torch.nn.LeakyReLU(leakyReluLeak)
        self.to(self.device)
        
    def loadScale(self, depthScale=[256,256,128,64,32,16]):
        for scale in depthScale:
            self.addScale(scale)
        self.setNewAlpha(1)
        
    def addScale(self, depthNewScale):

        depthLastScale = self.scalesDepth[-1]
        self.scalesDepth.append(depthNewScale)

        self.scaleLayers.append(nn.ModuleList().to(self.device))

        self.scaleLayers[-1].append(EqualizedConv2d(depthNewScale,
                            depthNewScale,
                            3,
                            padding=1,
                            equalized=self.equalizedlR,
                            initBiasToZero=self.initBiasToZero).to(self.device))
        self.scaleLayers[-1].append(EqualizedConv2d(depthNewScale,
                            depthLastScale,
                            3,
                            padding=1,
                            equalized=self.equalizedlR,
                            initBiasToZero=self.initBiasToZero).to(self.device))
        
        self.fromRGBLayers.append(EqualizedConv2d(self.dimInput,
                          depthNewScale,
                          1,
                          equalized=self.equalizedlR,
                          initBiasToZero=self.initBiasToZero).to(self.device))

    def setNewAlpha(self, alpha):
        r"""
        Update the value of the merging factor alpha
        Args:
            - alpha (float): merging factor, must be in [0, 1]
        """

        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be in [0,1]")

        if not self.fromRGBLayers:
            raise AttributeError("Can't set an alpha layer if only the scale 0"
                                 "is defined")

        self.alpha = alpha

    def initDecisionLayer(self, sizeDecisionLayer):

        self.decisionLayer = EqualizedLinear(self.scalesDepth[0],
                         sizeDecisionLayer,
                         equalized=self.equalizedlR,
                         initBiasToZero=self.initBiasToZero)



    def forward(self, x, getFeature = False):
        if not torch.is_tensor(x):
            # print("GAN generator convert input to FloatTensor")
            x = torch.FloatTensor(x)
        if not x.is_cuda:
            x = x.to(self.device)
            
        # Alpha blending
        if self.alpha != 1 and len(self.fromRGBLayers) == 2:
            pad = nn.ReplicationPad2d(padding=(1, 1, 1, 1))
            temp = pad(x)
            y = F.avg_pool2d(temp, (2, 2))
            y = self.leakyRelu(self.fromRGBLayers[- 2](y))

        elif self.alpha != 1 and len(self.fromRGBLayers) > 2:
            y = F.avg_pool2d(x, (2, 2))
            y = self.leakyRelu(self.fromRGBLayers[- 2](y))

        # From RGB layer
        x = self.leakyRelu(self.fromRGBLayers[-1](x))

        # Caution: we must explore the layers group in reverse order !
        # Explore all scales before 0
        mergeLayer = self.alpha != 1 and len(self.fromRGBLayers) > 1
        shift = len(self.fromRGBLayers) - 2
        for groupLayer in reversed(self.scaleLayers):

            for layer in groupLayer:
                x = self.leakyRelu(layer(x))
            if x.shape[-1] == 7:
                pad = nn.ReplicationPad2d(padding=(1, 1, 1, 1))
                x = pad(x)
            x = nn.AvgPool2d((2, 2))(x)

            if mergeLayer:
                mergeLayer = False
                x = self.alpha * x + (1-self.alpha) * y

            shift -= 1

        # Now the scale 0

        # Minibatch standard deviation
        if self.miniBatchNormalization:
            x = miniBatchStdDev(x)

        x = self.leakyRelu(self.groupScaleZero[0](x))

        x = x.view(-1, num_flat_features(x))
        x = self.leakyRelu(self.groupScaleZero[1](x))

        out = self.decisionLayer(x)

        if not getFeature:
            return out

        return out, x

def calc_gradient_penalty(netD, real_data, fake_data,LAMBDA=10):
    # print "real_data: ", real_data.size(), fake_data.size()
    batchSize = real_data.size(0)
    alpha = torch.rand(batchSize, 1)
    alpha = alpha.expand(batchSize, int(real_data.nelement() /
                                        batchSize)).contiguous().view(
                                            real_data.size())
    alpha = alpha.to(real_data.device)
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = torch.autograd.Variable(
        interpolates, requires_grad=True)
    decisionInterpolate = netD(interpolates)
    decisionInterpolate = decisionInterpolate[:, 0].sum()

    gradients = torch.autograd.grad(outputs=decisionInterpolate,
                                    inputs=interpolates,
                                    create_graph=True, retain_graph=True)

    gradients = gradients[0].view(batchSize, -1)
    gradients = (gradients * gradients).sum(dim=1).sqrt()
    gradient_penalty = (((gradients - 1.0)**2)).sum() * LAMBDA

    gradient_penalty.backward(retain_graph=True)

    return gradient_penalty.item()