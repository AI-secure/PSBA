import sys
import numpy as np
# import torchvision
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import utils
from models import PGenerator, PDiscriminator,calc_gradient_penalty
import matplotlib.pyplot as plt
import argparse
import model_settings
from torch import autograd
torch.backends.cudnn.benchmark = True
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# def epoch_train(REF, mean_, std_, modelD, modelG, optimizerD, optimizerG, N_Z, BATCH_SIZE=32, D_ITERS=1, G_ITERS=1, outf=None):
def epoch_train(REF, modelD, modelG, optimizerD, optimizerG, N_Z, BATCH_SIZE=32, D_ITERS=1, G_ITERS=1, outf=None, mounted=False,upepoch = 0):
    
    # if your gpu memory is great enough, you can change the BATCH_SIZE to 128 for quicker training,
    if TASK == 'celeba':
        # celeba train: 5087, test: 624
        N_train = 5087
        if REF == 'rnd':
            N_train = 15625
    elif TASK == 'imagenet':
        # test: 625; train: 8750
        N_train = 8750
        if REF == 'rnd':
            N_train = 15625
    elif TASK == 'celebaid' or TASK == 'celeba2':
        N_train = 9
        if REF == 'rnd':
            N_train = 5*9
    elif TASK == 'dogcat2':
        N_train = 625
        if REF == 'rnd':
            N_train = 5*625

    # elif TASK == 'mnist_224':
    #     N_train = 1875
    #     if REF == 'rnd':
    #         N_train = 5*1875
    # elif TASK == 'cifar10_224':
    #     N_train = 1563
    #     if REF == 'rnd':
    #         N_train = 5*1563

    elif TASK.startswith('mnist'):
        N_train = 1875
        if REF == 'rnd':
            N_train = 5*1875
    elif TASK.startswith('cifar10'):
        N_train = 1563
        if REF == 'rnd':
            N_train = 5*1563

    else:
        print("Not implemented")
        assert 0
    if mounted:
        data_path = '../raw_data/%s_%s/train_batch_%d.npy'
    else:
        data_path = '../raw_data/%s_%s/train_batch_%d.npy'
        
    N_used = N_train
    # N_used = 200
    epoch = pepoch[upepoch]
    total_iter = epoch
    perm = np.random.permutation(N_used)
    modelD.train()
    modelG.train()
    real_label = 1.0
    fake_label = 0.0
    global cur_iter
    total_iter = N_used * pepoch[upepoch] - 1
    with tqdm(perm) as pbar:
        for idx in pbar:
            temp = data_path % (TASK, REF, idx)
            data = np.load(temp)
            data = data/np.max(np.abs(data),keepdims=True,axis=1)
            cur_iter += 1
            B = data.shape[0]
            assert B <= BATCH_SIZE
            X = np.reshape(X, (B, n_channels, 224, 224))
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            alpha = cur_iter/total_iter if upepoch !=0 else 1
            if upepoch!=0:
                modelD.module.setNewAlpha(alpha)
            optimizerD.zero_grad()
            if not torch.is_tensor(X):
                X_tensor = torch.FloatTensor(X)
            # downscale the orginal gradient images for fitting current trainging
            for k in range(6-upepoch):
                if X_tensor.shape[-1] == 7:
                    pad = nn.ReplicationPad2d(padding=(1, 1, 1, 1))
                    X_tensor = pad(X_tensor)
                    X_tensor = F.avg_pool2d(X_tensor, (2, 2))
                else :
                    X_tensor = F.avg_pool2d(X_tensor, (2, 2))
            X_tensor = X_tensor.numpy()/np.max(np.abs(X_tensor.numpy()),keepdims=True,axis=(1,2,3))
            X_tensor = torch.FloatTensor(X_tensor)
            real_cpu = X_tensor.to(device)
            predRealD = modelD(real_cpu)
            output_r = - predRealD.sum()
            output_r.backward(retain_graph=True)
            D_x = - output_r.item()

            # train with fake
            if upepoch != 0:
                modelG.module.setNewAlpha(alpha)
            noise = torch.randn(B, N_Z)
            fake = modelG(noise)
            output_f = modelD(fake.detach()).sum()
            D_G_z1 = output_f.item()
            output_f.backward()
            gradient_penalty = calc_gradient_penalty(modelD, real_cpu.data, fake.data)

            lossEpsilon = (predRealD[:, 0] ** 2).sum() * 0.001
            lossEpsilon.backward()
            optimizerD.step()
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ############################
            optimizerG.zero_grad()
            output = - modelD(fake).sum()
            output.backward()
            D_G_z2 = - output.item()
            optimizerG.step()
            pbar.set_description('Upepoch %d, Epoch %d,Alpha: %.4f D(x): %.4f D(G(z)): %.4f / %.4f,gradient_penalty : %.4f'
                                 %(upepoch,_,alpha, D_x, D_G_z1, D_G_z2,gradient_penalty))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--TASK', type=str)
    parser.add_argument('--N_Z', default=9408, type=int)
    parser.add_argument('--mounted', action='store_true')

    parser.add_argument('--mnist_img_size', type=int, default=224)
    parser.add_argument('--mnist_padding_size', type=int, default=0)
    parser.add_argument('--mnist_padding_first', action='store_true')

    parser.add_argument('--cifar10_img_size', type=int, default=224)
    parser.add_argument('--cifar10_padding_size', type=int, default=0)
    parser.add_argument('--cifar10_padding_first', action='store_true')

    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    

    TASK=args.TASK
    device = args.device
    GPU = True
    if TASK == 'mnist' or TASK == 'cifar10':
        TASK = model_settings.get_model_file_name(TASK, args) 
    model_file_name = TASK

    if TASK.startswith('mnist'):
        n_channels = 1
    else:
        n_channels = 3

    N_Z = args.N_Z
    # N_Z = 9408
    
    modelD = PDiscriminator(dimInput=n_channels,device=args.device)
    modelG = PGenerator(dimLatent=N_Z, n_channels=n_channels, device=args.device)

    if args.pretrained:
        # training starts from specific stage, here is 112 × 112.
        modelD.loadScale([256,256,128,64,32])
        modelG.loadScale([256,256,128,64,32])
        modelD.load_state_dict(torch.load('./gen_models/%s_gradient_pgan_%d_discriminator_u5.model' % (TASK, N_Z),map_location = args.device))
        modelG.load_state_dict(torch.load('./gen_models/%s_gradient_pgan_%d_generator_u5.model' % (TASK, N_Z),map_location = args.device))
        modelD.setNewAlpha(0)
        modelD.addScale(16)
        modelG.setNewAlpha(0)
        modelG.addScale(16)

    modelD = nn.DataParallel(modelD)
    modelG = nn.DataParallel(modelG)
    lr = 2e-4
    fixed_noise = torch.FloatTensor(10, N_Z).normal_(0, 1)

    pepoch = [50,50,50,50,50,50,50]
    depthScale=[256,256,128,64,32,16]
    for upepoch in range(len(pepoch)):
    	# if args.pretrained, you should modified it a bit to strat from specific stage.
        current_size = modelG.module.getOutputSize()
        print("Current output size:",current_size )
        optimizerD = optim.Adam(modelD.parameters(), lr=lr, betas=(0, 0.99))
        optimizerG = optim.Adam(modelG.parameters(), lr=lr, betas=(0, 0.99))
        cur_iter = -1
        for _ in range(pepoch[upepoch]):
            if _ % 5 == 0:
                with torch.no_grad():
                    fake = modelG(fixed_noise)
                fig = plt.figure(figsize=(20, 8))
                # for i in range(10):
                #     plt.subplot(2, 5, i + 1)
                #     to_plt = fake[i].detach().cpu().numpy().transpose(1, 2, 0)
                #     to_plt = (to_plt - to_plt.min()) / (to_plt.max() - to_plt.min())
                #     plt.imshow(to_plt)
                for i in range(10):
    #                 if i<5:
                    if n_channels == 3:
                        plt.subplot(2, 5, i + 1)
                        to_plt = fake[i].detach().cpu().numpy().transpose(1, 2, 0)
                        to_plt = (to_plt - to_plt.min()) / (to_plt.max() - to_plt.min())
                        plt.imshow(to_plt)
                    else:
                        plt.subplot(2, 5, i + 1)
                        x = fake[i].detach().cpu().squeeze().numpy()
                        to_plt = (x - x.min()) / (x.max() - x.min())
                        plt.imshow(to_plt, cmap='gray')
    #                     plt.subplot(2, 5, i + 1)
    #                     to_plt = wdata[i-5].transpose(1, 2, 0)
    #                     to_plt = (to_plt - to_plt.min()) / (to_plt.max() - to_plt.min())
                plt.savefig('./plots/%s_gradient_pgan%d_up-%d_eg-%d.pdf' % (model_file_name, N_Z,upepoch, _))
                plt.close(fig)

            epoch_train(REF='rnd', modelD=modelD, modelG=modelG, optimizerD=optimizerD, optimizerG=optimizerG, N_Z=N_Z, outf=None, mounted=args.mounted,upepoch=upepoch)
        # u1 means 7×7, u2 means 14×14, ... , u6 means 224×224.
        torch.save(modelD.module.state_dict(), './gen_models/%s_gradient_pgan_%d_discriminator_u%d.model' % (model_file_name, N_Z,upepoch))
        torch.save(modelG.module.state_dict(), './gen_models/%s_gradient_pgan_%d_generator_u%d.model' % (model_file_name, N_Z,upepoch))
        if upepoch != len(pepoch) - 1:
	        modelG.module.addScale(depthScale[upepoch])
	        modelG.module.setNewAlpha(0)
	        modelD.module.addScale(depthScale[upepoch])
	        modelD.module.setNewAlpha(0)