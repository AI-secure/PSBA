import numpy as np
# import torchvision
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
from models import ExpCosGenerator
import math
import utils
import argparse
import matplotlib.pyplot as plt


def calc_cos_sim(x1, x2, dim=1):
    cos = (x1 * x2).sum(dim) / np.sqrt((x1 ** 2).sum(dim) * (x2 ** 2).sum(dim))
    return cos


def epoch_train(REF, model, optimizer, BATCH_SIZE=32, mounted=False):
    # N_train = 8750
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
    elif TASK == 'celebaid':
        N_train = 9
        if REF == 'rnd':
            N_train = 5 * 9
    else:
        print("Task not implemented")
        assert 0

    # if TASK == 'celeba' or TASK == 'imagenet':
    #     N_train=2000

    if mounted:
        data_path = '../raw_data/%s_%s/train_batch_%d.npy'
    else:
        data_path = '../raw_data/%s_%s/train_batch_%d.npy'
    # test: 625; train: 8750
    # ../raw_data/imagenet_avg/train_batch_%d.npy

    # N_used = 200
    N_used = N_train
    model.train()
    perm = np.random.permutation(N_used)
    tot_num = 0.0
    cum_loss = 0.0
    cum_cos = 0.0
    with tqdm(perm) as pbar:
        for idx in pbar:
            X = np.load(data_path % (TASK, REF, idx))
            X = X / np.sqrt((X ** 2).sum(1, keepdims=True))

            B = X.shape[0]
            assert B <= BATCH_SIZE
            X = np.reshape(X, (B, 3, 224, 224))

            rv = np.random.randn(B, *ENC_SHAPE)
            rv = rv / np.sqrt((rv ** 2).sum((1, 2, 3), keepdims=True))
            rv_var = torch.FloatTensor(rv)
            if GPU:
                rv_var = rv_var.cuda()
            X_dec = model(rv_var)

            l = model.loss(X_dec, X)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            cos_sim = calc_cos_sim(X.reshape(B, -1), X_dec.cpu().detach().numpy().reshape(B, -1), dim=1)
            if math.isnan(cos_sim.any()):
                print(X)
                print(X_dec)
                assert 0

            cum_loss += l.item() * B
            cum_cos += cos_sim.sum()
            tot_num += B
            pbar.set_description(
                "Cur loss = %.6f; Avg loss = %.6f; Avg cos = %.4f" % (l.item(), cum_loss / tot_num, cum_cos / tot_num))

    return cum_loss / tot_num, cum_cos / tot_num


def epoch_eval(REF, model, BATCH_SIZE=32, mounted=False):
    if TASK == 'celeba':
        N_test = 624
        if REF == 'rnd':
            N_test = 625
    elif TASK == 'imagenet':
        N_test = 625
        if REF == 'rnd':
            N_test = 625
    elif TASK == 'celebaid':
        N_test = 3
        if REF == 'rnd':
            N_test = 5 * 3
    else:
        print("Not implemented")
        assert False

    # if TASK == 'celeba' or TASK == 'imagenet':
    #     N_test=50

    if mounted:
        data_path = '../raw_data/%s_%s/test_batch_%d.npy'
    else:
        data_path = '../raw_data/%s_%s/test_batch_%d.npy'

    # N_used = 50
    N_used = N_test
    model.eval()
    perm = np.random.permutation(N_used)
    tot_num = 0.0
    cum_loss = 0.0
    cum_cos = 0.0
    with tqdm(perm) as pbar:
        for idx in pbar:
            X = np.load(data_path % (TASK, REF, idx))
            X = X / np.sqrt((X ** 2).sum(1, keepdims=True))

            B = X.shape[0]
            assert B <= BATCH_SIZE
            X = np.reshape(X, (B, 3, 224, 224))
            with torch.no_grad():
                rv = np.random.randn(B, *ENC_SHAPE)
                rv = rv / np.sqrt((rv ** 2).sum((1, 2, 3), keepdims=True))
                rv_var = torch.FloatTensor(rv)
                if GPU:
                    rv_var = rv_var.cuda()
                X_dec = model(rv_var)
                l = model.loss(X_dec, X)

            cos_sim = calc_cos_sim(X.reshape(B, -1), X_dec.cpu().detach().numpy().reshape(B, -1), dim=1)
            cum_loss += l.item() * B
            cum_cos += cos_sim.sum()
            tot_num += B
            pbar.set_description("Avg loss = %.6f; Avg cos = %.4f" % (cum_loss / tot_num, cum_cos / tot_num))

    return cum_loss / tot_num, cum_cos / tot_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--TASK', type=str)
    parser.add_argument('--N_Z', type=int)
    parser.add_argument('--mounted', action='store_true')
    parser.add_argument('--lmbd', default=0.05, type=float) # for expcos only
    args = parser.parse_args()

    GPU = True
    # TASK = 'celeba'
    TASK = args.TASK

    N_Z = args.N_Z

    model = ExpCosGenerator(n_channels=3, gpu=GPU, N_Z=N_Z, lmbd=args.lmbd)

    # fixed_noise = torch.FloatTensor(10, N_Z).normal_()
    if N_Z == 128:
        ENC_SHAPE = (8, 4, 4)
    elif N_Z == 9408:
        ENC_SHAPE = (48, 14, 14)
    else:
        print("Not implemented")
        assert 0
    fixed_noise = np.random.randn(10, *ENC_SHAPE)
    fixed_noise = fixed_noise / np.sqrt((fixed_noise ** 2).sum((1, 2, 3), keepdims=True))
    fixed_noise = torch.FloatTensor(fixed_noise)
    if GPU:
        fixed_noise = fixed_noise.cuda()

    # inp = np.ones((8,3,224,224))
    # X_enc, X_dec = model(inp)
    # print (X_enc.shape)
    # print (X_dec.shape)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # plot training data sample
    if not args.mounted:
        utils.plot_training_data(data_path='../raw_data/%s_%s/train_batch_%d.npy', TASK=TASK, REF='dense121', idx=0)

    # REFs = ['dense121']#, 'res50', 'vgg16', 'googlenet', 'wideresnet'] # TODO: change back
    # REFs = ['dense121', 'res50', 'vgg16', 'googlenet', 'wideresnet']
    for _ in range(100):
        fake_dec = model.forward(fixed_noise)
        # if _ % 10 == 1:
        #     print(fake_dec.shape)
        #     print(fake_dec[0])

        fig = plt.figure(figsize=(20, 8))
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            to_plt = fake_dec[i].detach().cpu().numpy().transpose(1, 2, 0)
            to_plt = (to_plt - to_plt.min()) / (to_plt.max() - to_plt.min())
            # if _ % 10 == 1:
            #     print(i)
            #     print(to_plt)
            plt.imshow(to_plt)
        plt.savefig('./plots/%s_gradient_expcos%d_eg-%d.pdf' % (TASK, N_Z, _))
        plt.close(fig)

        # for REF in np.random.permutation(REFs):
        # for REF in REFs:
        #     print(REF)
        #     print (epoch_train(REF=REF, model=model, optimizer=optimizer))
        #     print (epoch_eval(REF=REF, model=model))
        #     # torch.save(model.state_dict(), 'expcos_generator.model')
        #     torch.save(model.state_dict(), './gen_models/%s_gradient_expcos_%d_generator.model' %(TASK, N_Z))

        print(epoch_train('rnd', model, optimizer, mounted=args.mounted))
        print(epoch_eval('rnd', model, mounted=args.mounted))
        torch.save(model.state_dict(), './gen_models/%s_gradient_expcos_%d_generator.model' % (TASK, N_Z))

        torch.save(model.state_dict(), './gen_models/%s_gradient_expcos_%d_generator_%d.model' % (TASK, N_Z, _))