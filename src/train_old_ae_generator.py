import numpy as np
# import torchvision
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
from models import OldAEGenerator
import matplotlib.pyplot as plt
import argparse
import utils


def calc_cos_sim(x1, x2, dim=1):
    cos = (x1 * x2).sum(dim) / np.sqrt((x1 ** 2).sum(dim) * (x2 ** 2).sum(dim))
    return cos


def epoch_train(REF, model, optimizer, BATCH_SIZE=32, mounted=False):
    # N_train = 8750
    if TASK == 'celeba':
        # celeba train: 5087, test: 624
        N_train = 5087
        # N_train = 500 # TODO: change back
    elif TASK == 'imagenet':
        # test: 625; train: 8750
        N_train = 8750
        # N_train = 500
    else:
        print("Task not implemented")
        assert 0

    if REF == 'rnd':
        # N_train = 15625
        if args.small:
            N_train = 5000
        else:
            N_train = 15625

    # data_path = '../raw_data/imagenet_res50/train_batch_%d.npy'
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
            X = np.load(data_path%(TASK, REF, idx))
            X = X / np.sqrt((X**2).sum(1, keepdims=True))
            # X = utils.regularize(X)

            B = X.shape[0]
            assert B <= BATCH_SIZE
            X = np.reshape(X, (B, 3, 224, 224))
            X_enc, X_dec = model(X)
            l = model.loss(X_dec, X)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            cos_sim = calc_cos_sim(X.reshape(B, -1), X_dec.cpu().detach().numpy().reshape(B, -1), dim=1)
            cum_loss += l.item() * B
            cum_cos += cos_sim.sum()
            tot_num += B
            pbar.set_description(
                "Cur loss = %.6f; Avg loss = %.6f; Avg cos = %.4f" % (l.item(), cum_loss / tot_num, cum_cos / tot_num))

    return cum_loss / tot_num, cum_cos / tot_num


def epoch_eval(REF, model, BATCH_SIZE=32, mounted=False):
    # N_test = 625
    if TASK == 'celeba':
        N_test = 624
        # N_test = 50 # TODO: change back
    elif TASK == 'imagenet':
        N_test = 625
        # N_test = 50
    else:
        print("Not implemented")
        assert False

    if REF == 'rnd':
        # N_test = 625
        if args.small:
            N_test = 100
        else:
            N_test = 625
    # data_path = '../raw_data/imagenet_res50/test_batch_%d.npy'
    if mounted:
        data_path = '../raw_data/data/%s_%s/test_batch_%d.npy'
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
            X = np.load(data_path%(TASK, REF, idx))
            X = X / np.sqrt((X**2).sum(1, keepdims=True))
            # X = utils.regularize(X)

            B = X.shape[0]
            assert B <= BATCH_SIZE
            X = np.reshape(X, (B, 3, 224, 224))
            with torch.no_grad():
                X_enc, X_dec = model(X)
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
    parser.add_argument('--mounted', action='store_true')
    parser.add_argument('--small', action='store_true')
    args = parser.parse_args()

    TASK = args.TASK

    GPU = True

    model = OldAEGenerator(n_channels=3, gpu=GPU)
    # inp = np.ones((8,3,224,224))
    # X_enc, X_dec = model(inp)
    # print (X_enc.shape)
    # print (X_dec.shape)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    fixed_noise = np.random.randn(10, 48, 14, 14)
    fixed_noise = fixed_noise / np.sqrt((fixed_noise ** 2).sum((1, 2, 3), keepdims=True))
    fixed_noise = torch.FloatTensor(fixed_noise)
    if GPU:
        fixed_noise = fixed_noise.cuda()
    for _ in range(100):
        fake_dec = model.decode(fixed_noise)
        if _ % 10 == 1:
            print(fake_dec.shape)
            print(fake_dec[0])

        fig = plt.figure(figsize=(20, 8))
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            to_plt = fake_dec[i].detach().cpu().numpy().transpose(1, 2, 0)
            to_plt = (to_plt - to_plt.min()) / (to_plt.max() - to_plt.min())
            if _ % 10 == 1:
                print(i)
                print(to_plt)
            plt.imshow(to_plt)
        if args.small:
            plt.savefig('./plots/%s_gradient_old_ae%d_eg-%d_small.pdf' % (TASK, 9408, _))
        else:
            plt.savefig('./plots/%s_gradient_old_ae%d_eg-%d.pdf' % (TASK, 9408, _))
        plt.close(fig)

        print(epoch_train('rnd', model, optimizer, mounted=args.mounted))
        print(epoch_eval('rnd', model, mounted=args.mounted))
        # torch.save(model.state_dict(), 'ae_generator.model')
        if args.small:
            torch.save(model.state_dict(), './gen_models/%s_gradient_old_ae_%d_generator_small.model' %(TASK, 9408))
        else:
            torch.save(model.state_dict(), './gen_models/%s_gradient_old_ae_%d_generator.model' % (TASK, 9408))
            torch.save(model.state_dict(), './gen_models/%s_gradient_old_ae_%d_generator_%d.model' % (TASK, 9408, _))

