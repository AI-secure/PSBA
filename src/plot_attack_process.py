import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def plot(PGENs, TASK, root_dir, npz_name, src_image, tgt_image, STEPS):
    # Process
    n_pgens = len(PGENs)
    if True:
        N = len(STEPS)
        fig = plt.figure(figsize=(9, 12))
        plt.subplot(n_pgens, N + 1, 1)
        plt.subplots_adjust(wspace=0.0, hspace=0.20)
        plt.imshow(src_image)
        plt.xlabel('Source Image')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(n_pgens, N + 1, (n_pgens - 1) * (N + 1) + 1)
        plt.imshow(tgt_image)
        plt.xlabel('Target Image')
        plt.xticks([])
        plt.yticks([])
        for pid, PGEN in enumerate(PGENs):
            for i, step in enumerate(STEPS):
                ax = fig.add_subplot(n_pgens, N + 1, pid * (N + 1) + i + 2)
                # data = np.load('%s/perturbed__facepp_%s_%d_%d_%d.npz' % (root_dir, PGEN, src_id, tgt_id, step))
                npz_name_i = npz_name%(PGEN, step)
                data = np.load('%s/perturbed_%s.npz' % (root_dir, npz_name_i))
                plt.imshow(data['pert'])
                plt.xticks([])
                plt.yticks([])
                if i == N - 1:
                    ax.yaxis.set_label_position("right")
                    plt.ylabel(PGEN_NAMEs[pid])
                if (pid == (n_pgens - 1)):
                    plt.xlabel('d=%.2e\n#q=%d' % (data['info'][1], data['info'][0]))
                else:
                    plt.xlabel('d=%.2e' % data['info'][1])
        fig.savefig('plots/%s_attack_process.pdf' % (TASK), bbox_inches='tight')

    # Diff
    # if False:
    #     fig = plt.figure(figsize=(20, 5))
    #     for pid, PGEN in enumerate(PGENs):
    #         plt.subplot(1, 4, pid + 1)
    #         data = np.load('steps/perturbed%s99.npz' % (PGEN))
    #         plt.imshow(5 * np.abs(data['pert'] - tgt_image))
    #         plt.xticks([])
    #         plt.yticks([])
    #     plt.show()
    #     fig.savefig('diff.pdf')


def plot_grey(PGENs, TASK, root_dir, npz_name, src_image, tgt_image, STEPS):
    mnist_img_shape = (224, 224)
    n_pgens = len(PGENs)
    N = len(STEPS)
    fig = plt.figure(figsize=(9, 12))
    plt.subplot(n_pgens, N + 1, 1)
    plt.subplots_adjust(wspace=0.0, hspace=0.20)
    plt.imshow(src_image.reshape(mnist_img_shape), cmap = 'gray')
    plt.xlabel('Source Image')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(n_pgens, N + 1, (n_pgens - 1) * (N + 1) + 1)
    plt.imshow(tgt_image.reshape(mnist_img_shape), cmap = 'gray')
    plt.xlabel('Target Image')
    plt.xticks([])
    plt.yticks([])
    for pid, PGEN in enumerate(PGENs):
        for i, step in enumerate(STEPS):
            ax = fig.add_subplot(n_pgens, N + 1, pid * (N + 1) + i + 2)
            # data = np.load('%s/perturbed__facepp_%s_%d_%d_%d.npz' % (root_dir, PGEN, src_id, tgt_id, step))
            npz_name_i = npz_name % (PGEN, step)
            data = np.load('%s/perturbed_%s.npz' % (root_dir, npz_name_i))
            plt.imshow(data['pert'].reshape(mnist_img_shape), cmap = 'gray')
            plt.xticks([])
            plt.yticks([])
            if i == N - 1:
                ax.yaxis.set_label_position("right")
                plt.ylabel(PGEN_NAMEs[pid])
            if (pid == (n_pgens - 1)):
                plt.xlabel('d=%.2e\n#q=%d' % (data['info'][1], data['info'][0]))
            else:
                plt.xlabel('d=%.2e' % data['info'][1])
    fig.savefig('plots/%s_attack_process.pdf' % (TASK), bbox_inches='tight')


if __name__ == '__main__':
    TASK = 'mnist'
    pipeline_name = 'NLBA'
    PGEN_NAMEs = ['HSJA', 'QEBA-S', 'QEBA-F', 'QEBA-I', '%s-AE' % (pipeline_name), '%s-VAE' % (pipeline_name),
                  '%s-GAN' % (pipeline_name)]
    root_dir = "BAPP_result/steps/%s" % (TASK)

    if TASK == 'facepp':
        PGENs = ['naive', 'resize9408', 'DCT9408', 'PCA9408', 'AE9408', 'VAE9408', 'GAN9408']
        src_id = 163922
        tgt_id = 80037
        npz_name = "_facepp_%s_%d_%d_%s" %('%s', src_id, tgt_id, '%d')
        STEPS = (1, 5, 10, 15, 20)

        src_path = "BAPP_result/steps/original/%06d.jpg"% (src_id)
        src_image = Image.open(src_path).convert("RGB").resize((224, 224))
        tgt_path = "BAPP_result/steps/original/%06d.jpg" %(tgt_id)
        tgt_image = Image.open(tgt_path).convert("RGB").resize((224, 224))

    elif TASK == 'imagenet' or TASK == 'celeba':
        PGENs = ['naive', 'resize9408', 'DCT9408', 'PCA9408', 'AE9408', 'VAE9408', 'GAN9408']

        npz_name = "%s_%s_casestudy_%s" %(TASK, '%s', '%d')
        STEPS = (0, 10, 50, 100, 200)

        src_image = np.load('BAPP_result/steps/original/src_img_0_%s_naive_casestudy.npy' %(TASK))
        src_image = src_image.transpose((1, 2, 0))
        tgt_image = np.load('BAPP_result/steps/original/tgt_img_0_%s_naive_casestudy.npy' %(TASK))
        tgt_image = tgt_image.transpose((1, 2, 0))

    elif TASK == 'cifar10':
        PGENs = ['naive', 'resize9408', 'DCT9408', 'PCA9408', 'AE9408', 'VAE9408', 'DCGAN']

        npz_name = "%s_%s_casestudy_%s" %(TASK, '%s', '%d')
        STEPS = (0, 10, 50, 100, 200)

        src_image = np.load('BAPP_result/steps/original/src_img_0_%s_naive_casestudy.npy' %(TASK))
        src_image = src_image.transpose((1, 2, 0))
        tgt_image = np.load('BAPP_result/steps/original/tgt_img_0_%s_naive_casestudy.npy' %(TASK))
        tgt_image = tgt_image.transpose((1, 2, 0))

    elif TASK == 'mnist':
        PGENs = ['naive', 'resize9408', 'DCT9408', 'PCA9408', 'AE9408', 'VAE9408', 'DCGAN']

        npz_name = "%s_%s_casestudy_%s" % (TASK, '%s', '%d')
        STEPS = (0, 10, 50, 100, 200)

        src_image = np.load('BAPP_result/steps/original/src_img_0_%s_naive_casestudy.npy' % (TASK))
        tgt_image = np.load('BAPP_result/steps/original/tgt_img_0_%s_naive_casestudy.npy' % (TASK))

    else:
        print("Not implemented")
        assert 0

    if TASK != 'mnist':
        plot(PGENs, TASK, root_dir, npz_name, src_image, tgt_image, STEPS)
    else:
        plot_grey(PGENs, TASK, root_dir, npz_name, src_image, tgt_image, STEPS)


