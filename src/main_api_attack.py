import foolbox
from foolbox.criteria import TargetClass
import numpy as np
import json
import argparse
import attack_setting
import random


size_224 = 224, 224


def load_img(path):
    from PIL import Image
    image = Image.open(path)
    image = image.resize(size_224)
    image = np.asarray(image, dtype=np.float32)
    image = image[:, :, :3]
    image = image / 255
    image = image.transpose(2, 0, 1) # from [224, 224, 3] to [3, 224, 224]
    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', type=str, default='facepp')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--use_gpu', action='store_true')
    # parser.add_argument('--gen', type=int, default=0)
    parser.add_argument('--pgen', type=str)
    # parser.add_argument('--src', type=str)
    # parser.add_argument('--tgt', type=str)
    parser.add_argument('--mounted', action='store_true')
    parser.add_argument('--account', default=0, type=int) # the api account
    parser.add_argument('--dct_factor', type=float, default=1.)
    parser.add_argument('--topk', type=int, default=0)
    parser.add_argument('--scale', type=int, default=6)
    parser.add_argument('--reduction', type=int, default=224,help = 'the reduction dimension for EA attack' ) 
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--stepsize_search', type=str, default = 'geometric_progression', help = "for EA: the stepsize_search is 'evolution_search'; for sign-opt: the stepsize_search is 'fine_grained_binary_search'; for other methods: the stepsize_search is 'geometric_progression'")
    args = parser.parse_args()
    TASK = 'imagenet'
    args.TASK = TASK

    N_pair = 50
    if args.mounted:
        root_dir = '../raw_data/celebA/img_align_celeba'
    else:
        root_dir = '../raw_data/celebA/img_align_celeba'

    pairs = np.load('./api_results/src_tgt_pairs.npy')
    # while(N_pair):
    for i in range(N_pair):
        # src_id, tgt_id = random.sample(range(0, 202599+1), 2)
        src_id, tgt_id = pairs[i]

        src_img_path = '%s/%06d.jpg' %(root_dir, src_id)
        tgt_img_path = '%s/%06d.jpg' %(root_dir, tgt_id)
        fmodel = foolbox.models.FacePlusPlusModel(bounds=(0, 1), src_img_path=src_img_path, suffix=args.suffix,
                                                  account=args.account, channel_axis=0, simi_threshold=args.threshold)

        src_image = load_img(src_img_path)
        tgt_image = load_img(tgt_img_path)
        src_label = np.argmax(fmodel.forward_one(src_image))
        tgt_label = np.argmax(fmodel.forward_one(tgt_image))
        if src_label == tgt_label:
            print("src tgt same prediction", src_id, tgt_id)
            continue
        # if src_label != tgt_label:
        #     N_pair -= 1
        # else:
        #     continue
        # print (src_image.shape)
        # print (tgt_image.shape)
        print("======== # %d, src id %d, tgt id %d, src label %d, tgt label %d ========"%(i, src_id, tgt_id, src_label, tgt_label))
        # print ("Source Image Label:", src_label)
        # print ("Target Image Label:", tgt_label)

        mask = None
        p_gen = attack_setting.load_pgen(task=TASK, pgen_type=args.pgen, args=args)
        
        if args.pgen == 'evolution':
            ITER = 2000
            initN = 1
            maxN = 1
        else:
            ITER =20
            initN = 100
            maxN = 100
            
        attack = foolbox.attacks.BAPP_custom(fmodel, criterion=TargetClass(src_label))
        adv = attack(tgt_image, tgt_label, starting_point = src_image, iterations=ITER, batch_size=9999,
                     stepsize_search=args.stepsize_search, verbose=True, unpack=False, max_num_evals=maxN,
                     initial_num_evals=initN, internal_dtype=np.float32, rv_generator = p_gen, atk_level=999, mask=mask,
                     discretize=True, suffix='_%s_%s_%d_%d'%(args.suffix,args.pgen, src_id, tgt_id))

        print (attack.logger)
        with open('BAPP_result/api_attack_%s_%s_%d_%d.log'%(args.suffix, args.pgen, src_id, tgt_id), 'w') as outf:
            json.dump(attack.logger, outf)


