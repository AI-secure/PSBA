## Progressive-Scale Boundary Blackbox Attack via Projective Gradient Estimation

This repository contains the official code for the ICML 2021 paper:

["Progressive-Scale Boundary Blackbox Attack via Projective Gradient Estimation".](https://arxiv.org/abs/2106.06056)

Jiawei Zhang\*, Linyi Li\*, Huichen Li, Xiaolu Zhang, Shuang Yang, Bo Li

## Motivation

Boundary Blackbox Attack requires only decision labels to perform adversarial attacks, where query efficiency directly determines the attack efficiency. Therefore, how we estimate the gradient on the current boundary is a crucial step in this series of work.

In this paper, we theoretically show that there actually exist a trade-off between the projected length of the true gradient on subspace(the brown item) and the dimensionality of the projection subspace (purple item).

![projection](imgs\projection.png)

Based on this interesting finding, we propose *Progressive-Scale based projective Boundary Attack (PSBA)* via progressively searching for the optimal scale in a self-adaptive way under spatial, frequency, and spectrum scales. The image below just shows how we progressively search the optimal projection subspace on the spatial domain, and then attack the target models with this optimal scale.

![progressive_attack](imgs\progressive_attack.png)

## Environment Requirements ##

* GPU access (The pretrained models are on CUDA. Need extra modification to the code if no CUDA is available.)
* Python 3.5.2 (Python 3 should work in general.)
* PyTorch 1.5.1
* Torchvision 0.6.1
* Numpy 1.15.2
* Access to public image Datasets: ImageNet, CelebA, CIFAR10 and MNIST
* Face++ API access: please register an account at [Face++ website](https://www.faceplusplus.com/) to get the key and secret for access.


## Run with Pretrained Gradient Estimator ##
### Pretrained Models ###
We include some of the pretrained models that are necessary to reproduce a subset of the experimental results in the code repository, including the pretrained gradient estimators in ./src/gen_models and the pretrained target models in ./src/class_models.

Note that the pretrained models provided are only a subset of all the pretrained models. The reason is the size limit of the supplementary materials. We cannot upload all the model weights since some of them are too large (e.g., the generator of a PGAN model takes about 316 MB space each), so we only keep the pretrained models that are relatively small: the AE and VAE models in this case.

### Datasets ###
If you want to run experiments on the following datasets, you need to:
* ImageNet: need to change line 439 and 440 in attack_setting.py to the path to downloaded dataset
* CelebA: need to change line 482 to 487 in attack_setting.py (for offline experiments) and line 46-49 in main_api_attack.py (for online API experiments) to the path the dataset is downloaded
* CIFAR10 and MNIST: no need to change

### Offline Tasks ###
To run the offline attack experiments on ImageNet, CelebA, CIFAR10 or MNIST with pretrained models, example command:
`python main_pytorch_multi.py --use_gpu --TASK CelebA --pgen AE9408 --N_img 50`
or `python main_pytorch_multi.py --use_gpu --TASK mnist --pgen VAE9408 --N_img 50 --mnist_img_size 224`
For [Sign-OPT](https://github.com/cmhcbb/attackbox/blob/de692afc5db5c76b5d6a641c56df1d4dd052a463/attack/Sign_OPT.py):

`python main_pytorch_multi.py --use_gpu --TASK mnist --pgen sign_opt --N_img 50 --mnist_img_size 224 --stepsize_search fine_grained_binary_search`

For [Evolution Attack]( https://github.com/thu-ml/ares/blob/main/ares/attack/evolutionary_worker.py):

`python main_pytorch_multi.py --use_gpu --TASK mnist --pgen evolution --N_img 50 --mnist_img_size 224 --stepsize_search evolution_search`

Scale for PGAN:

1 => 7*7, 2 => 14\*14, ..., 6 => 224\*224

So the command for PGAN28:

`python main_pytorch_multi.py --use_gpu --TASK mnist --pgen PGAN9408 --device cuda:0 --N_img 50 --mnist_img_size 224 --scale 3 --model res18 --suffix s3 `

Frequency: The command for PGAN224d28 (28 comes from 224//8 = 24, so the dct_factor = 8):

`python main_pytorch_multi.py --use_gpu --TASK mnist --pgen PGAN9408 --device cuda:0 --N_img 50 --mnist_img_size 224 --scale 6 --dct_factor 8  --suffix d28` 

Spectrum: The command for PGAN224p400:

`python main_pytorch_multi.py --use_gpu --TASK mnist --pgen PGAN9408 --device cuda:0 --N_img 50 --mnist_img_size 224 --scale 6 --topk 400 --suffix p400 `

### Online API Experiments ###
We stored the randomly sampled pair IDs in api_results/src_tgt_pairs.npy
 so that if you want to reproduce the experimental results reported in the paper, run
`python main_api_attack.py --threshold 0.5 --use_gpu --pgen resize9408` and replace 'resize9408' with other methods like 'PGAN9408' etc. for other experiments.
Or you can leverage the code in helper.py to run the attacks sequentially to avoid manually running the commands one by one.

If you instead want to try a different set of source-target image pairs, then comment out line 45 and line 49 in main_api_attack.py and uncomment line 48. Then run the above commands.


## Run from scratch ##
If you don't want to use our pretrained estimators but instead want to train them from scratch yourself. Please follow the steps below.

For simplicity define command suffix cmd_suf as `--TASK imagenet` or `--TASK celeba` or `--TASK mnist --mnist_img_size 224` or `--TASK cifar10 --cifar10_img_size 224`. The choice depends on which image dataset is running.

### Preprocess Raw ImageNet Data ###
Please run `mkdir raw_data` in the main directory to create a new empty directory to store the raw datasets.

Change line 19 in `preprocess_data.py` to the root directory storing ImageNet dataset and then run
`python preprocess_data.py --TASK imagenet_preprocess`.

### Train Target and Reference Models ###
First we need both target and reference models.

`python train_celeba_model.py --do_train --do_test`

* Need to change line 18-21, line 67-70.

`python train_cifar10_model.py --cifar10_img_size 224 --do_train` and `python3 train_mnist_model.py --mnist_img_size 224 --do_train`

* No need to change

### Gradient Data Preparation ###
`python gradient_generate.py` with cmd_suf.

* Need to change root directory names in line 172 to line 175, line 209 to line 212.

`python preprocess_data.py` with cmd_suf.

* Need to change directory name in line 42, line 82-83, line 102-103, line 116.


### Train Estimators ###
`python train_pca_generator.py` with cmd_suf.

* Need to change line 14-17, 34-37.

`python train_vae_generator.py` with cmd_suf.

* Need to change line 60-63, 137-140.

`python train_ae_generator.py` with cmd_suf.

* Need to change line 56-59, line 131-134, line 218-223.

`train_pgan_generator.py`

* Need to change line 64-67.

## Theoretical Validation

### Long Tail Distribution

+ the implementation is shown on the `Long_Tail_Distribution.ipynb`

### The Sensitivity of PGAN

+ Need to comment off the code between line 285-287

`src/models/pgan_generator.py`

+ Need to comment off the code between line 395-406

`src/foolbox/attacks/bapp_custom.py`

+ if you want to deliberately ajust the sensitivity then you need to comment off the code between line 761-767

`src/foolbox/attacks/bapp_custom.py`
