import argparse
import logging
import time
import datetime
import sys
sys.path.append('../')
import copy
import numpy as np
import os
# import shutil
import random
import warnings
# import xlwt
# import dill as pickle
import json
# import scipy
from scipy.stats import t

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, ConcatDataset
from torch.autograd import Variable
#from check_dataset import check_dataset
#from check_model import check_model
from utils.utils import AverageMeter, accuracy, set_logging_config, add_noise_cifar_w
from utils.cifar100_dataloader import get_permute_cifar100, get_inheritable_cifar100, get_val_cifar100
from utils.imagenetDataloader import getDataloader_imagenet_inheritable
from models.engine import train_one_epoch, evaluate, compute_mean_attention_dist, load_image_from_path

import models.distiller as distiller
import models.load_settings as load_settings
from models.utils import distLinear as distLinear
from utils.datasets import build_dataset, build_transform
from utils.utils import get_world_size, get_rank, init_distributed_mode, save_on_master, is_main_process, _load_checkpoint_for_ema

import torch.backends.cudnn as cudnn
from timm.data import Mixup
try:
    from timm.data import DatasetTar
except ImportError:
    # for higher version of timm
    from timm.data import ImageDataset as DatasetTar
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from sklearn.manifold import TSNE

torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser(description='featuredistill')

# Model parameters
parser.add_argument('--batchSize', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 10)')

parser.add_argument('--weight_decay', type=float, default=0.05, metavar='LR', help='weight decay (default: 0.05)') 
#parser.add_argument('--lr_drop', type=float, default=0.4)
#parser.add_argument('--epochs_drop', type=int, default=110)
#parser.add_argument('--print_freq', type=int, default=50)
#parser.add_argument('--paper_setting', default='vit-dino', type=str)
#parser.add_argument('--method', type=str, default='meta-learngene', help='different learngene')
#parser.add_argument('--arch', type=str, default='dino_small_patch16', help='unsupervised pretraining')
#parser.add_argument('--num_works', type=int, default=5, help='number of descendant tasks')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--path', type=str, default='./', help='path of base classes')
parser.add_argument('--model', default='deit_base', type=str, help='ancestry model')
parser.add_argument('--des_model', default='savit_base', type=str, help='descendant model')
parser.add_argument('--experiment', default='logs/featuredistill', help='Where to store models')
parser.add_argument('--num_imgs_per_class', type=int, default=20)
parser.add_argument('--num_classes', type=int, default=5)
parser.add_argument('--noise_level', type=float, default=20.0, help='percentage of noise added to the data (values from 0. to 100.), default: 20.')
parser.add_argument('--input-size', default=224, type=int, help='images input size')
# parser.add_argument('--embed_dim', default='same_768', type=str, help='embed_dim')
# parser.add_argument('--num_heads', default='same_12', type=str, help='embed_dim')
parser.add_argument('--depth', default=12, type=int, help='depth')
parser.add_argument('--width_ratio', default=1, type=float, help='depth')

parser.add_argument('--model-ema', action='store_true')
parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
parser.set_defaults(model_ema=True)
parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

# Augmentation parameters
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)')
parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')

# * Random Erase params
parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')

# Dataset parameters
parser.add_argument('--data-set', default='IMNET', choices=['CIFAR100', 'CIFAR10', 'IMNET', 'INAT', 'INAT19', 'MiniIMNET', 'TinyIMNET', 'food101', 'cub200', 'cars196', 'flowers102'],
                        type=str, help='Image Net dataset path')
parser.add_argument('--data-path', default='/dataset/imagenet', type=str,
                        help='dataset path')
parser.add_argument('--inat-category', default='name',
                    choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                    type=str, help='semantic granularity')
parser.add_argument('--repeated-aug', action='store_true')
parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
parser.add_argument('--dist-eval', action='store_true', default=True, help='Enabling distributed evaluation')
parser.add_argument('--num_workers', default=16, type=int)
parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--resume', default='', help='resume from checkpoint')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
parser.add_argument('--compute_mean_attn_dist', action='store_true', help='Perform evaluation only')

# * Mixup params
parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--output_dir', default='./checkpoint/deit/',
                        help='path where to save, empty for no saving')

# Optimizer parameters
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')

# Learning rate schedule parameters
parser.add_argument('--lr', type=float, default=5e-4, metavar='LR', help='learning rate (default: 0.0005)') 
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')

parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

parser.add_argument('--similarity', default='euclidean', type=str,
                        help='mahalanobis/euclidean')

# distill params
parser.add_argument('--distill_method', type=str, default='ot-learngene', help='ot-learngene/multilayer-distill')
parser.add_argument('--distill_ffn', type=str, default='patch', help='patch/cls/patch+cls')
parser.add_argument('--distill_token', type=str, default='ffn', help='ffn/attn/ffn+attn')
parser.add_argument('--number_block_per_part', default=1, type=int, help='number block per part')
parser.add_argument('--descendant_number_block_per_part', default=0, type=int, help='number block per part')
parser.add_argument('--max_update_distill', default=0, type=int, help='no dynamic distill/dynamic distill')

# * Finetuning params
parser.add_argument('--finetune', default='', help='finetune from checkpoint')
parser.add_argument('--weightinherit', default='', type=str,
                        help='weight_assignment/weight_clone/weight_stitch')
parser.add_argument('--self_distillation_adaptation', type=str, default='not', help='adaptation or not or stitch')
parser.add_argument('--expand_method', default='', type=str,
                        help='weight_assignment/weight_clone')

 # distributed training parameters
parser.add_argument("--local_rank", type=int, default=1)
parser.add_argument('--world_size', default=2, type=int,
                        help='number of distributed processes')
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')


args = parser.parse_args()
#if args.model == 'vgg':
#    args.lr = 0.005

'''if args.dataset == 'cifar100':
    if args.num_imgs_per_class == 500:
        inherit_path = './exp_data/data_cifar100/2022-08-13_07:06:19/inheritabledataset'
    elif args.num_imgs_per_class in [50, 100]: #[50, 100]
        inherit_path = './exp_data/data_cifar100/2022-08-19_05:25:06/inheritabledataset'
    else: # [5, 10, 20]
        inherit_path = './exp_data/data_cifar100/2022-09-27_05:00:26/inheritabledataset'
        

elif args.dataset == 'MiniImageNet':
    if args.num_classes == 20:
        inherit_path = './exp_data/data_MiniImageNet/2022-08-11_06:11:39/inheritabledataset'
    elif args.num_imgs_per_class == 500:
        inherit_path = './exp_data/data_MiniImageNet/2022-05-24_23:52:01/inheritabledataset'
    else: #[5, 10, 20, 50, 100]
        inherit_path = './exp_data/data_MiniImageNet/2022-08-11_00:30:42/inheritabledataset'''
        
'''
log_file = args.data_set+'_'+args.model+'_'+str(args.batchSize)+'_'+str(args.epochs) \
           +'_'+str(args.num_classes)+'way'+str(args.num_imgs_per_class)+'shot_'+args.distill_method\
+'_'+args.distill_ffn+'.txt'

set_logging_config(args.experiment, log_file)'''
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    
lr_ =args.lr

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

record_time = str((datetime.datetime.now() + datetime.timedelta(hours=8)).strftime('%Y-%m-%d_%H:%M:%S'))

RESULT_PATH_VAL = ''


def save_tsne(model, epoch):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    # print(len(testset))
    # exit()
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    extract = model
    extract.cuda()
    extract.eval()

    out_target = []
    out_output = []

    for batch_idx, (inputs, targets) in enumerate(testloader):
        # Filter out classes beyond 30
        # mask = targets < 10
        # if not mask.any():
        #     continue
        # inputs, targets = inputs[mask], targets[mask]
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = extract(inputs)
        output_np = outputs.data.cpu().numpy()
        target_np = targets.data.cpu().numpy()
        out_output.append(output_np)
        out_target.append(target_np[:,np.newaxis])

    output_array = np.concatenate(out_output, axis=0)
    target_array = np.concatenate(out_target, axis=0)
    #np.save(npy_path, output_array, allow_pickle=False)
    #np.save(npy_target, target_array, allow_pickle=False)

    #feature = np.load('./label_smooth1.npy').astype(np.float64)
    #target = np.load('./label_smooth_target1.npy')

    print('Pred shape :',output_array.shape)
    print('Target shape :',target_array.shape)

    tsne = TSNE(n_components=2, init='pca', random_state=0, n_iter=3000, perplexity=5)
    output_array = tsne.fit_transform(output_array)

    num_classes = 10
    colors = cm.Spectral(np.linspace(0, 1, num_classes))
    plt.rcParams['figure.figsize'] = 10,5
    scatter = plt.scatter(output_array[:, 0], output_array[:, 1], c=target_array[:, 0], s=10, cmap=cm.Paired) #cm.Spectral, Pastel1, Paired 
    #plt.colorbar(scatter, ticks=np.arange(num_classes))

    #plt.scatter(output_array[:, 0], output_array[:, 1], c= target_array[:,0], s=10)
    # for i in range(num_classes):
        #plt.scatter(output_array[:, 0], output_array[:, 1], color=colors, c= target_array[:,0], s=10)
        # plt.scatter(xx[out_target==i], yy[out_target==i], color=colors[i], label=labels[i], s=10)
    # 改下1-numclasses颜色
    # 改下图片尺寸
    # 调整标签的尺寸
    plt.tick_params(axis='both', which='major', labelsize=24)  # 调整主要刻度的标签尺寸
    plt.tick_params(axis='both', which='minor', labelsize=24)  # 调整次要刻度的标签尺寸
    title = 'T='+str(epoch)
    plt.title(title, fontsize=30)
    plt.savefig('./tsne/cluster-learngene-base/'+title+'.png', bbox_inches='tight')
    # plt.savefig('./tsne/scratch/'+title+'.png', bbox_inches='tight')


def save_checkpoint(states, is_best, output_dir, filename='checkpoint.pth'):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        print('making dir: %s'%output_dir)
        
    torch.save(states, os.path.join(output_dir, filename))
    
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'], os.path.join(output_dir, 'model_best.pth'))
        
def main(args):
    logger = logging.getLogger('main')
    logger.info(' '.join(os.sys.argv))
    logger.info(args)
    
    init_distributed_mode(args)

    print(args)

    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)
    
    
    print("Data loading...")
    '''if args.load_tar:
        train_dir = os.path.join(args.data_path, 'train.tar')
        train_transform = build_transform(True, args)
        dataset_train = DatasetTar(train_dir, transform=train_transform)
        args.nb_classes = 1000
        val_transform = build_transform(False, args)
        eval_dir = os.path.join(args.data_path, 'val.tar')
        dataset_val = DatasetTar(eval_dir, transform=val_transform)
    else:'''
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)
    
    if True:  # args.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            
        print('ok')
        
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batchSize,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batchSize),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    
    print("Model constructing...")
    
    from models import get_model
    if "swin" in args.model:
        ancestry_model = get_model(arch=args.model +'_patch4_window7_224', pretrained=True, args = args)
    else: 
        ancestry_model = get_model(arch=args.model +'_patch16_224', pretrained=True, args = args)
        #ancestry_model = get_model(arch='LeViT_384', pretrained=True, args = args)
    ancestry_model.to(device)
    ancestry_model.eval()
    # dummy_input = torch.randn(1, 3, 224, 224)
    # dummy_input = dummy_input.cuda()
    # outputs =  ancestry_model(dummy_input) 
    # print(ancestry_model)
    # exit()

    class CustomImageDataset(torch.utils.data.Dataset):
        def __init__(self, image_dir):
            self.image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir)]
        
        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image_path = self.image_paths[idx]
            image, preprocessed_image = load_image_from_path(image_path)
            return preprocessed_image[0]

    if args.compute_mean_attn_dist:
        # Create an instance of the dataset
        image_custom = CustomImageDataset("1000_val_images_sampled")

        # Create a DataLoader
        image_ds = DataLoader(image_custom, batch_size=args.batchSize, num_workers=args.num_workers)

        print(ancestry_model)
        model_name = args.model +'_patch16_224'
        splits = model_name.split("_")
        patch_size = int(splits[-2].replace("patch", ""))
        num_cls_tokens = 2 if "distilled" in model_name else 1
        mean_distances = dict()
        for idx, image in tqdm(enumerate(image_ds)):
            image = image.to(device, non_blocking=True)
            attention_score_dict = ancestry_model.get_all_selfattention_with_name(image)
            #_, attention_score_dict = loaded_model.predict(image)
            # Calculate the mean distances for every transformer block.
            for name, attention_weight in attention_score_dict.items():
                mean_distance = compute_mean_attention_dist(
                    patch_size=patch_size,
                    attention_weights=attention_weight,
                    num_cls_tokens=num_cls_tokens 
                )
                if idx == 0:
                    mean_distances[f"{name}_mean_dist"] = mean_distance
                else:
                    mean_distances[f"{name}_mean_dist"] = torch.cat(
                        [mean_distance, mean_distances[f"{name}_mean_dist"]], dim=0
                    )
        #import scipy.io
        #scipy.io.savemat("mean_distances.mat", mean_distances)
        import pickle
        if args.model == 'deit_base':
            with open('mean_distances.pkl', 'wb') as f:
                pickle.dump(mean_distances, f)
        elif args.model == 'deit_small':
            with open('mean_distances_small.pkl', 'wb') as f:
                pickle.dump(mean_distances, f)
        elif args.model == 'deit_tiny':
            with open('mean_distances_tiny.pkl', 'wb') as f:
                pickle.dump(mean_distances, f)
        print(mean_distances)
        print("save mean_distances.pkl")

    if 'savit_base' in args.des_model:
        shared_head_index = [[1, 5, 4, 2, 8], [2, 6, 11], [6, 3, 5, 7], [3, 7, 5, 9, 1, 2], [5, 2, 1, 8, 10], [6, 7, 10, 11], [0, 1, 5, 3], [10, 6, 4, 11], [0, 3, 7, 9], [3], [3, 4], [5]]
    elif 'savit_small' in args.des_model:
        shared_head_index = [[1, 2, 3], [0, 1, 3], [4, 2, 3], [1, 3], [4, 1, 3, 5], [4, 0, 1], [2, 0, 1, 3, 5], [4], [5, 1, 2], [5], [2], [3]]
    elif 'savit_tiny' in args.des_model:
        shared_head_index = [[0, 1, 2], [0, 2], [0, 2], [0, 1, 2], [0, 1, 2], [1, 0], [0, 1, 2], [2, 0], [0, 1, 2], [1, 0], [1], [0]]
    elif 'saswin_base' in args.des_model:
        shared_head_index = [
            [[0, 1, 2]] * 2,  
            [[0, 1, 2]] * 2,  
            [[0, 1, 2]] * 18,  
            [[0, 1, 2]] * 2    
        ]


    # if args.embed_dim == 'same_768':
    #     embed_dim_list = [768] * 12
    # elif args.embed_dim == 'same_384':
    #     embed_dim_list = [384] * 12     

    # if args.num_heads == 'same_12':
    #     num_heads_list = [12] * 12
    # elif args.num_heads == 'same_6':
    #     num_heads_list = [6] * 12

    
    #descendant_model = vit.__dict__['savit_base'](shared_head_index=shared_head_index, pretrained_vit=ancestry_model, expand_method=args.expand_method, patch_size=16, depth=args.depth, width_ratio=args.width_ratio, embed_dim=768, num_heads=12, num_classes=args.nb_classes)
    # print(shared_head_index)
    # exit()
    if "swin" in args.model:
        from models import model_transformer as vit
        descendant_model = vit.__dict__[args.des_model](shared_head_index=shared_head_index, pretrained_swin=ancestry_model, expand_method=args.expand_method, width_ratio=args.width_ratio, depths=(2, 2, 18, 2), num_classes=args.nb_classes)
    else:
        if args.expand_method in ["auto-learngene", "heuristic-learngene", "front10", "middle10", "last10", "last6"]:
            descendant_model = ancestry_model
        elif args.expand_method == "weight_assignment" or args.expand_method == "weight_clone":
            from models import vision_transformer as vit
            descendant_model = vit.__dict__[args.des_model](shared_head_index=shared_head_index, pretrained_vit=ancestry_model, expand_method=args.expand_method, patch_size=16, depth=args.depth, width_ratio=args.width_ratio, num_classes=args.nb_classes)
    
    # descendant_model = vit.__dict__['vit_small_depth6'](patch_size=16, num_classes=args.nb_classes)
    descendant_model.to(device)
    # print(descendant_model) 
    # print(args.des_model)
    # exit()


    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            descendant_model,
            decay=args.model_ema_decay,
            device='cuda',
            resume='')

    model_without_ddp = descendant_model
    if args.distributed:
        descendant_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(descendant_model)
        # descendant_model = torch.nn.parallel.DistributedDataParallel(descendant_model, device_ids=[args.gpu], broadcast_buffers=False, find_unused_parameters=True)
        descendant_model = torch.nn.parallel.DistributedDataParallel(descendant_model, device_ids=[args.gpu])
        descendant_model._set_static_graph()
        model_without_ddp = descendant_model.module                         
    
    # TODO 解除注释 debug
    # calculate flops and params
    from thop import profile
    from thop import clever_format

    dummy_input = torch.randn(1, 3, 224, 224)
    dummy_input = dummy_input.cuda()

    # print(args.nb_classes)
    # args.nb_classes = 0
    # ancestry_model = get_model(arch='deit_small_depth8_patch16_224', pretrained=True, args = args).to(device)
    flops, params = profile(ancestry_model, inputs=(dummy_input,))
    flops, params = clever_format([flops, params], '%.3f')

    print('ancestry Model parameters: {}'.format(params) )
    print('Floating-point operations per sample: {}'.format(flops) )
    # exit()
    
    flops, params = profile(descendant_model, inputs=(dummy_input,))
    flops, params = clever_format([flops, params], '%.3f')

    print('descendant Model parameters: {}'.format(params) )
    print('Floating-point operations per sample: {}'.format(flops) )    
        
    n_parameters = sum(p.numel() for p in descendant_model.parameters() if p.requires_grad)
    print('descendant Model parameters:', n_parameters) 

    linear_scaled_lr = args.lr * args.batchSize * get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)
    
    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        
    output_dir = Path(args.output_dir)
    if args.resume: #训练过程resume
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        print(checkpoint['epoch'])
        args.start_epoch = checkpoint['epoch']+1
        
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

    if args.finetune: #主要用于下游任务fine-tune
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')
            print(checkpoint['model'].keys())
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint['model']:
                    print(f"removing key {k} from pretrained checkpoint")
                    del checkpoint['model'][k]
            # exit()
        print(checkpoint['epoch'])
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)

        # args.start_epoch = checkpoint['epoch']+1
        # if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        #     lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        #     args.start_epoch = checkpoint['epoch'] + 1
        #     if args.model_ema:
        #         _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
        #     if 'scaler' in checkpoint:
        #         loss_scaler.load_state_dict(checkpoint['scaler'])
    
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, args)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return
    
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        #if epoch > 5:
        #    exit()
        
        
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            descendant_model, ancestry_model, criterion, args.similarity, args.max_update_distill, args.distill_method, args.distill_ffn, args.distill_token, args.number_block_per_part, args.descendant_number_block_per_part, data_loader_train, optimizer, device, epoch, loss_scaler, args.clip_grad, 
            model_ema, mixup_fn, args, set_training_mode=args.finetune == '' 
        ) # keep in eval mode during finetuning

        lr_scheduler.step(epoch)
        #print('Current learning rate:', lr_scheduler.state_dict())
        #continue
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)

        test_stats = evaluate(data_loader_val, descendant_model, device, args)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if epoch % 50 == 0: 
            save_tsne(descendant_model, epoch)
            print("save tsne")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    main(args)