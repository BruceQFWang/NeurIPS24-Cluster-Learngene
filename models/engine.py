"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from .losses import DistillationLoss
from utils.utils import MetricLogger, SmoothedValue
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
import models.distiller as distiller
import argparse
import pickle
from timm.optim import create_optimizer

def mahalanobis(u, v, cov):
    delta = u - v
    m = torch.dot(delta, torch.matmul(torch.inverse(cov), delta))
    return torch.sqrt(m)

def _batch_mahalanobis(bL, bx):
    r"""
    Computes the squared Mahalanobis distance :math:`\mathbf{x}^\top\mathbf{M}^{-1}\mathbf{x}`
    for a factored :math:`\mathbf{M} = \mathbf{L}\mathbf{L}^\top`.
    Accepts batches for both bL and bx. They are not necessarily assumed to have the same batch
    shape, but `bL` one should be able to broadcasted to `bx` one.
    """
    n = bx.size(-1)
    bx_batch_shape = bx.shape[:-1]

    # Assume that bL.shape = (i, 1, n, n), bx.shape = (..., i, j, n),
    # we are going to make bx have shape (..., 1, j,  i, 1, n) to apply batched tri.solve
    bx_batch_dims = len(bx_batch_shape)
    bL_batch_dims = bL.dim() - 2
    outer_batch_dims = bx_batch_dims - bL_batch_dims
    old_batch_dims = outer_batch_dims + bL_batch_dims
    new_batch_dims = outer_batch_dims + 2 * bL_batch_dims
    # Reshape bx with the shape (..., 1, i, j, 1, n)
    bx_new_shape = bx.shape[:outer_batch_dims]
    for (sL, sx) in zip(bL.shape[:-2], bx.shape[outer_batch_dims:-1]):
        bx_new_shape += (sx // sL, sL)
    bx_new_shape += (n,)
    bx = bx.reshape(bx_new_shape)
    # Permute bx to make it have shape (..., 1, j, i, 1, n)
    permute_dims = (list(range(outer_batch_dims)) +
                    list(range(outer_batch_dims, new_batch_dims, 2)) +
                    list(range(outer_batch_dims + 1, new_batch_dims, 2)) +
                    [new_batch_dims])
    bx = bx.permute(permute_dims)

    flat_L = bL.reshape(-1, n, n)  # shape = b x n x n
    flat_x = bx.reshape(-1, flat_L.size(0), n)  # shape = c x b x n
    flat_x_swap = flat_x.permute(1, 2, 0)  # shape = b x n x c
    M_swap = torch.linalg.solve_triangular(flat_L, flat_x_swap, upper=False).pow(2).sum(-2)  # shape = b x c
    M = M_swap.t()  # shape = c x b

    # Now we revert the above reshape and permute operators.
    permuted_M = M.reshape(bx.shape[:-1])  # shape = (..., 1, j, i, 1)
    permute_inv_dims = list(range(outer_batch_dims))
    for i in range(bL_batch_dims):
        permute_inv_dims += [outer_batch_dims + i, old_batch_dims + i]
    reshaped_M = permuted_M.permute(permute_inv_dims)  # shape = (..., 1, i, j, 1)
    return reshaped_M.reshape(bx_batch_shape)

'''
def compute_optimal_transport(M, r, c, lam, epsilon=1e-5):
    """
    Computes the optimal transport matrix and Slinkhorn distance using the
    Sinkhorn-Knopp algorithm

    Inputs:
        - M : cost matrix (n x m)
        - r : vector of marginals (n, )
        - c : vector of marginals (m, )
        - lam : strength of the entropic regularization  
        - epsilon : convergence parameter

    Output:
        - P : optimal transport matrix (n x m)
        - dist : Sinkhorn distance
    """
    n, m = M.shape
    P = np.exp(- lam * M)
    # Avoiding poor math condition
    P /= P.sum()
    u = np.zeros(n)
    # Normalize this matrix so that P.sum(1) == r, P.sum(0) == c
    cnt = 0
    while np.max(np.abs(u - P.sum(1))) > epsilon:
        #if cnt%100 == 0:
        #    print(cnt)
        #    print(np.max(np.abs(u - P.sum(1))))
        # Shape (n, )
        u = P.sum(1)
        P *= (r / u).reshape((-1, 1))
        P *= (c / P.sum(0)).reshape((1, -1))
    return P, np.sum(P * M) '''

def compute_optimal_transport(M, r, c, lam, epsilon=1e-5):
    """
    Computes the optimal transport matrix and Slinkhorn distance using the
    Sinkhorn-Knopp algorithm

    Inputs:
        - M : cost matrix (n x m)
        - r : vector of marginals (n, )
        - c : vector of marginals (m, )
        - lam : strength of the entropic regularization  
        - epsilon : convergence parameter

    Output:
        - P : optimal transport matrix (n x m)
        - dist : Sinkhorn distance
    """
    n, m = M.shape
    P = torch.exp(- lam * M)
    # Avoiding poor math condition
    P /= P.sum()
    u = torch.zeros(n)
    # Normalize this matrix so that P.sum(1) == r, P.sum(0) == c
    cnt = 0
    while torch.max(torch.abs(u - P.sum(1))) > epsilon:
        u = P.sum(1)
        P *= (r / u).reshape((-1, 1))
        P *= (c / P.sum(0)).reshape((1, -1))
    return P, torch.sum(P * M)

def compute_ffn_loss(ancestry_feats_num, descendant_feats_num, ancestry_feats, descendant_feats, similarity_metric):
    token_similarity_matrix = torch.zeros(ancestry_feats_num, descendant_feats_num)

    # Mahalanobis Distance for solving inconsistent dimensions
    for i in range(ancestry_feats_num):
        for j in range(descendant_feats_num):

            # interpolation to perform up/down sampling operations on ancestry_feats to align the matrix
            an_feat = F.interpolate(  
                        ancestry_feats[i],
                        scale_factor= descendant_feats[j].size(2) / ancestry_feats[i].size(2),
                        mode='linear'
                )

            de_feat = descendant_feats[j]

            if similarity_metric == 'euclidean':
            # similarity of the last dim / token
                token_similarity_matrix[i,j] =  torch.mean(torch.cosine_similarity(an_feat, de_feat, dim=-1)) 
                #print('similarity', token_similarity_matrix[i,j])

            elif similarity_metric == 'mahalanobis':
                an_feat = torch.reshape(an_feat, [-1, descendant_feats[j].size(2)])
                de_feat = torch.reshape(de_feat, [-1, descendant_feats[j].size(2)]) 
                #token_similarity_matrix[i,j] =  torch.mean(_batch_mahalanobis(an_feat, de_feat))

    #print(token_similarity_matrix)
    Cost = 1.0 - token_similarity_matrix 
    r = torch.ones((ancestry_feats_num), device=Cost.device, dtype=Cost.dtype)
    c = torch.ones((descendant_feats_num), device=Cost.device, dtype=Cost.dtype)

    lam = 10
    # P, d = compute_optimal_transport(Cost.detach().numpy(), r.detach().numpy(), c.detach().numpy(), lam=lam)
    P, d = compute_optimal_transport(Cost, r, c, lam=lam)

    return  d

def compute_cls_loss(ancestry_cls_token, descendant_cls_token):
    ancestry_cls_token_num = len(ancestry_cls_token)
    descendant_cls_token_num = len(descendant_cls_token)
    token_similarity_matrix = torch.zeros(ancestry_cls_token_num, descendant_cls_token_num)
     

    for k in range(ancestry_cls_token_num):
        ancestry_cls_token[k] = ancestry_cls_token[k].unsqueeze(0)
    for k in range(descendant_cls_token_num):
        descendant_cls_token[k] = descendant_cls_token[k].unsqueeze(0)
            
    for i in range(ancestry_cls_token_num):
        for j in range(descendant_cls_token_num):
            #print(ancestry_cls_token[i].shape)
            an_cls = F.interpolate(  
                    ancestry_cls_token[i],
                    scale_factor= descendant_cls_token[j].size(2) / ancestry_cls_token[i].size(2),
                    mode='linear'
                    ).detach()
            de_cls = descendant_cls_token[j]
            token_similarity_matrix[i,j] =  torch.mean(torch.cosine_similarity(an_cls, de_cls, dim=-1))
    
    #print(token_similarity_matrix)
    Cost = 1.0 - token_similarity_matrix 
    r = torch.ones((ancestry_cls_token_num), device=Cost.device, dtype=Cost.dtype)
    c = torch.ones((descendant_cls_token_num), device=Cost.device, dtype=Cost.dtype)

    lam = 10
    # P, d = compute_optimal_transport(Cost.detach().numpy(), r.detach().numpy(), c.detach().numpy(), lam=lam)
    P, d = compute_optimal_transport(Cost, r, c, lam=lam)

    return d     

def compute_attn_loss(ancestry_attn, descendant_attn):

    an_attn_avg = [torch.mean(ancestry_attn[i], dim=1) for i in range(len(ancestry_attn))] 
    de_attn_avg = [torch.mean(descendant_attn[i], dim=1) for i in range(len(descendant_attn))]
    an_attn_avg_num = len(an_attn_avg)
    de_attn_avg_num = len(de_attn_avg)

    an_attn_avg_resize = [torch.reshape(an_attn_avg[i], (an_attn_avg[i].shape[0], -1)) for i in range(len(an_attn_avg))] 
    de_attn_avg_resize = [torch.reshape(de_attn_avg[i], (de_attn_avg[i].shape[0], -1)) for i in range(len(de_attn_avg))] 

    #print(an_attn_avg[0].shape)
    #print(de_attn_avg[0].shape)
    #print(an_attn_avg_resize[0].shape)
    #print(de_attn_avg_resize[0].shape)

    token_similarity_matrix = torch.zeros(an_attn_avg_num, de_attn_avg_num)
    for i in range(an_attn_avg_num):
        for j in range(de_attn_avg_num):
            token_similarity_matrix[i,j] =  torch.mean(torch.cosine_similarity(an_attn_avg[i].detach(), de_attn_avg[j], dim=-1))
    
    #print(token_similarity_matrix)
    Cost = 1.0 - token_similarity_matrix 
    r = torch.ones((an_attn_avg_num), device=Cost.device, dtype=Cost.dtype)
    c = torch.ones((de_attn_avg_num), device=Cost.device, dtype=Cost.dtype)

    lam = 10
    # P, d = compute_optimal_transport(Cost.detach().numpy(), r.detach().numpy(), c.detach().numpy(), lam=lam)
    P, d = compute_optimal_transport(Cost, r, c, lam=lam)

    return d     

def ot_multilayer_distill(ancestry_model, model, samples, similarity_metric, distill_method, distill_ffn, distill_token, number_block_per_part, descendant_number_block_per_part):
    #Optimal Transport
    ancestry_feats, ancestry_cls_token, ancestry_patch_and_cls = ancestry_model.extract_feature(samples, number_block_per_part) # an_feats[layer_num, batch, patch=196, token=768] an_cls_token[layer_num, 1, batch, token=768]
    descendant_feats, descendant_cls_token, descendant_patch_and_cls = model.module.extract_feature(samples, descendant_number_block_per_part) # deit_tiny  de_feats[layer_num, batch, patch=196, token=192] de_cls_token[layer_num, 1, batch, token=192]
    ancestry_feats_num = len(ancestry_feats)
    descendant_feats_num = len(descendant_feats)
    ancestry_attn = ancestry_model.get_all_selfattention(samples, number_block_per_part) # deit_base  t_attn[layer_num, batch, head=12, patch+cls=197, patch+cls=197] 
    descendant_attn = model.module.get_all_selfattention(samples, descendant_number_block_per_part)  # deit_tiny  s_attn[layer_num, batch, head=3, patch+cls=197, patch+cls=197] 
    #print(ancestry_feats_num)
    #print(descendant_feats_num)

    
    if distill_method == 'ot-learngene':     
        d1 = compute_ffn_loss(ancestry_feats_num, descendant_feats_num, ancestry_feats, descendant_feats, similarity_metric)
        d2 = compute_cls_loss(ancestry_cls_token, descendant_cls_token)
        d3 = compute_attn_loss(ancestry_attn, descendant_attn)
        if distill_token == 'ffn+attn':
            d = d2+d3
        elif distill_token == 'ffn':
            #if distill_ffn == 'cls':
            d = d2 
        elif distill_token == 'attn':
            d = d3
    
    elif distill_method == 'multilayer-distill':
        distill_weight = torch.ones(ancestry_feats_num, descendant_feats_num)
        distill_weight = 1.0 * distill_weight/ ancestry_feats_num
    
    return d, ancestry_feats_num*descendant_feats_num
    

def train_one_epoch(model: torch.nn.Module, ancestry_model: torch.nn.Module, criterion: DistillationLoss, 
                    similarity_metric: 'mahalanobis', max_update_distill: 0, distill_method:'ot-learngene', distill_ffn:'patch', distill_token:'ffn', number_block_per_part:1, descendant_number_block_per_part:0,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, args: Optional[argparse.Namespace] = None,
                    set_training_mode=True):
    model.train(set_training_mode) # close dropout&bn
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if descendant_number_block_per_part == 0:
        descendant_number_block_per_part = number_block_per_part
    cnt = 0
    init = False
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        cnt = cnt+1
        if cnt > (len(data_loader)/2):
            continue 
        
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():  
            outputs = model.module.head( model(samples) )
            #loss = criterion(samples, outputs, targets)
            loss = criterion(outputs, targets)

        # attention可视化
        # if epoch == 49:
        #     with open('small_attention_distribution.pkl', 'wb') as f:
        #         pickle.dump(model.module.get_all_selfattention_with_name(samples), f)
        #     exit()

        batchSize = samples.shape[0]
        if max_update_distill == 0:
            distill_ratio = 0.5
        else:
            distill_ratio = 1.0 - float(epoch) / max_update_distill
            if distill_ratio < 0:
                distill_ratio = 0


        # finetuning
        if args.self_distillation_adaptation == 'adaptation':
            descendant_feats, descendant_cls_token, descendant_patch_and_cls = model.module.extract_feature(samples, descendant_number_block_per_part) 
            if init is False and epoch == 0:
                    #   init the adaptation layers.
                    #   we add feature adaptation layers here to soften the influence from feature distillation loss
                    #   the feature distillation in our conference version :  | f1-f2 | ^ 2
                    #   the feature distillation in the final version : |Fully Connected Layer(f1) - f2 | ^ 2
                layer_list = []
                for index in range(0, len(descendant_cls_token)-1):
                    student_feature_size = descendant_cls_token[index].size(1)
                    teacher_feature_size = model.module.head.out_features
                    layer_list.append(nn.Linear(student_feature_size, teacher_feature_size))
                model.adaptation_layers = nn.ModuleList(layer_list)
                model.adaptation_layers.cuda()
                optimizer = create_optimizer(args, model.module)
                    #   define the optimizer here again so it will optimize the net.adaptation_layers
                init = True

            loss_self_distill = distiller.self_distillation_adaptation_loss(model, descendant_cls_token[:-1], outputs) 
                #early stage of self-distillation, the cls loss accounts for a larger proportion
            loss_value = loss.item() * distill_ratio + loss_self_distill.item() / batchSize * (1.0 - distill_ratio) 
        elif args.self_distillation_adaptation == 'not':
            descendant_feats, descendant_cls_token, descendant_patch_and_cls = model.module.extract_feature(samples, descendant_number_block_per_part) 
            loss_self_distill = distiller.self_distillation_loss(descendant_cls_token) 
                #early stage of self-distillation, the cls loss accounts for a larger proportion
            loss_value = loss.item() * distill_ratio + loss_self_distill.item() / batchSize * (1.0 - distill_ratio) 

                #print(loss.item() )
                #print(loss_self_distill.item() / batchSize)
                #exit()
            
        elif args.self_distillation_adaptation == 'stitch':
            loss_value = loss.item()

        #n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        #print('number of requires_grad params:', n_parameters)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        
        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])  
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    cnt = 0
    for images, target in metric_logger.log_every(data_loader, 10, header):
        # if cnt > (len(data_loader)/4):
        #     break
        # cnt = cnt+1 
            
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model.module.head( model(images) )
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

from torchvision import transforms
from PIL import Image
input_resolution = 224
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

crop_layer = transforms.CenterCrop(input_resolution)
norm_layer = transforms.Normalize(mean=mean * 255, std=std * 255)
rescale_layer = transforms.Lambda(lambda x: (x / 127.5) - 1)

def preprocess_image(image, size=input_resolution):
    model_type = 'deit'
    # turn the image into a PyTorch tensor and add batch dim
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0)

    # if model type is vit rescale the image to [-1, 1]
    if model_type == "vit":
        image = rescale_layer(image)

    # resize the image using bicubic interpolation
    resize_size = int((256 / 224) * size)
    image = transforms.Resize((resize_size, resize_size), interpolation=Image.BICUBIC)(image)

    # crop the image
    image = crop_layer(image)

    # if model type is deit normalize the image
    if model_type != "vit":
        image = norm_layer(image)

    # return the image
    return image

def load_image_from_path(path):
    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    preprocessed_image = preprocess_image(image)
    return image, preprocessed_image
    
def compute_distance_matrix(patch_size, num_patches, length):
    """Helper function to compute distance matrix."""
    distance_matrix = np.zeros((num_patches, num_patches))
    for i in range(num_patches):
        for j in range(num_patches):
            if i == j:  # zero distance
                continue

            xi, yi = (int(i / length)), (i % length)
            xj, yj = (int(j / length)), (j % length)
            distance_matrix[i, j] = patch_size * np.linalg.norm([xi - xj, yi - yj])

    return distance_matrix

def compute_mean_attention_dist(patch_size, attention_weights, num_cls_tokens):
    # The attention_weights shape = (batch, num_heads, num_patches, num_patches)
    attention_weights = attention_weights[..., num_cls_tokens:, num_cls_tokens:]  # Removing the CLS token
    num_patches = attention_weights.shape[-1]
    length = int(np.sqrt(num_patches))
    assert length ** 2 == num_patches, "Num patches is not perfect square"

    distance_matrix = compute_distance_matrix(patch_size, num_patches, length)
    h, w = distance_matrix.shape

    distance_matrix = distance_matrix.reshape((1, 1, h, w))
    # The attention_weights along the last axis adds to 1
    # this is due to the fact that they are softmax of the raw logits
    # summation of the (attention_weights * distance_matrix)
    # should result in an average distance per token
    mean_distances = attention_weights.cpu().detach().numpy() * distance_matrix
    mean_distances = np.sum(
        mean_distances, axis=-1
    )  # sum along last axis to get average distance per token
    mean_distances = np.mean(
        mean_distances, axis=-1
    )  # now average across all the tokens

    return torch.from_numpy(mean_distances)