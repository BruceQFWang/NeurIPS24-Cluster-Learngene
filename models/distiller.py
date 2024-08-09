import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
import scipy

# We refer to overhaul-distillation (https://github.com/clovaai/overhaul-distillation)

import math

#def distillation_loss(source, target, margin):
    #target = torch.max(target, margin)
def distillation_loss(source, target):
    loss = torch.nn.functional.mse_loss(source, target, reduction="none")
    #loss = loss * ((source > target) | (target > 0)).float()
    return loss.sum()

def self_distillation_loss(cls_tokens): # distill only cls 
    """
    Compute the distillation loss.
    """
    last_cls_token = cls_tokens[-1]  # Get the last layer CLS token

    # Initialize loss
    loss = 0.0

    # Iterate over all CLS tokens except the last one
    for cls_token in cls_tokens[:-1]:
        loss += nn.functional.mse_loss(last_cls_token.detach(), cls_token)  # Compute loss as MSE

    return loss / (len(cls_tokens) - 1)  # Return the average loss

def self_distillation_adaptation_loss(model, cls_tokens, teacher_cls_token): # distill after adaptation_layers
    """
    Compute the distillation loss.
    """
    # Detach the teacher_cls_token so that we don't compute gradients for it
    teacher_cls_token = teacher_cls_token.detach()

    # Initialize loss
    loss = 0.0

    # Iterate over all CLS tokens
    for i, cls_token in enumerate(cls_tokens):
        # Apply the corresponding adaptation layer to the cls_token
        adapted_cls_token = model.adaptation_layers[i](cls_token)
        # Compute loss as MSE
        loss += nn.functional.mse_loss(teacher_cls_token, adapted_cls_token)

    return loss / len(cls_tokens)  # Return the average loss


def build_feature_connector(t_channel, s_channel):
    C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
         nn.BatchNorm2d(t_channel)]

    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return nn.Sequential(*C)

def get_margin_from_BN(bn):
    margin = []
    std = bn.weight.data
    mean = bn.bias.data
    for (s, m) in zip(std, mean):
        s = abs(s.item())
        m = m.item()
        if norm.cdf(-m / s) > 0.001:
            margin.append(- s * math.exp(- (m / s) ** 2 / 2) / math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m)
        else:
            margin.append(-3 * s)

    return torch.FloatTensor(margin).to(std.device)

class Distiller(nn.Module):
    def __init__(self, t_net, s_net):
        super(Distiller, self).__init__()

        t_channels = t_net.get_channel_num()
        s_channels = s_net.get_channel_num()

        self.Connectors = nn.ModuleList([build_feature_connector(t, s) for t, s in zip(t_channels, s_channels)])

        teacher_bns = t_net.get_bn_before_relu()
        margins = [get_margin_from_BN(bn) for bn in teacher_bns]
        for i, margin in enumerate(margins):
            self.register_buffer('margin%d' % (i+1), margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach())

        self.t_net = t_net
        self.s_net = s_net

    def forward(self, x):

        t_feats, t_out = self.t_net.extract_feature(x, preReLU=True)
        s_feats, s_out = self.s_net.extract_feature(x, preReLU=True)
        feat_num = len(t_feats)

        loss_distill = 0
        for i in range(feat_num):
            s_feats[i] = self.Connectors[i](s_feats[i])
            loss_distill += distillation_loss(s_feats[i], t_feats[i].detach(), getattr(self, 'margin%d' % (i+1))) \
                            / 2 ** (feat_num - i - 1)

        return s_out, loss_distill

# 用于新的层蒸馏 (P2K)
class Cnn_Distiller(nn.Module):
    def __init__(self, t_net, s_net, model_name):
        super(Cnn_Distiller, self).__init__()

        if model_name == 'vgg':
            #t_channels = [64, 128, 256, 512, 512]
            #s_channels = [64, 128, 256, 512, 512]
            t_channels = [512]
            s_channels = [512]

        self.Connectors = nn.ModuleList([build_feature_connector(t, s) for t, s in zip(t_channels, s_channels)])

        #teacher_bns = t_net.get_bn_before_relu()
        #margins = [get_margin_from_BN(bn) for bn in teacher_bns]
        #for i, margin in enumerate(margins):
        #    self.register_buffer('margin%d' % (i+1), margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach())

        self.t_net = t_net
        self.s_net = s_net

    def forward(self, x):

        t_feats, t_out = self.t_net.extract_feature_cnn(x)
        s_feats, s_out = self.s_net.extract_feature_cnn(x)
        feat_num = len(t_feats)

        loss_distill = 0
        for i in range(feat_num):
            s_feats[i] = self.Connectors[i](s_feats[i])
            loss_distill += distillation_loss(s_feats[i], t_feats[i].detach()) / 2 ** (feat_num - i - 1) #teacher不需要在计算图中更新
            #loss_distill += distillation_loss(s_feats[i], t_feats[i].detach(), getattr(self, 'margin%d' % (i+1))) \
            #                / 2 ** (feat_num - i - 1)

        return s_out, loss_distill

class Direct_Distiller(nn.Module):
    def __init__(self, t_net, s_net):
        super(Direct_Distiller, self).__init__()

        self.t_net = t_net
        self.s_net = s_net

    def forward(self, x):

        t_feats, t_out = self.t_net.extract_feature(x)
        s_feats, s_out = self.s_net.extract_feature(x)
        feat_num = len(t_feats)

        loss_distill = 0
        for i in range(feat_num):
            #s_feats[i] = self.Connectors[i](s_feats[i])
            loss_distill += distillation_loss(s_feats[i], t_feats[i].detach()) / 2 ** (feat_num - i - 1) #teacher不需要在计算图中更新
            #loss_distill += distillation_loss(s_feats[i], t_feats[i].detach(), getattr(self, 'margin%d' % (i+1))) \
            #                / 2 ** (feat_num - i - 1)

        return s_out, loss_distill

    
'''def compute_patch_loss(t_feats, s_feats):
    t_feat_num = len(t_feats)
    s_feat_num = len(s_feats)

    loss_matrix = torch.zeros(t_feat_num, s_feat_num)

    for i in range(t_feat_num):
        for j in range(s_feat_num):
            t_feat = F.interpolate(  
                    t_feats[i],
                    scale_factor= s_feats[j].size(2) / t_feats[i].size(2),
                    mode='linear'
                    )
            loss_matrix[i,j] = distillation_loss(s_feats[j], t_feat.detach() )   
    return loss_matrix'''

def compute_patch_loss(t_feats, s_feats):
    t_feat_num = len(t_feats)
    s_feat_num = len(s_feats)

    scale_factors = [s_feats[j].size(2) / t_feats[i].size(2) for j in range(s_feat_num) for i in range(t_feat_num)]
    t_feats_interpolated = F.interpolate(t_feats[None, ...], scale_factor=scale_factors, mode='linear').squeeze(0)
    
    t_feats_interpolated_detach = t_feats_interpolated.detach()

    loss_matrix = distillation_loss(s_feats[:, None, ...], t_feats_interpolated_detach[:, None, ...])
    
    return loss_matrix

def compute_cls_loss(t_cls_token, s_cls_token):
    t_cls_token_num = len(t_cls_token)
    s_cls_token_num = len(s_cls_token)
    loss_matrix = torch.zeros(t_cls_token_num, s_cls_token_num)

    for k in range(t_cls_token_num):
        t_cls_token[k] = t_cls_token[k].unsqueeze(0)
    for k in range(s_cls_token_num):
        s_cls_token[k] = s_cls_token[k].unsqueeze(0)
            
    
    for i in range(t_cls_token_num):
        for j in range(s_cls_token_num):
            #print(t_cls_token[i].shape)
            t_cls = F.interpolate(  
                    t_cls_token[i],
                    scale_factor= s_cls_token[j].size(2) / t_cls_token[i].size(2),
                    mode='linear'
                    )
            loss_matrix[i,j] = distillation_loss(s_cls_token[j], t_cls.detach() ) 
    return loss_matrix

def compute_attn_loss(t_attn_avg, s_attn_avg):
    t_attn_avg_num = len(t_attn_avg)
    s_attn_avg_num = len(s_attn_avg)
    loss_matrix = torch.zeros(t_attn_avg_num, s_attn_avg_num)
    for i in range(t_attn_avg_num):
        for j in range(s_attn_avg_num):
            loss_matrix[i,j] = distillation_loss(s_attn_avg[j], t_attn_avg[i].detach() )
    return loss_matrix
    
    
class Vit_multilayer_Distiller(nn.Module):
    def __init__(self, t_net, s_net):
        super(Vit_multilayer_Distiller, self).__init__()

        self.t_net = t_net
        self.s_net = s_net

    def forward(self, x, distill_weight, distill_ffn, distill_token, number_block_per_part, descendant_number_block_per_part):
        distill_str = distill_token.split("+") 
        loss_matrix = torch.zeros(distill_weight.size(0), distill_weight.size(1))
        #mul_num = 0.1
        
        if 'ffn' in distill_str:
            t_feats, t_cls_token, t_patch_and_cls = self.t_net.extract_feature(x, number_block_per_part) # t_feats[layer_num, batch, patch=196, token=768] t_cls_token[layer_num, 1, batch, token=768]
            s_feats, s_cls_token, s_patch_and_cls = self.s_net.module.extract_feature(x, descendant_number_block_per_part)  # deit_tiny  s_feats[layer_num, batch, patch=196, token=192] s_cls_token[layer_num, 1, batch, token=192]
            if distill_ffn == 'cls':
                loss_matrix = loss_matrix + compute_cls_loss(t_cls_token, s_cls_token) #pair*s_token=piar*192
            '''if distill_ffn == 'patch': 
                loss_matrix = loss_matrix + compute_patch_loss(t_feats, s_feats) #pair*patch*s_token=piar*196*192
                mul_num = 2
            elif distill_ffn == 'cls':
                loss_matrix = loss_matrix + compute_cls_loss(t_cls_token, s_cls_token) #pair*s_token=piar*192
                mul_num = 0.1
                #print('cls loss {}'.format( loss_matrix.sum() ) )
            elif distill_ffn == 'patch+cls': 
                loss_matrix = loss_matrix + compute_patch_loss(t_patch_and_cls, s_patch_and_cls)
                mul_num = 2.1'''


            '''
            print('patch loss {}'.format(compute_patch_loss(t_feats, s_feats)) )
            print('cls loss {}'.format(compute_cls_loss(t_cls_token, s_cls_token)) )
            print('t_feats[0].shape {}'.format(t_feats[0].shape) )
            print('t_cls_token[0].shape {}'.format(t_cls_token[0].shape) )
            print('s_feats[0].shape {}'.format(s_feats[0].shape))
            print('s_cls_token[0].shape {}'.format(s_cls_token[0].shape) )
            print("================")'''
                
        if 'attn' in distill_str:
            t_attn = self.t_net.get_all_selfattention(x, number_block_per_part) # deit_base  t_attn[layer_num, batch, head=12, patch+cls=197, patch+cls=197] 
            s_attn = self.s_net.module.get_all_selfattention(x, descendant_number_block_per_part)  # deit_tiny  s_attn[layer_num, batch, head=3, patch+cls=197, patch+cls=197] 
            t_attn_avg = [torch.mean(t_attn[i], dim=1) for i in range(len(t_attn))]
            s_attn_avg = [torch.mean(s_attn[i], dim=1) for i in range(len(s_attn))]
            loss_matrix = loss_matrix + 200 * compute_attn_loss(t_attn_avg, s_attn_avg) # pair*(patch+cls)*(patch+cls)=layer_num*197*197
            #loss_matrix = loss_matrix + mul_num * 2000 * compute_attn_loss(t_attn_avg, s_attn_avg) # pair*(patch+cls)*(patch+cls)=layer_num*197*197

            #print('attn loss {}'.format(compute_attn_loss(t_attn_avg, s_attn_avg).sum()))
            #exit()

            '''print('attn loss {}'.format(compute_attn_loss(t_attn_avg, s_attn_avg).sum() ) )
            print('attn loss {}'.format(compute_attn_loss(t_attn_avg, s_attn_avg)) )
            print('t_attn[0].shape {}'.format(t_attn[0].shape) )
            print('s_attn[0].shape {}'.format(s_attn[0].shape) )'''
        
        loss_distill = torch.mul(distill_weight, loss_matrix )
        
        return loss_distill