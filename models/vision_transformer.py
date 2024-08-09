import torch
import torch.nn as nn

import math
from functools import partial
from .utils import trunc_normal_


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer_Residual(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], residual_interval=3, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.residual_interval = residual_interval
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
            align_corners=False,
            recompute_scale_factor=False
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x, ada_token=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        if ada_token is not None:
            ada_tokens = ada_token.expand(B, -1, -1) # B, p, d
            x = torch.cat((x, ada_tokens), dim=1)

        return self.pos_drop(x)

    def forward(self, x, ada_token=None, use_patches=False):
        x = self.prepare_tokens(x, ada_token)
        for i, blk in enumerate(self.blocks):
            if i % self.residual_interval == 0:  # add a residual connection 
                res = x
            x = blk(x)
            if i % self.residual_interval == 2 and i != 5:  # add the residual connection to the output 
                x = x + res
        x = self.norm(x)

        if use_patches:
            return x[:, 1:]
        else:
            return x[:, 0]


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
            align_corners=False,
            recompute_scale_factor=False
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x, ada_token=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        if ada_token is not None:
            ada_tokens = ada_token.expand(B, -1, -1) # B, p, d
            x = torch.cat((x, ada_tokens), dim=1)

        return self.pos_drop(x)

    def forward(self, x, ada_token=None, use_patches=False):
        x = self.prepare_tokens(x, ada_token)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if use_patches:
            return x[:, 1:]
        else:
            return x[:, 0]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output

    def get_layers_num_0(self):  
        return self.blocks[:2] #前两层
    
    def get_layers_num_2(self):  
        return self.blocks[-4:] #最后四层
    
    def get_layers_num_2_front6blocks(self):  
        return self.blocks[:6] #最后六层
    
    def get_layers_num_2_last6blocks(self):  
        return self.blocks[-6:] #最后六层

    def extract_feature(self, x, number_block_per_part, ada_token=None):
        x = self.prepare_tokens(x, ada_token)
        feat = []
        cls_token = []
        patch_and_cls_token = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i+1)%number_block_per_part == 0:
                feat.append(x[:, 1:]) # remove cls token
                cls_token.append(x[:, 0])
                patch_and_cls_token.append(x)
        x = self.norm(x)

        return feat, cls_token, patch_and_cls_token
    
    def get_all_selfattention(self, x, number_block_per_part):
        x = self.prepare_tokens(x)
        attn = []
        for i, blk in enumerate(self.blocks):
            if (i+1)%number_block_per_part == 0:
                attn.append(blk(x, return_attention=True) )
            x = blk(x)
            
        # return attention of the all blocks
        return attn
    
    def get_all_selfattention_with_name(self, x):
        x = self.prepare_tokens(x)
        dict_attn = {}
        for name, blk in self.blocks._modules.items():
        #for i, blk in enumerate(self.blocks):
        #    if (i+1)%number_block_per_part == 0:
        #attn.append(blk(x, return_attention=True) )
            dict_attn[name] = blk(x, return_attention=True)
            x = blk(x)
            
        # return attention of the all blocks
        return dict_attn

class PrunedMlp(nn.Module):
    def __init__(self, mlp, width_ratio):
        super().__init__()
        
        # 裁剪fc1的权重
        fc1_out_features = mlp.fc1.weight.data.size(0) // width_ratio
        fc1_in_features = mlp.fc1.weight.data.size(1) // width_ratio
        self.fc1 = nn.Linear(fc1_in_features, fc1_out_features)
        self.fc1.weight.data = mlp.fc1.weight.data[:fc1_out_features, :fc1_in_features]

        # 裁剪fc2的权重
        fc2_out_features = mlp.fc2.weight.data.size(0) // width_ratio
        fc2_in_features = mlp.fc2.weight.data.size(1) // width_ratio
        self.fc2 = nn.Linear(fc2_in_features, fc2_out_features)
        self.fc2.weight.data = mlp.fc2.weight.data[:fc2_out_features, :fc2_in_features]

        self.act = mlp.act
        self.drop = mlp.drop

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def PruneNorm(norm, width_ratio):
    norm_out = norm.weight.data.size(0) // width_ratio
    prune_norm = nn.LayerNorm(norm_out)
    prune_norm.weight.data = norm.weight.data[:norm_out]
    return prune_norm

def PruneProj(proj, width_ratio): 
    proj_out = proj.weight.data.size(0) #prune
    proj_in = int(proj.weight.data.size(1) / width_ratio) #不prune
    prune_proj = nn.Linear(proj_in, proj_out)
    #print(prune_proj.weight.data.shape)
    if proj_in <= proj.weight.data.size(1):
        prune_proj.weight.data = proj.weight.data[:proj_out, :proj_in] #prune_proj.weight.data = proj.weight.data[:proj_out, :proj_in]
    else:
        prune_proj.weight.data = torch.repeat_interleave(proj.weight.data, repeats=int(1.0/width_ratio), dim=1)
    
    # print("prune_proj.weight.data.shape")
    #print(prune_proj.weight.data.shape)
    # print("prune_proj.bias.data")
    # print(prune_proj.bias.data.shape)
    # exit()
    # 对于bias（如果bias存在）
    new_bias = proj.bias.data if proj.bias is not None else None
    if new_bias is not None:
        prune_proj.bias.data = new_bias
    return prune_proj

class SharedAttention(Attention):
    def __init__(self, pretrained_attn, expand_method, dim, num_heads=8, shared_head_index=[0, 1, 2, 3], width_ratio=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__(dim, num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = PruneProj(pretrained_attn.proj, width_ratio)
        self.proj_drop = nn.Dropout(proj_drop)

        out_dim, in_dim = pretrained_attn.qkv.weight.data.shape

        # 浓缩len(shared_head_index)个attn head
        shared_heads = pretrained_attn.qkv.weight.data.reshape(num_heads, out_dim//num_heads, in_dim)[shared_head_index, :, :].clone()
        #print(shared_heads.data.shape)

        self.width_ratio = width_ratio
        self.num_heads =  int(num_heads // width_ratio)
        if expand_method == 'weight_assignment': #新参数用原参数赋值，共享相同的内存。
            
            expand_tensor = torch.zeros(self.num_heads, int(out_dim//num_heads), in_dim)
            shared_heads_idx = 0
            for i in range(self.num_heads):
                if i % len(shared_head_index) == 0:
                    shared_heads_idx = i % shared_heads.shape[0]

                expand_tensor[i, :, :] = shared_heads[shared_heads_idx, :, :]
                shared_heads_idx = (shared_heads_idx + 1) % shared_heads.shape[0]
            expanded_shared_heads = nn.Parameter(expand_tensor.reshape(int(out_dim // width_ratio), in_dim))
        
        elif expand_method == 'weight_clone': #新参数用原参数初始化，在内存中是独立的对象。
        # 按倍数扩展shared_heads，然后把多的排序靠后的裁剪掉 [2304,768]
            expanded_shared_heads = nn.Parameter(shared_heads.repeat(math.ceil(num_heads / len(shared_head_index)), 1, 1)[:num_heads, :, :].reshape(out_dim, in_dim) )   
        #self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) # [2304,768]
        #self.qkv.weight = expanded_shared_heads
        self.qkv = expanded_shared_heads

    def forward(self, x): 
        B, N, C = x.shape
        #qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = torch.matmul(x, self.qkv.T)  
        qkv = qkv.reshape(B, N, 3, self.num_heads, int(C // self.num_heads // self.width_ratio) ).permute(2, 0, 3, 1, 4)

        #print(qkv.shape)
        # Use the expanded shared_heads instead of the original qkv
        #qkv = self.qkv.unsqueeze(1).repeat(1, B, 1, 1, 1)
        
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, int(C // self.width_ratio) )
        # print("x.shape")
        # print(x.shape)
        # print("self.proj")
        # print(self.proj)
        #exit()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class SABlock(nn.Module):
    def __init__(self, shared_head_index, pretrained_block, inherited_mlp, expand_method, dim, width_ratio, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
          
        
        self.norm1 = pretrained_block.norm1#PruneNorm(pretrained_block.norm1, width_ratio)
        self.attn = SharedAttention(
            pretrained_block.attn, expand_method, dim=dim, num_heads=num_heads, shared_head_index=shared_head_index, width_ratio=width_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = pretrained_block.norm2#PruneNorm(pretrained_block.norm2, width_ratio)

        self.mlp = pretrained_block.mlp if inherited_mlp==None else inherited_mlp #PrunedMlp(pretrained_block.mlp, width_ratio)


    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

def transform_mlp(learngene_depth, descendant_depth, hdp_proj_bias):
    hdp_proj_layers = []
    _layer_1 = nn.Linear(
            learngene_depth,
            descendant_depth,
            bias=hdp_proj_bias,
        )
    hdp_proj_layers.append(_layer_1)
    hdp_proj_layers.append(nn.ReLU())
    _layer_2 = nn.Linear(
            descendant_depth,
            descendant_depth,
            bias=hdp_proj_bias,
        )
    hdp_proj_layers.append(_layer_2)
    hdp_proj = nn.Sequential(*hdp_proj_layers)
    return hdp_proj

class TransMlp(nn.Module):
    def __init__(self, descendant_fc1_weights, descendant_fc1_biases, descendant_fc2_weights, descendant_fc2_biases, in_features, hidden_features=None, act_layer=nn.GELU, drop=0., out_features=None):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        # print(descendant_fc1_biases)
        # self.fc1.weight = nn.Parameter(descendant_fc1_weights.T)
        # self.fc1.bias = nn.Parameter(descendant_fc1_biases)
        self.fc1.weight.data = descendant_fc1_weights
        self.fc1.bias.data = descendant_fc1_biases
        self.act = act_layer()
        # print(self.fc1.bias.data)

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.fc2.weight.data = descendant_fc2_weights
        self.fc2.bias.data = descendant_fc2_biases
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransMlp_Tiny(nn.Module):
    def __init__(self, depth, descendant_fc1_weights, descendant_fc1_biases, descendant_fc2_weights, descendant_fc2_biases, in_features, hidden_features=None, act_layer=nn.GELU, drop=0., out_features=None):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        if depth in [1,2]:
            self.fc1.weight.data = descendant_fc1_weights[depth-1]
            self.fc1.bias.data = descendant_fc1_biases[depth-1]
            self.fc2.weight.data = descendant_fc2_weights[depth-1]
            self.fc2.bias.data = descendant_fc2_biases[depth-1]
        elif depth in [10,11]: 
            self.fc1.weight.data = descendant_fc1_weights[depth-8]
            self.fc1.bias.data = descendant_fc1_biases[depth-8]
            self.fc2.weight.data = descendant_fc2_weights[depth-8]
            self.fc2.bias.data = descendant_fc2_biases[depth-8]

        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransMlp_Base(nn.Module):
    def __init__(self, depth, descendant_fc1_weights, descendant_fc1_biases, descendant_fc2_weights, descendant_fc2_biases, in_features, hidden_features=None, act_layer=nn.GELU, drop=0., out_features=None):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        if depth in [6,7]:
            self.fc1.weight.data = descendant_fc1_weights[depth-6]
            self.fc1.bias.data = descendant_fc1_biases[depth-6]
            self.fc2.weight.data = descendant_fc2_weights[depth-6]
            self.fc2.bias.data = descendant_fc2_biases[depth-6]
        elif depth in [9,10,11]: 
            self.fc1.weight.data = descendant_fc1_weights[depth-7]
            self.fc1.bias.data = descendant_fc1_biases[depth-7]
            self.fc2.weight.data = descendant_fc2_weights[depth-7]
            self.fc2.bias.data = descendant_fc2_biases[depth-7]

        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SAVisionTransformer(nn.Module): 
    """ Vision Transformer """
    def __init__(self, shared_head_index=[], pretrained_vit=[], expand_method='',
                 patch_size=16, in_chans=3, num_classes=0, depth=12, width_ratio=1, embed_dim=768, num_heads=12, mlp_ratio=4., qkv_bias=False, 
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):

        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = pretrained_vit.patch_embed
        self.cls_token = pretrained_vit.cls_token
        self.pos_embed = pretrained_vit.pos_embed
        self.pos_drop = pretrained_vit.pos_drop
        num_patches = self.patch_embed.num_patches

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        hdp_non_linear = True
        #扩展MLP数量
        if num_heads == 6:
            self.mlp_proj = transform_mlp(1, 3, bool(hdp_non_linear)).cuda()
            descendant_fc1_weights = self.mlp_proj(pretrained_vit.blocks[9].mlp.fc1.weight.unsqueeze(0).transpose(-3, -1)).transpose(-3, -1)
            descendant_fc1_biases = self.mlp_proj(pretrained_vit.blocks[9].mlp.fc1.bias.unsqueeze(0).transpose(-2, -1)).transpose(-2, -1)
            descendant_fc2_weights = self.mlp_proj(pretrained_vit.blocks[9].mlp.fc2.weight.unsqueeze(0).transpose(-3, -1)).transpose(-3, -1)
            descendant_fc2_biases = self.mlp_proj(pretrained_vit.blocks[9].mlp.fc2.bias.unsqueeze(0).transpose(-2, -1)).transpose(-2, -1)

            if width_ratio > 999:
                self.blocks = nn.ModuleList([
                    SABlock(
                        shared_head_index[i], pretrained_vit.blocks[i], 
                        TransMlp(descendant_fc1_weights[i-9], descendant_fc1_biases[i-9], descendant_fc2_weights[i-9], descendant_fc2_biases[i-9], embed_dim, int(embed_dim * mlp_ratio), nn.GELU, drop_rate) if i>=9 else None, 
                        expand_method, dim=embed_dim, width_ratio=2.0 if i < 4 else (1.5 if i < 8 else 1.0), 
                        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer
                    ) for i in range(depth)])

            elif width_ratio < 0:
                self.blocks = nn.ModuleList([
                    SABlock(
                        shared_head_index[i], pretrained_vit.blocks[i], 
                        TransMlp(descendant_fc1_weights[i-9], descendant_fc1_biases[i-9], descendant_fc2_weights[i-9], descendant_fc2_biases[i-9], embed_dim, int(embed_dim * mlp_ratio), nn.GELU, drop_rate) if i>=9 else None, 
                        expand_method, dim=embed_dim, width_ratio=1.0 if i < 4 else (1.5 if i < 8 else 2.0), 
                        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer
                    ) for i in range(depth)])

            else:
                self.blocks = nn.ModuleList([
                    SABlock(
                        shared_head_index[i], pretrained_vit.blocks[i],
                        TransMlp(descendant_fc1_weights[i-9], descendant_fc1_biases[i-9], descendant_fc2_weights[i-9], descendant_fc2_biases[i-9], embed_dim, int(embed_dim * mlp_ratio), nn.GELU, drop_rate) if i>=9 else None, 
                        expand_method, dim=embed_dim, width_ratio=width_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                    for i in range(depth)])

        elif num_heads == 3:
            self.mid_proj = transform_mlp(1, 2, bool(hdp_non_linear)).cuda()
            mid_fc1_weights = self.mid_proj(pretrained_vit.blocks[1].mlp.fc1.weight.unsqueeze(0).transpose(-3, -1)).transpose(-3, -1)
            mid_fc1_biases = self.mid_proj(pretrained_vit.blocks[1].mlp.fc1.bias.unsqueeze(0).transpose(-2, -1)).transpose(-2, -1)
            mid_fc2_weights = self.mid_proj(pretrained_vit.blocks[1].mlp.fc2.weight.unsqueeze(0).transpose(-3, -1)).transpose(-3, -1)
            mid_fc2_biases = self.mid_proj(pretrained_vit.blocks[1].mlp.fc2.bias.unsqueeze(0).transpose(-2, -1)).transpose(-2, -1)

            self.last_proj = transform_mlp(1, 2, bool(hdp_non_linear)).cuda()
            last_fc1_weights = self.last_proj(pretrained_vit.blocks[10].mlp.fc1.weight.unsqueeze(0).transpose(-3, -1)).transpose(-3, -1)
            last_fc1_biases = self.last_proj(pretrained_vit.blocks[10].mlp.fc1.bias.unsqueeze(0).transpose(-2, -1)).transpose(-2, -1)
            last_fc2_weights = self.last_proj(pretrained_vit.blocks[10].mlp.fc2.weight.unsqueeze(0).transpose(-3, -1)).transpose(-3, -1)
            last_fc2_biases = self.last_proj(pretrained_vit.blocks[10].mlp.fc2.bias.unsqueeze(0).transpose(-2, -1)).transpose(-2, -1)

            descendant_fc1_weights = torch.cat([mid_fc1_weights, last_fc1_weights], dim=0)
            descendant_fc1_biases = torch.cat([mid_fc1_biases, last_fc1_biases], dim=0)
            descendant_fc2_weights = torch.cat([mid_fc2_weights, last_fc2_weights], dim=0)
            descendant_fc2_biases = torch.cat([mid_fc2_biases, last_fc2_biases], dim=0)


            if width_ratio > 999:
                self.blocks = nn.ModuleList([
                    SABlock(
                        shared_head_index[i], pretrained_vit.blocks[i], 
                        TransMlp_Tiny(i, descendant_fc1_weights, descendant_fc1_biases, descendant_fc2_weights, descendant_fc2_biases, embed_dim, int(embed_dim * mlp_ratio), nn.GELU, drop_rate) if i in [1,2,10,11] else None, 
                        expand_method, dim=embed_dim, width_ratio=2.0 if i < 4 else (1.5 if i < 8 else 1.0), 
                        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer
                    ) for i in range(depth)])

            elif width_ratio < 0:
                self.blocks = nn.ModuleList([
                    SABlock(
                        shared_head_index[i], pretrained_vit.blocks[i], 
                        TransMlp_Tiny(i, descendant_fc1_weights, descendant_fc1_biases, descendant_fc2_weights, descendant_fc2_biases, embed_dim, int(embed_dim * mlp_ratio), nn.GELU, drop_rate) if i in [1,2,10,11] else None, 
                        expand_method, dim=embed_dim, width_ratio=1.0 if i < 4 else (1.5 if i < 8 else 2.0), 
                        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer
                    ) for i in range(depth)])

            else:
                self.blocks = nn.ModuleList([
                    SABlock(
                        shared_head_index[i], pretrained_vit.blocks[i],
                        TransMlp_Tiny(i, descendant_fc1_weights, descendant_fc1_biases, descendant_fc2_weights, descendant_fc2_biases, embed_dim, int(embed_dim * mlp_ratio), nn.GELU, drop_rate) if i in [1,2,10,11] else None,  
                        expand_method, dim=embed_dim, width_ratio=width_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                    for i in range(depth)])

        elif num_heads == 12:
            self.mid_proj = transform_mlp(1, 2, bool(hdp_non_linear)).cuda()
            mid_fc1_weights = self.mid_proj(pretrained_vit.blocks[6].mlp.fc1.weight.unsqueeze(0).transpose(-3, -1)).transpose(-3, -1)
            mid_fc1_biases = self.mid_proj(pretrained_vit.blocks[6].mlp.fc1.bias.unsqueeze(0).transpose(-2, -1)).transpose(-2, -1)
            mid_fc2_weights = self.mid_proj(pretrained_vit.blocks[6].mlp.fc2.weight.unsqueeze(0).transpose(-3, -1)).transpose(-3, -1)
            mid_fc2_biases = self.mid_proj(pretrained_vit.blocks[6].mlp.fc2.bias.unsqueeze(0).transpose(-2, -1)).transpose(-2, -1)

            self.last_proj = transform_mlp(1, 3, bool(hdp_non_linear)).cuda()
            last_fc1_weights = self.last_proj(pretrained_vit.blocks[9].mlp.fc1.weight.unsqueeze(0).transpose(-3, -1)).transpose(-3, -1)
            last_fc1_biases = self.last_proj(pretrained_vit.blocks[9].mlp.fc1.bias.unsqueeze(0).transpose(-2, -1)).transpose(-2, -1)
            last_fc2_weights = self.last_proj(pretrained_vit.blocks[9].mlp.fc2.weight.unsqueeze(0).transpose(-3, -1)).transpose(-3, -1)
            last_fc2_biases = self.last_proj(pretrained_vit.blocks[9].mlp.fc2.bias.unsqueeze(0).transpose(-2, -1)).transpose(-2, -1)

            descendant_fc1_weights = torch.cat([mid_fc1_weights, last_fc1_weights], dim=0)
            descendant_fc1_biases = torch.cat([mid_fc1_biases, last_fc1_biases], dim=0)
            descendant_fc2_weights = torch.cat([mid_fc2_weights, last_fc2_weights], dim=0)
            descendant_fc2_biases = torch.cat([mid_fc2_biases, last_fc2_biases], dim=0)


            if width_ratio > 999:
                self.blocks = nn.ModuleList([
                    SABlock(
                        shared_head_index[i], pretrained_vit.blocks[i], 
                        TransMlp_Base(i, descendant_fc1_weights, descendant_fc1_biases, descendant_fc2_weights, descendant_fc2_biases, embed_dim, int(embed_dim * mlp_ratio), nn.GELU, drop_rate) if i in [6,7,9,10,11] else None, 
                        expand_method, dim=embed_dim, width_ratio=2.0 if i < 4 else (1.5 if i < 8 else 1.0), 
                        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer
                    ) for i in range(depth)])

            elif width_ratio < 0:
                self.blocks = nn.ModuleList([
                    SABlock(
                        shared_head_index[i], pretrained_vit.blocks[i], 
                        TransMlp_Base(i, descendant_fc1_weights, descendant_fc1_biases, descendant_fc2_weights, descendant_fc2_biases, embed_dim, int(embed_dim * mlp_ratio), nn.GELU, drop_rate) if i in [6,7,9,10,11] else None, 
                        expand_method, dim=embed_dim, width_ratio=1.0 if i < 4 else (1.5 if i < 8 else 2.0), 
                        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer
                    ) for i in range(depth)])

            else:
                self.blocks = nn.ModuleList([
                    SABlock(
                        shared_head_index[i], pretrained_vit.blocks[i],
                        TransMlp_Base(i, descendant_fc1_weights, descendant_fc1_biases, descendant_fc2_weights, descendant_fc2_biases, embed_dim, int(embed_dim * mlp_ratio), nn.GELU, drop_rate) if i in [6,7,9,10,11] else None,  
                        expand_method, dim=embed_dim, width_ratio=width_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                    for i in range(depth)])

        self.norm = pretrained_vit.norm#PruneNorm(pretrained_vit.norm, width_ratio)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()#nn.Linear(embed_dim//width_ratio, num_classes) if num_classes > 0 else nn.Identity()

        #####继承不初始化####
        '''trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)'''

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
            align_corners=False,
            recompute_scale_factor=False
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x, ada_token=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        #x = nn.functional.avg_pool2d(x.unsqueeze(0), kernel_size=(1, 2), stride=(1, 2)).squeeze(0)
        
        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        #cls_tokens = nn.functional.avg_pool2d(cls_tokens.unsqueeze(0), kernel_size=(1, 2), stride=(1, 2)).squeeze(0)

        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        #x = x + nn.functional.avg_pool2d(self.interpolate_pos_encoding(x, w, h).unsqueeze(0), kernel_size=(1, 2), stride=(1, 2)).squeeze(0)
        x = x + self.interpolate_pos_encoding(x, w, h)

        if ada_token is not None:
            ada_tokens = ada_token.expand(B, -1, -1) # B, p, d
            x = torch.cat((x, ada_tokens), dim=1)

        return self.pos_drop(x)

    def forward(self, x, ada_token=None, use_patches=False):
        x = self.prepare_tokens(x, ada_token)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if use_patches:
            return x[:, 1:]
        else:
            return x[:, 0]

    def get_all_selfattention_with_name(self, x):
        x = self.prepare_tokens(x)
        dict_attn = {}
        for name, blk in self.blocks._modules.items():
        #for i, blk in enumerate(self.blocks):
        #    if (i+1)%number_block_per_part == 0:
        #attn.append(blk(x, return_attention=True) )
            dict_attn[name] = blk(x, return_attention=True)
            x = blk(x)
            
        # return attention of the all blocks
        return dict_attn




class vitBlock_with_learngene(nn.Module):
    def __init__(self, norm1, attn, drop_path, norm2, mlp, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm1#norm_layer(dim)
        self.attn = attn#Attention(
            #dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = drop_path#DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm2#norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = mlp#Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class vitsmall_with_learngene(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, inherited_layers_0 = [], inherited_layers_2 = [], method = '', patch_embed=[], cls_token=[], pos_embed=[], pos_drop=[], embed_dim=384, depth=12,
                 num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        if method == 'scratch':
            self.patch_embed = PatchEmbed(img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
            num_patches = self.patch_embed.num_patches
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)
        else:
            self.patch_embed = patch_embed
            self.cls_token = cls_token
            self.pos_embed = pos_embed
            self.pos_drop = pos_drop

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        if method == 'meta-learngene':
            self.blocks = nn.ModuleList([
                vitBlock_with_learngene(inherited_layers_0[i].norm1, inherited_layers_0[i].attn, inherited_layers_0[i].drop_path, inherited_layers_0[i].norm2, inherited_layers_0[i].mlp,
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer)
                for i in range(2)])

            for i in range(4):
                self.blocks.append( vitBlock_with_learngene(inherited_layers_2[i].norm1, inherited_layers_2[i].attn, inherited_layers_2[i].drop_path, inherited_layers_2[i].norm2, inherited_layers_2[i].mlp,
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer) )
        
        elif method == 'front-learngene':
            self.blocks = nn.ModuleList([
                vitBlock_with_learngene(inherited_layers_0[i].norm1, inherited_layers_0[i].attn, inherited_layers_0[i].drop_path, inherited_layers_0[i].norm2, inherited_layers_0[i].mlp,
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer)
                for i in range(6)])
            
        elif method == 'orig-learngene':
            self.blocks = nn.ModuleList([
                vitBlock_with_learngene(inherited_layers_2[i].norm1, inherited_layers_2[i].attn, inherited_layers_2[i].drop_path, inherited_layers_2[i].norm2, inherited_layers_2[i].mlp,
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer)
                for i in range(6)])
            
        elif method == 'scratch':
            self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(6)])
            
        

        self.norm = norm_layer(embed_dim)
        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        if method == 'scratch':
            trunc_normal_(self.pos_embed, std=.02)
            trunc_normal_(self.cls_token, std=.02)
            self.apply(self._init_weights)
        #####不初始化####trunc_normal_(self.pos_embed, std=.02)
        #####不初始化####trunc_normal_(self.cls_token, std=.02)
        #####不初始化####self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
            align_corners=False,
            recompute_scale_factor=False
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x, ada_token=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        if ada_token is not None:
            ada_tokens = ada_token.expand(B, -1, -1) # B, p, d
            x = torch.cat((x, ada_tokens), dim=1)

        return self.pos_drop(x)

    def forward(self, x, ada_token=None, use_patches=False):
        x = self.prepare_tokens(x, ada_token)
        for blk in self.blocks:
            #print(x.shape)
            #print(blk)
            x = blk(x)
        x = self.norm(x)

        if use_patches:
            return x[:, 1:]
        else:
            return x[:, 0]
        
    def extract_feature(self, x, number_block_per_part, ada_token=None):
        x = self.prepare_tokens(x, ada_token)
        feat = []
        cls_token = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i+1)%number_block_per_part ==0 :
                feat.append(x[:, 1:]) # remove cls token
                cls_token.append(x[:, 0])
        x = self.norm(x)

        return feat, cls_token, x[:, 0]



class WeightTransformAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.conv_l = nn.Conv2d(num_heads, num_heads, kernel_size=1)  # Weight transform layers
        self.conv_w = nn.Conv2d(num_heads, num_heads, kernel_size=1)  # Weight transform layers

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.conv_l(attn)  # Weight transform before softmax
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = self.conv_w(attn)  # Weight transform after softmax
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class WeightTransformBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = WeightTransformAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class WeightTransform_VisionTransformer(nn.Module):
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            WeightTransformBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
            align_corners=False,
            recompute_scale_factor=False
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x, ada_token=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        if ada_token is not None:
            ada_tokens = ada_token.expand(B, -1, -1) # B, p, d
            x = torch.cat((x, ada_tokens), dim=1)

        return self.pos_drop(x)

    def forward(self, x, ada_token=None, use_patches=False):
        x = self.prepare_tokens(x, ada_token)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if use_patches:
            return x[:, 1:]
        else:
            return x[:, 0]

    def extract_feature(self, x, number_block_per_part, ada_token=None):
        x = self.prepare_tokens(x, ada_token)
        feat = []
        cls_token = []
        patch_and_cls_token = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i+1)%number_block_per_part == 0:
                feat.append(x[:, 1:]) # remove cls token
                cls_token.append(x[:, 0])
                patch_and_cls_token.append(x)
        x = self.norm(x)

        return feat, cls_token, patch_and_cls_token


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_tiny_depth2(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=2, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_tiny_depth3(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=3, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_tiny_depth4(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=4, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_tiny_depth6(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=6, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_tiny_depth8(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=8, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_tiny_depth9(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=9, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_tiny_depth10(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=10, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small_depth2(patch_size=16, num_classes=0, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, num_classes=num_classes, embed_dim=384, depth=2, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small_depth3(patch_size=16, num_classes=0, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, num_classes=num_classes, embed_dim=384, depth=3, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small_depth4(patch_size=16, num_classes=0, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, num_classes=num_classes, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small_depth6(patch_size=16, num_classes=0, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, num_classes=num_classes, embed_dim=384, depth=6, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small_depth8(patch_size=16, num_classes=0, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, num_classes=num_classes, embed_dim=384, depth=8, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small_depth9(patch_size=16, num_classes=0, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, num_classes=num_classes, embed_dim=384, depth=9, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small_depth10(patch_size=16, num_classes=0, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, num_classes=num_classes, embed_dim=384, depth=10, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small_depth6_shared(patch_size=16, num_classes=0, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, num_classes=num_classes, embed_dim=384, depth=6, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small_depth6_shared_residual(residual_interval=3, patch_size=16, num_classes=0, **kwargs):
    model = VisionTransformer_Residual(residual_interval=residual_interval, 
        patch_size=patch_size, num_classes=num_classes, embed_dim=384, depth=6, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small_weight_transform_depth6(patch_size=16, num_classes=0, **kwargs):
    model = WeightTransform_VisionTransformer(
        patch_size=patch_size, num_classes=num_classes, embed_dim=384, depth=6, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small_weight_transform_depth12(patch_size=16, num_classes=0, **kwargs):
    model = WeightTransform_VisionTransformer(
        patch_size=patch_size, num_classes=num_classes, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_depth3(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=3, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_depth6(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=6, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_depth10(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=10, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def savit_base(shared_head_index, pretrained_vit, expand_method, patch_size, depth, width_ratio, **kwargs):
    model = SAVisionTransformer(shared_head_index=shared_head_index, pretrained_vit=pretrained_vit, expand_method=expand_method,
        patch_size=patch_size,  depth=depth, width_ratio=width_ratio, embed_dim=768, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def savit_base_d9(shared_head_index, pretrained_vit, expand_method, patch_size, depth, width_ratio, **kwargs):
    model = SAVisionTransformer(shared_head_index=shared_head_index, pretrained_vit=pretrained_vit, expand_method=expand_method,
        patch_size=patch_size,  depth=9, width_ratio=width_ratio, embed_dim=768, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def savit_base_d10(shared_head_index, pretrained_vit, expand_method, patch_size, depth, width_ratio, **kwargs):
    model = SAVisionTransformer(shared_head_index=shared_head_index, pretrained_vit=pretrained_vit, expand_method=expand_method,
        patch_size=patch_size,  depth=10, width_ratio=width_ratio, embed_dim=768, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def savit_base_d11(shared_head_index, pretrained_vit, expand_method, patch_size, depth, width_ratio, **kwargs):
    model = SAVisionTransformer(shared_head_index=shared_head_index, pretrained_vit=pretrained_vit, expand_method=expand_method,
        patch_size=patch_size,  depth=11, width_ratio=width_ratio, embed_dim=768, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def savit_small(shared_head_index, pretrained_vit, expand_method, patch_size, depth, width_ratio, **kwargs):
    model = SAVisionTransformer(shared_head_index=shared_head_index, pretrained_vit=pretrained_vit, expand_method=expand_method,
        patch_size=patch_size,  depth=depth, width_ratio=width_ratio, embed_dim=384, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def savit_small_d6(shared_head_index, pretrained_vit, expand_method, patch_size, depth, width_ratio, **kwargs):
    model = SAVisionTransformer(shared_head_index=shared_head_index, pretrained_vit=pretrained_vit, expand_method=expand_method,
        patch_size=patch_size,  depth=6, width_ratio=width_ratio, embed_dim=384, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
 
def savit_small_d9(shared_head_index, pretrained_vit, expand_method, patch_size, depth, width_ratio, **kwargs):
    print("savit_small_d9=====================")
    model = SAVisionTransformer(shared_head_index=shared_head_index, pretrained_vit=pretrained_vit, expand_method=expand_method,
        patch_size=patch_size,  depth=9, width_ratio=width_ratio, embed_dim=384, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def savit_small_d10(shared_head_index, pretrained_vit, expand_method, patch_size, depth, width_ratio, **kwargs):
    print("savit_small_d10=====================")
    model = SAVisionTransformer(shared_head_index=shared_head_index, pretrained_vit=pretrained_vit, expand_method=expand_method,
        patch_size=patch_size,  depth=10, width_ratio=width_ratio, embed_dim=384, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def savit_small_d11(shared_head_index, pretrained_vit, expand_method, patch_size, depth, width_ratio, **kwargs):
    model = SAVisionTransformer(shared_head_index=shared_head_index, pretrained_vit=pretrained_vit, expand_method=expand_method,
        patch_size=patch_size,  depth=11, width_ratio=width_ratio, embed_dim=384, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def savit_tiny(shared_head_index, pretrained_vit, expand_method, patch_size, depth, width_ratio, **kwargs):
    model = SAVisionTransformer(shared_head_index=shared_head_index, pretrained_vit=pretrained_vit, expand_method=expand_method,
        patch_size=patch_size,  depth=depth, width_ratio=width_ratio, embed_dim=192, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def savit_tiny_d9(shared_head_index, pretrained_vit, expand_method, patch_size, depth, width_ratio, **kwargs):
    model = SAVisionTransformer(shared_head_index=shared_head_index, pretrained_vit=pretrained_vit, expand_method=expand_method,
        patch_size=patch_size,  depth=9, width_ratio=width_ratio, embed_dim=192, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def savit_tiny_d10(shared_head_index, pretrained_vit, expand_method, patch_size, depth, width_ratio, **kwargs):
    model = SAVisionTransformer(shared_head_index=shared_head_index, pretrained_vit=pretrained_vit, expand_method=expand_method,
        patch_size=patch_size,  depth=10, width_ratio=width_ratio, embed_dim=192, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def savit_tiny_d11(shared_head_index, pretrained_vit, expand_method, patch_size, depth, width_ratio, **kwargs):
    model = SAVisionTransformer(shared_head_index=shared_head_index, pretrained_vit=pretrained_vit, expand_method=expand_method,
        patch_size=patch_size,  depth=11, width_ratio=width_ratio, embed_dim=192, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
