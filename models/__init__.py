import os
import numpy as np
import torch
import torch.nn as nn
from functools import partial

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
#from .protonet import ProtoNet
#from .deploy import ProtoNet_Finetune, ProtoNet_Auto_Finetune, ProtoNet_AdaTok, ProtoNet_AdaTok_EntMin


def get_backbone(arch, pretrained, args):
    if arch == 'vit_base_patch16_224_in21k':
        from .vit_google import VisionTransformer, CONFIGS

        config = CONFIGS['ViT-B_16']
        model = VisionTransformer(config, 224)

        url = 'https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz'
        pretrained_weights = 'pretrained_ckpts/vit_base_patch16_224_in21k.npz'

        if not os.path.exists(pretrained_weights):
            try:
                import wget
                os.makedirs('pretrained_ckpts', exist_ok=True)
                wget.download(url, pretrained_weights)
            except:
                print(f'Cannot download pretrained weights from {url}. Check if `pip install wget` works.')

        model.load_from(np.load(pretrained_weights))
        print('Pretrained weights found at {}'.format(pretrained_weights))

    elif arch == 'dino_base_patch16':
        from . import vision_transformer as vit

        model = vit.__dict__['vit_base'](patch_size=16, num_classes=0)

        url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)

        model.load_state_dict(state_dict, strict=True)
        print('Pretrained weights found at {}'.format(url))

    elif arch == "deit_base_distilled_patch16_384":
        model = DistilledVisionTransformer(
            img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.default_cfg = _cfg()
        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
                map_location="cpu", check_hash=True
            )
            model.load_state_dict(checkpoint["model"])

    elif arch == 'deit_base_patch16_224':
        
        # use timm
        '''model = create_model(
            arch,
            pretrained=pretrained,
            num_classes=args.nb_classes
        )'''
        
        '''from .mini_vision_transformer import VisionTransformer, _cfg
        model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        model.default_cfg = _cfg()'''
        
        from . import vision_transformer as vit
        # if args.expand_method == "auto-learngene" or args.expand_method == "heuristic-learngene":
        #     model = vit.__dict__['vit_base'](patch_size=16, num_classes=args.nb_classes)
        # elif args.expand_method == "weight_assignment" or args.expand_method == "weight_clone":
        #     model = vit.__dict__['vit_base'](patch_size=16, num_classes=0)
        model = vit.__dict__['vit_base'](patch_size=16, num_classes=args.nb_classes)
        #state_dict = torch.load('./models/checkpoint/deit_base_patch16_224-b5f2ef4d.pth')['model']
        
        url = "https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth"  # DeiT-base
        #url = "https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth"  # DeiT-base distilled 384 (1000 epochs)
        state_dict = torch.hub.load_state_dict_from_url(url=url)
        # state_dict = torch.load('./models/checkpoint/deit_base_patch16_224-b5f2ef4d.pth')

        if args.expand_method == "auto-learngene":
            keys_to_remove = ['blocks.6.', 'blocks.7.', 'blocks.8.', 'blocks.9.', 'blocks.10.', 'blocks.11.']
            state_dict['model'] = {k: v for k, v in state_dict['model'].items() if not any(key in k for key in keys_to_remove)}
            print(state_dict['model'].keys())
        elif args.expand_method == "heuristic-learngene":
            keys_to_remove = ['blocks.0.', 'blocks.1.', 'blocks.2.', 'blocks.3.', 'blocks.4.', 'blocks.5.', 'blocks.6.', 'blocks.7.', 'blocks.8.']
            state_dict['model'] = {k: v for k, v in state_dict['model'].items() if not any(key in k for key in keys_to_remove)}
            print(state_dict['model'].keys())    
        elif args.expand_method == "front10":
            keys_to_remove = ['blocks.10.', 'blocks.11.']
            state_dict = {k: v for k, v in state_dict.items() if not any(key in k for key in keys_to_remove)}
            print(state_dict.keys())
        elif args.expand_method == "middle10":
            keys_to_remove = ['blocks.0.', 'blocks.11.']
            state_dict = {k: v for k, v in state_dict.items() if not any(key in k for key in keys_to_remove)}
            print(state_dict.keys())
        elif args.expand_method == "last10":
            keys_to_remove = ['blocks.0.', 'blocks.1.']
            state_dict = {k: v for k, v in state_dict.items() if not any(key in k for key in keys_to_remove)}
            print(state_dict.keys())
        elif args.expand_method == "last6":
            keys_to_remove = ['blocks.0.', 'blocks.1.', 'blocks.2.', 'blocks.3.', 'blocks.4.', 'blocks.5.']
            state_dict = {k: v for k, v in state_dict.items() if not any(key in k for key in keys_to_remove)}
            print(state_dict.keys())
        
        #elif args.expand_method == "weight_assignment" or args.expand_method == "weight_clone":

        for k in ['head.weight', 'head.bias']:
            if k in state_dict['model']:
                print(f"removing key {k} from pretrained checkpoint")
                del state_dict['model'][k]

        model.load_state_dict(state_dict['model'], strict=False)
        print('Pretrained weights found at {}'.format(url))
        
        #model.load_state_dict(state_dict, strict=True)
        #print('Successfully load deit_base_patch16_224')

    elif arch == 'deit_base_d10_patch16_224':
        
        from . import vision_transformer as vit
        # if args.expand_method == "auto-learngene" or args.expand_method == "heuristic-learngene":
        #     model = vit.__dict__['vit_base'](patch_size=16, num_classes=args.nb_classes)
        # elif args.expand_method == "weight_assignment" or args.expand_method == "weight_clone":
        #     model = vit.__dict__['vit_base'](patch_size=16, num_classes=0)
        model = vit.__dict__['vit_base_depth10'](patch_size=16, num_classes=args.nb_classes)
        #state_dict = torch.load('./models/checkpoint/deit_base_patch16_224-b5f2ef4d.pth')['model']
        
        url = "https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth"  # DeiT-base
        #url = "https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth"  # DeiT-base distilled 384 (1000 epochs)
        state_dict = torch.hub.load_state_dict_from_url(url=url)
        # state_dict = torch.load('./models/checkpoint/deit_base_patch16_224-b5f2ef4d.pth')

        for k in ['head.weight', 'head.bias']:
            if k in state_dict['model']:
                print(f"removing key {k} from pretrained checkpoint")
                del state_dict['model'][k]

        model.load_state_dict(state_dict['model'], strict=False)
        print('Pretrained weights found at {}'.format(url))
        
        #model.load_state_dict(state_dict, strict=True)
        #print('Successfully load deit_base_patch16_224')

    elif arch == 'deit_base_w/o_ffn_patch16_224':

        from . import vision_transformer as vit
        model = vit.__dict__['vit_base'](patch_size=16, num_classes=0)
        
        url = "https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth"  # DeiT-base
        state_dict = torch.hub.load_state_dict_from_url(url=url)

        for k in ['head.weight', 'head.bias']:
            if k in state_dict['model']:
                print(f"removing key {k} from pretrained checkpoint")
                del state_dict['model'][k]
        
        state_dict['model'] = {k: v for k, v in state_dict['model'].items() if 'mlp' not in k}

        print(state_dict['model'].keys())
        inherit_params = sum(p.numel() for p in state_dict['model'].values())
        print(f"Inheriting parameters in the model: {inherit_params}")
        inherit_params_in_M = inherit_params / 1e6
        print(f"Inheriting parameters in the model: {inherit_params_in_M:.2f}M")

        model.load_state_dict(state_dict['model'], strict=False)
        print('Pretrained weights found at {}'.format(url))

    elif arch == 'deit_base_w/o_pretrained_patch16_224':
        from . import vision_transformer as vit
        model = vit.__dict__['vit_base'](patch_size=16, num_classes=args.nb_classes)


    elif arch == 'deit_small_w/o_ffn_patch16_224':
        from . import vision_transformer as vit
        model = vit.__dict__['vit_small'](patch_size=16, num_classes=args.nb_classes)
        
        # descendant model directly trains
        url = "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth"
        state_dict = torch.hub.load_state_dict_from_url(url=url)["model"]

        for k in ['head.weight', 'head.bias']:
            if k in state_dict:
                print(f"removing key {k} from pretrained checkpoint")
                del state_dict[k]
        
        state_dict = {k: v for k, v in state_dict.items() if 'mlp' not in k}

        print(state_dict.keys())
        inherit_params = sum(p.numel() for p in state_dict.values())
        print(f"Inheriting parameters in the model: {inherit_params}")
        inherit_params_in_M = inherit_params / 1e6
        print(f"Inheriting parameters in the model: {inherit_params_in_M:.2f}M")

        model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {}'.format(url))

    elif arch == 'deit_tiny_w/o_ffn_patch16_224':
        from . import vision_transformer as vit
        model = vit.__dict__['vit_tiny'](patch_size=16, num_classes=args.nb_classes)
        
        # descendant model directly trains
        url = "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth"
        state_dict = torch.hub.load_state_dict_from_url(url=url)["model"]

        for k in ['head.weight', 'head.bias']:
            if k in state_dict:
                print(f"removing key {k} from pretrained checkpoint")
                del state_dict[k]
        
        state_dict = {k: v for k, v in state_dict.items() if 'mlp' not in k}

        print(state_dict.keys())
        inherit_params = sum(p.numel() for p in state_dict.values())
        print(f"Inheriting parameters in the model: {inherit_params}")
        inherit_params_in_M = inherit_params / 1e6
        print(f"Inheriting parameters in the model: {inherit_params_in_M:.2f}M")

        model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {}'.format(url))

    elif arch == 'deit_small_patch16_224':
        from . import vision_transformer as vit
        model = vit.__dict__['vit_small'](patch_size=16, num_classes=args.nb_classes)
        
        # descendant model directly trains
        url = "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth"
        state_dict = torch.hub.load_state_dict_from_url(url=url)["model"]

        if args.expand_method == "auto-learngene":
            keys_to_remove = ['blocks.6.', 'blocks.7.', 'blocks.8.', 'blocks.9.', 'blocks.10.', 'blocks.11.']
            state_dict = {k: v for k, v in state_dict.items() if not any(key in k for key in keys_to_remove)}
            print(state_dict.keys())
        elif args.expand_method == "heuristic-learngene":
            keys_to_remove = ['blocks.0.', 'blocks.1.', 'blocks.2.', 'blocks.3.', 'blocks.4.', 'blocks.5.', 'blocks.6.', 'blocks.7.', 'blocks.8.']
            state_dict = {k: v for k, v in state_dict.items() if not any(key in k for key in keys_to_remove)}
            print(state_dict.keys())  
        elif args.expand_method == "front10":
            keys_to_remove = ['blocks.10.', 'blocks.11.']
            state_dict = {k: v for k, v in state_dict.items() if not any(key in k for key in keys_to_remove)}
            print(state_dict.keys())
        elif args.expand_method == "middle10":
            keys_to_remove = ['blocks.0.', 'blocks.11.']
            state_dict = {k: v for k, v in state_dict.items() if not any(key in k for key in keys_to_remove)}
            print(state_dict.keys())
        elif args.expand_method == "last10":
            keys_to_remove = ['blocks.0.', 'blocks.1.']
            state_dict = {k: v for k, v in state_dict.items() if not any(key in k for key in keys_to_remove)}
            print(state_dict.keys())
        elif args.expand_method == "last6":
            keys_to_remove = ['blocks.0.', 'blocks.1.', 'blocks.2.', 'blocks.3.', 'blocks.4.', 'blocks.5.']
            state_dict = {k: v for k, v in state_dict.items() if not any(key in k for key in keys_to_remove)}
            print(state_dict.keys())

        for k in ['head.weight', 'head.bias']:
            if k in state_dict:
                print(f"removing key {k} from pretrained checkpoint")
                del state_dict[k]

        model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {}'.format(url))
        
        #model.load_state_dict(torch.load('./models/checkpoint/deit_small_patch16_224-cd65a155.pth'), strict=True)

    elif arch == 'deit_small_d10_patch16_224':
        from . import vision_transformer as vit
        model = vit.__dict__['vit_small_depth10'](patch_size=16, num_classes=args.nb_classes)
        
        # descendant model directly trains
        url = "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth"
        state_dict = torch.hub.load_state_dict_from_url(url=url)["model"]

        for k in ['head.weight', 'head.bias']:
            if k in state_dict:
                print(f"removing key {k} from pretrained checkpoint")
                del state_dict[k]

        model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {}'.format(url))
    
    elif arch == 'deit_tiny_patch16_224':
        from . import vision_transformer as vit
        model = vit.__dict__['vit_tiny'](patch_size=16, num_classes=args.nb_classes)
        
        # descendant model directly trains
        url = "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth"
        state_dict = torch.hub.load_state_dict_from_url(url=url)["model"]

        if args.expand_method == "auto-learngene":
            keys_to_remove = ['blocks.6.', 'blocks.7.', 'blocks.8.', 'blocks.9.', 'blocks.10.', 'blocks.11.']
            state_dict = {k: v for k, v in state_dict.items() if not any(key in k for key in keys_to_remove)}
            print(state_dict.keys())
        elif args.expand_method == "heuristic-learngene":
            keys_to_remove = ['blocks.0.', 'blocks.1.', 'blocks.2.', 'blocks.3.', 'blocks.4.', 'blocks.5.', 'blocks.6.', 'blocks.7.', 'blocks.8.']
            state_dict = {k: v for k, v in state_dict.items() if not any(key in k for key in keys_to_remove)}
            print(state_dict.keys())  
        elif args.expand_method == "front10":
            keys_to_remove = ['blocks.10.', 'blocks.11.']
            state_dict = {k: v for k, v in state_dict.items() if not any(key in k for key in keys_to_remove)}
            print(state_dict.keys())
        elif args.expand_method == "middle10":
            keys_to_remove = ['blocks.0.', 'blocks.11.']
            state_dict = {k: v for k, v in state_dict.items() if not any(key in k for key in keys_to_remove)}
            print(state_dict.keys())
        elif args.expand_method == "last10":
            keys_to_remove = ['blocks.0.', 'blocks.1.']
            state_dict = {k: v for k, v in state_dict.items() if not any(key in k for key in keys_to_remove)}
            print(state_dict.keys())
        elif args.expand_method == "last6":
            keys_to_remove = ['blocks.0.', 'blocks.1.', 'blocks.2.', 'blocks.3.', 'blocks.4.', 'blocks.5.']
            state_dict = {k: v for k, v in state_dict.items() if not any(key in k for key in keys_to_remove)}
            print(state_dict.keys())

        for k in ['head.weight', 'head.bias']:
            if k in state_dict:
                print(f"removing key {k} from pretrained checkpoint")
                del state_dict[k]
        
        model.load_state_dict(state_dict, strict=False)
        print('Successfully load deit_tiny_patch16_224')

        
        '''model = create_model(
            arch,
            pretrained=pretrained,
            num_classes=args.nb_classes
        )'''

    elif arch == 'deit_tiny_d10_patch16_224':
        from . import vision_transformer as vit
        model = vit.__dict__['vit_tiny_depth10'](patch_size=16, num_classes=args.nb_classes)
        
        # descendant model directly trains
        url = "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth"
        state_dict = torch.hub.load_state_dict_from_url(url=url)["model"]

        for k in ['head.weight', 'head.bias']:
            if k in state_dict:
                print(f"removing key {k} from pretrained checkpoint")
                del state_dict[k]
        
        model.load_state_dict(state_dict, strict=False)
        print('Successfully load deit_tiny_patch16_224')

        
        '''model = create_model(
            arch,
            pretrained=pretrained,
            num_classes=args.nb_classes
        )'''

    elif arch == 'deit_tiny_depth2_patch16_224':
        from . import vision_transformer as vit
        model = vit.__dict__['vit_tiny_depth2'](patch_size=16, num_classes=args.nb_classes)
    
    elif arch == 'deit_tiny_depth3_patch16_224':
        from . import vision_transformer as vit
        model = vit.__dict__['vit_tiny_depth3'](patch_size=16, num_classes=args.nb_classes)
    
    elif arch == 'deit_tiny_depth4_patch16_224':
        from . import vision_transformer as vit
        model = vit.__dict__['vit_tiny_depth4'](patch_size=16, num_classes=args.nb_classes)
    
    elif arch == 'deit_tiny_depth6_patch16_224':
        from . import vision_transformer as vit
        model = vit.__dict__['vit_tiny_depth6'](patch_size=16, num_classes=args.nb_classes)
    
    elif arch == 'deit_tiny_depth8_patch16_224':
        from . import vision_transformer as vit
        model = vit.__dict__['vit_tiny_depth8'](patch_size=16, num_classes=args.nb_classes)
    
    elif arch == 'deit_tiny_depth9_patch16_224':
        from . import vision_transformer as vit
        model = vit.__dict__['vit_tiny_depth9'](patch_size=16, num_classes=args.nb_classes)
    
    
    elif arch == 'deit_small_depth2_patch16_224':
        from . import vision_transformer as vit
        model = vit.__dict__['vit_small_depth2'](patch_size=16, num_classes=args.nb_classes)
    
    elif arch == 'deit_small_depth3_patch16_224':
        from . import vision_transformer as vit
        model = vit.__dict__['vit_small_depth3'](patch_size=16, num_classes=args.nb_classes)
    
    elif arch == 'deit_small_depth4_patch16_224':
        from . import vision_transformer as vit
        model = vit.__dict__['vit_small_depth4'](patch_size=16, num_classes=args.nb_classes)
    
    elif arch == 'deit_small_depth6_patch16_224':
        from . import vision_transformer as vit
        model = vit.__dict__['vit_small_depth6'](patch_size=16, num_classes=args.nb_classes)
    
    elif arch == 'deit_small_depth8_patch16_224':
        from . import vision_transformer as vit
        model = vit.__dict__['vit_small_depth8'](patch_size=16, num_classes=args.nb_classes)
    
    elif arch == 'deit_small_depth9_patch16_224':
        from . import vision_transformer as vit
        model = vit.__dict__['vit_small_depth9'](patch_size=16, num_classes=args.nb_classes)
    
    elif arch == 'deit_small_shared_depth6_patch16_224':
        from . import vision_transformer as vit
        # Create a 6-layer model
        model = vit.__dict__['vit_small_depth6_shared'](patch_size=16, num_classes=args.nb_classes)
    
    elif arch == 'deit_small_shared_depth6_residual_3_patch16_224':
        from . import vision_transformer as vit
        # Create a 6-layer model
        model = vit.__dict__['vit_small_depth6_shared_residual'](residual_interval=3, patch_size=16, num_classes=args.nb_classes)

    elif arch == 'deit_base_depth3_patch16_224':
        from . import vision_transformer as vit
        model = vit.__dict__['vit_base_depth3'](patch_size=16, num_classes=args.nb_classes)

    elif arch == 'deit_base_depth6_patch16_224':
        from . import vision_transformer as vit
        model = vit.__dict__['vit_base_depth6'](patch_size=16, num_classes=args.nb_classes)

    elif arch == 'LeViT_256':
        from . import levit
        model = levit.__dict__['LeViT_256'](pretrained=True)

    elif arch == 'LeViT_384':
        from . import levit
        model = levit.__dict__['LeViT_384'](pretrained=True)

    elif arch == 'deit_small_weight_transform_depth6_patch16_224':
        from . import vision_transformer as vit
        # Create a 6-layer model
        model = vit.__dict__['vit_small_weight_transform_depth6'](patch_size=16, num_classes=args.nb_classes)

    elif arch == 'deit_small_weight_transform_depth12_patch16_224':
        from . import vision_transformer as vit
        # Create a 12-layer model
        model = vit.__dict__['vit_small_weight_transform_depth12'](patch_size=16, num_classes=args.nb_classes)

    elif arch == 'dino_small_patch16':
        from . import vision_transformer as vit

        model = vit.__dict__['vit_small'](patch_size=16, num_classes=0)

        #if not args.no_pretrain:
        #url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        #state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)

        model.load_state_dict(torch.load('./models/checkpoint/dino_deitsmall16_pretrain.pth'), strict=True)
        #print('Pretrained weights found at {}'.format(url))

    elif arch == 'beit_base_patch16_224_pt22k':
        from .beit import default_pretrained_model 
        model = default_pretrained_model(args)
        print('Pretrained BEiT loaded')

    elif arch == 'clip_base_patch16_224':
        from . import clip
        model, _ = clip.load('ViT-B/16', 'cpu')

    elif arch == 'clip_resnet50':
        from . import clip
        model, _ = clip.load('RN50', 'cpu')

    elif arch == 'dino_resnet50':
        from torchvision.models.resnet import resnet50

        model = resnet50(pretrained=False)
        model.fc = torch.nn.Identity()

        if not args.no_pretrain:
            state_dict = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth",
                map_location="cpu",
            )
            model.load_state_dict(state_dict, strict=False)

    elif arch == 'resnet50':
        from torchvision.models.resnet import resnet50

        pretrained = not args.no_pretrain
        model = resnet50(pretrained=pretrained)
        model.fc = torch.nn.Identity()

    elif arch == 'resnet18':
        from torchvision.models.resnet import resnet18

        pretrained = not args.no_pretrain
        model = resnet18(pretrained=pretrained)
        model.fc = torch.nn.Identity()

    elif arch == 'dino_xcit_medium_24_p16':
        model = torch.hub.load('facebookresearch/xcit:main', 'xcit_medium_24_p16')
        model.head = torch.nn.Identity()
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain.pth",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=False)

    elif arch == 'dino_xcit_medium_24_p8':
        model = torch.hub.load('facebookresearch/dino:main', 'dino_xcit_medium_24_p8')

    elif arch == 'simclrv2_resnet50':
        import sys
        sys.path.insert(
            0,
            'cog',
        )
        import model_utils

        model_utils.MODELS_ROOT_DIR = 'cog/models'
        ckpt_file = os.path.join(args.pretrained_checkpoint_path, 'pretrained_ckpts/simclrv2_resnet50.pth')
        resnet, _ = model_utils.load_pretrained_backbone(args.arch, ckpt_file)

        class Wrapper(torch.nn.Module):
            def __init__(self, model):
                super(Wrapper, self).__init__()
                self.model = model

            def forward(self, x):
                return self.model(x, apply_fc=False)

        model = Wrapper(resnet)

    elif arch in ['mocov2_resnet50', 'swav_resnet50', 'barlow_resnet50']:
        from torchvision.models.resnet import resnet50

        model = resnet50(pretrained=False)
        ckpt_file = os.path.join(args.pretrained_checkpoint_path, 'pretrained_ckpts_converted/{}.pth'.format(args.arch))
        ckpt = torch.load(ckpt_file)

        msg = model.load_state_dict(ckpt, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

        # remove the fully-connected layer
        model.fc = torch.nn.Identity()


    elif arch == 'swin_base_patch4_window7_224':
        
        from . import model_transformer as vit
        model = vit.__dict__['swin_base_patch4_window7_224'](num_classes=0)
        state_dict = torch.load('./checkpoint/swin/swin_base_patch4_window7_224.pth')
        
        #url = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth"  # swin-base
        # state_dict = torch.hub.load_state_dict_from_url(url=url)


        # if args.expand_method == "auto-learngene":
        #     keys_to_remove = ['blocks.6.', 'blocks.7.', 'blocks.8.', 'blocks.9.', 'blocks.10.', 'blocks.11.']
        #     state_dict['model'] = {k: v for k, v in state_dict['model'].items() if not any(key in k for key in keys_to_remove)}
        #     print(state_dict['model'].keys())
        # elif args.expand_method == "heuristic-learngene":
        #     keys_to_remove = ['blocks.0.', 'blocks.1.', 'blocks.2.', 'blocks.3.', 'blocks.4.', 'blocks.5.', 'blocks.6.', 'blocks.7.', 'blocks.8.']
        #     state_dict['model'] = {k: v for k, v in state_dict['model'].items() if not any(key in k for key in keys_to_remove)}
        #     print(state_dict['model'].keys())    
        
        #elif args.expand_method == "weight_assignment" or args.expand_method == "weight_clone":

        for k in ['head.weight', 'head.bias']:
            if k in state_dict['model']:
                print(f"removing key {k} from pretrained checkpoint")
                del state_dict['model'][k]

        model.load_state_dict(state_dict['model'], strict=False)
        #print('Pretrained weights found at {}'.format(url))
        print('Pretrained weights found at checkpoint')
        
        #model.load_state_dict(state_dict, strict=True)
        #print('Successfully load deit_base_patch16_224')

    else:
        raise ValueError(f'{args.arch} is not conisdered in the current code.')

    return model


def get_model(arch, pretrained, args):

    backbone = get_backbone(arch, pretrained, args)

    '''
    if args.deploy == 'vanilla':
        model = ProtoNet(backbone)
    elif args.deploy == 'finetune':
        model = ProtoNet_Finetune(backbone, args.ada_steps, args.ada_lr, args.aug_prob, args.aug_types)
    elif args.deploy == 'finetune_autolr':
        model = ProtoNet_Auto_Finetune(backbone, args.ada_steps, args.aug_prob, args.aug_types)
    elif args.deploy == 'ada_tokens':
        model = ProtoNet_AdaTok(backbone, args.num_adapters,
                                args.ada_steps, args.ada_lr)
    elif args.deploy == 'ada_tokens_entmin':
        model = ProtoNet_AdaTok_EntMin(backbone, args.num_adapters,
                                       args.ada_steps, args.ada_lr)
    else:
        raise ValueError(f'deploy method {args.deploy} is not supported.')'''
    return backbone
