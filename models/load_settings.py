import torch
import os

def load_paper_settings(args):

    #WRN_path = os.path.join(args.data_path, 'WRN28-4_21.09.pt')
    #Pyramid_path = os.path.join(args.data_path, 'pyramid200_mixup_15.6.tar')

    if args.model == 'vit-dino':
        from models import get_model
        from models.vision_transformer import vit_small_6
        teacher = get_model(arch='dino_small_patch16')
        student = vit_small_6(patch_size=16, num_classes=5)
    
    elif args.model == 'resnet':
        from models.resnet_ilsvrc import resnet18
        from models.res12 import resnet12, ResNet_auto_learngene
        teacher = resnet18(num_classes=64)
        teacher.load_state_dict(torch.load('./checkpoint/resnet18/Monday_29_August_2022_15h_16m_37s/resnet18-180-regular.pth'))
        layer_1 = []
        layer_2 = [] 
        layer_3 = [] 
        layer_4 = []
        student = ResNet_auto_learngene(teacher.conv1, teacher.bn1, teacher.relu, teacher.maxpool,
                layer_1, layer_2, layer_3, layer_4, 5, method = 'scratch')

    elif args.model == 'vgg':
        from models.vgg_cifar import vgg16_bn_cifar64, vgg_compression_meta_learngene
        teacher = vgg16_bn_cifar64()
        teacher.load_state_dict(torch.load('./checkpoint/vgg16/Wednesday_07_September_2022_14h_46m_51s/vgg16-180-regular.pth'))
        layer_1 = []
        layer_2 = []
        layer_3 = []
        layer_4 = []
        student = vgg_compression_meta_learngene(layer_1, layer_2, layer_3, layer_4, num_classes = 5, method = 'scratch')

    else: 
        print('Undefined setting name !!!')
    '''
    elif args.paper_setting == 'a':
        teacher = WRN.WideResNet(depth=28, widen_factor=4, num_classes=100)
        state = torch.load(WRN_path, map_location={'cuda:0': 'cpu'})['model']
        teacher.load_state_dict(state)
        student = WRN.WideResNet(depth=16, widen_factor=4, num_classes=100)

    elif args.paper_setting == 'b':
        teacher = WRN.WideResNet(depth=28, widen_factor=4, num_classes=100)
        state = torch.load(WRN_path, map_location={'cuda:0': 'cpu'})['model']
        teacher.load_state_dict(state)
        student = WRN.WideResNet(depth=28, widen_factor=2, num_classes=100)

    elif args.paper_setting == 'c':
        teacher = WRN.WideResNet(depth=28, widen_factor=4, num_classes=100)
        state = torch.load(WRN_path, map_location={'cuda:0': 'cpu'})['model']
        teacher.load_state_dict(state)
        student = WRN.WideResNet(depth=16, widen_factor=2, num_classes=100)

    elif args.paper_setting == 'd':
        teacher = WRN.WideResNet(depth=28, widen_factor=4, num_classes=100)
        state = torch.load(WRN_path, map_location={'cuda:0': 'cpu'})['model']
        teacher.load_state_dict(state)
        student = RN.ResNet(depth=56, num_classes=100)

    elif args.paper_setting == 'e':
        teacher = PYN.PyramidNet(depth=200, alpha=240, num_classes=100, bottleneck=True)
        state = torch.load(Pyramid_path, map_location={'cuda:0': 'cpu'})['state_dict']
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in state.items():
            name = k[7:]  # remove 'module.' of dataparallel
            new_state[name] = v
        teacher.load_state_dict(new_state)
        student = WRN.WideResNet(depth=28, widen_factor=4, num_classes=100)

    elif args.paper_setting == 'f':
        teacher = PYN.PyramidNet(depth=200, alpha=240, num_classes=100, bottleneck=True)
        state = torch.load(Pyramid_path, map_location={'cuda:0': 'cpu'})['state_dict']
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in state.items():
            name = k[7:]  # remove 'module.' of dataparallel
            new_state[name] = v
        teacher.load_state_dict(new_state)
        student = PYN.PyramidNet(depth=110, alpha=84, num_classes=100, bottleneck=False)
    '''

    

    return teacher, student, args