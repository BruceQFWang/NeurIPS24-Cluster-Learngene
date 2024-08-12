# Cluster-Learngene


## Requirements

- python >= 3.7
- python libraries:
```bash
pip install -r requirements.txt
```

## Data preparation

We use the standard ImageNet dataset, you can download it from http://image-net.org/. Validation images are put in labeled sub-folders. The file structure should look like:
```bash
$ tree data
imagenet
├── train
│   ├── class1
│   │   ├── img1.jpeg
│   │   ├── img2.jpeg
│   │   └── ...
│   ├── class2
│   │   ├── img3.jpeg
│   │   └── ...
│   └── ...
└── val
    ├── class1
    │   ├── img4.jpeg
    │   ├── img5.jpeg
    │   └── ...
    ├── class2
    │   ├── img6.jpeg
    │   └── ...
    └── ...
```

## Compute mean attention distance
### deit_base
```
python -m torch.distributed.launch --nproc_per_node=5 --master_port=31385 --use_env train.py --model deit_base --compute_mean_attn_dist --weightinherit weight_assignment --data-path /home/user/datasets/ImageNet2012/Data/CLS-LOC --batchSize 20 --epochs 100 --warmup-epochs 5 --finetune ./checkpoint/deit_base_depth6_attn_patch+cls/base_depth_6.pth --output_dir ./checkpoint/deit_finetuning/ImageNet/base_depth6_weightassign_final_base_depth12/  --pin-mem
```

### deit_small
```
python -m torch.distributed.launch --nproc_per_node=5 --master_port=31385 --use_env train.py --model deit_small --compute_mean_attn_dist --weightinherit weight_assignment --data-path /home/user/datasets/ImageNet2012/Data/CLS-LOC --batchSize 20 --epochs 100 --warmup-epochs 5 --finetune ./checkpoint/deit_base_depth6_attn_patch+cls/base_depth_6.pth --output_dir ./checkpoint/deit_finetuning/ImageNet/base_depth6_weightassign_final_base_depth12/  --pin-mem
```

### deit_tiny 
```
python -m torch.distributed.launch --nproc_per_node=5 --master_port=31385 --use_env train.py --model deit_tiny --compute_mean_attn_dist --weightinherit weight_assignment --data-path /home/user/datasets/ImageNet2012/Data/CLS-LOC --batchSize 20 --epochs 100 --warmup-epochs 5   --pin-mem
```

## Results on ImageNet

### deit_base
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=31385 --use_env train.py --model deit_base --expand_method weight_assignment --self_distillation_adaptation stitch --data-path /home/user/datasets/ImageNet2012/Data/CLS-LOC --batchSize 160 --epochs 50 --warmup-epochs 5 --finetune ./checkpoint/ --output_dir  ./checkpoint/deit_base_assignment/ --pin-mem
```

### deit_small
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=33415 --use_env train.py --model deit_small --des_model savit_small --expand_method weight_assignment --self_distillation_adaptation stitch --data-path /home/user/datasets/ImageNet2012/Data/CLS-LOC --batchSize 320 --epochs 50 --warmup-epochs 5 --finetune ./checkpoint/ --output_dir  ./checkpoint/deit_small_assignment/ --pin-mem
```

### deit_tiny
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=39025 --use_env train.py --model deit_tiny --des_model savit_tiny --expand_method weight_assignment --self_distillation_adaptation stitch --data-path /home/user/datasets/ImageNet2012/Data/CLS-LOC --batchSize 640 --epochs 50 --warmup-epochs 5 --finetune ./checkpoint/ --output_dir  ./checkpoint/deit_tiny_assignment/ --pin-mem
```

## Downstream

### iNat-2019, deit_small
```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=43325 --use_env train.py --model deit_small --des_model savit_small --expand_method weight_assignment --self_distillation_adaptation stitch --data-set INAT19 --data-path /home/user/datasets/iNat19 --batchSize 320 --epochs 500 --warmup-epochs 10 --finetune ./checkpoint/ --output_dir  ./checkpoint/small_finetuning/inat/ --pin-mem
```
