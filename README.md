# Cluster-Learngene

## compute mean attention distance
### deit_base
python -m torch.distributed.launch --nproc_per_node=5 --master_port=31385 --use_env train.py --model deit_base --compute_mean_attn_dist --weightinherit weight_assignment --data-path /home/user/datasets/ImageNet2012/Data/CLS-LOC --batchSize 20 --epochs 100 --max_update_distill 100 --warmup-epochs 5 --finetune ./checkpoint/deit_base_depth6_attn_patch+cls/base_depth_6.pth --output_dir ./checkpoint/deit_finetuning/ImageNet/base_depth6_weightassign_final_base_depth12/  --pin-mem

### deit_small
python -m torch.distributed.launch --nproc_per_node=5 --master_port=31385 --use_env train.py --model deit_small --compute_mean_attn_dist --weightinherit weight_assignment --data-path /home/user/datasets/ImageNet2012/Data/CLS-LOC --batchSize 20 --epochs 100 --max_update_distill 100 --warmup-epochs 5 --finetune ./checkpoint/deit_base_depth6_attn_patch+cls/base_depth_6.pth --output_dir ./checkpoint/deit_finetuning/ImageNet/base_depth6_weightassign_final_base_depth12/  --pin-mem

### deit_tiny 
python -m torch.distributed.launch --nproc_per_node=5 --master_port=31385 --use_env train.py --model deit_tiny --compute_mean_attn_dist --weightinherit weight_assignment --data-path /home/user/datasets/ImageNet2012/Data/CLS-LOC --batchSize 20 --epochs 100 --max_update_distill 100 --warmup-epochs 5   --pin-mem

## ImageNet上结果 继承FFN

### 参数assignment, 共享相同内存, savit_base, 参数和Flops从(85.647M, 16.863G)降低到(64.385M,12.680G)

CUDA_VISIBLE_DEVICES=6,7,8,9 python -m torch.distributed.launch --nproc_per_node=4 --master_port=31385 --use_env train_individual_imagenet1k_multidistill.py --model deit_base --expand_method weight_assignment --self_distillation_adaptation stitch --data-path /home/user/datasets/ImageNet2012/Data/CLS-LOC --batchSize 160 --epochs 50 --warmup-epochs 5 --finetune ./checkpoint/ --output_dir  ./checkpoint/deit_base_assignment/ --pin-mem

### 参数clone, 内存是独立空间,savit_base

CUDA_VISIBLE_DEVICES=6,7,8,9 python -m torch.distributed.launch --nproc_per_node=4 --master_port=31925 --use_env train_individual_imagenet1k_multidistill.py --model deit_base --expand_method weight_clone --self_distillation_adaptation stitch --data-path /home/user/datasets/ImageNet2012/Data/CLS-LOC --batchSize 160 --epochs 50 --warmup-epochs 5 --finetune ./checkpoint/ --output_dir  ./checkpoint/deit_base_assignment/ --pin-mem

### 参数assignment, 共享相同内存, savit_base, --width_ratio 1.5 参数和Flops从(85.647M, 16.863G)降低到

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=38275 --use_env train_individual_imagenet1k_multidistill.py --model deit_base --expand_method weight_assignment --width_ratio 1.5  --self_distillation_adaptation stitch --data-path /home/user/datasets/ImageNet2012/Data/CLS-LOC --batchSize 200 --epochs 50 --warmup-epochs 5 --finetune ./checkpoint/ --output_dir  ./checkpoint/deit_base_assignment_1.5/ --pin-mem

### 参数clone, 内存是独立空间,savit_base, --width_ratio 1.5

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=38275 --use_env train_individual_imagenet1k_multidistill.py --model deit_base --expand_method weight_clone --width_ratio 1.5  --self_distillation_adaptation stitch --data-path /home/user/datasets/ImageNet2012/Data/CLS-LOC --batchSize 200 --epochs 50 --warmup-epochs 5 --finetune ./checkpoint/ --output_dir  ./checkpoint/deit_base_clone_1.5/ --pin-mem

### 参数assignment, 共享相同内存, savit_base, --width_ratio 2 参数和Flops从(85.647M, 16.863G)降低到(60.85M,11.98G)

python -m torch.distributed.launch --nproc_per_node=4 --master_port=38275 --use_env train_individual_imagenet1k_multidistill.py --model deit_base --expand_method weight_assignment --width_ratio 2  --self_distillation_adaptation stitch --data-path /home/user/datasets/ImageNet2012/Data/CLS-LOC --batchSize 200 --epochs 50 --warmup-epochs 5 --finetune ./checkpoint/ --output_dir  ./checkpoint/deit_base_assignment_2/ --pin-mem

### 参数clone, 内存是独立空间,savit_base, --width_ratio 2
python -m torch.distributed.launch --nproc_per_node=4 --master_port=39025 --use_env train_individual_imagenet1k_multidistill.py --model deit_base --expand_method weight_clone --width_ratio 2  --self_distillation_adaptation stitch --data-path /home/user/datasets/ImageNet2012/Data/CLS-LOC --batchSize 200 --epochs 50 --warmup-epochs 5 --finetune ./checkpoint/ --output_dir  ./checkpoint/deit_base_clone_2/ --pin-mem


### 参数assignment, 共享相同内存, savit_small, 参数和Flops从(22M, 4.3G)降低到(16.267M,3.203G)
CUDA_VISIBLE_DEVICES=6,7,8,9 python -m torch.distributed.launch --nproc_per_node=4 --master_port=33415 --use_env train_individual_imagenet1k_multidistill.py --model deit_small --des_model savit_small --expand_method weight_assignment --self_distillation_adaptation stitch --data-path /home/user/datasets/ImageNet2012/Data/CLS-LOC --batchSize 320 --epochs 50 --warmup-epochs 5 --finetune ./checkpoint/ --output_dir  ./checkpoint/deit_small_assignment/ --pin-mem

### 参数assignment, 共享相同内存, savit_small, --width_ratio 1.5, 参数和Flops从(22M, 4.3G)降低到(16.267M,3.203G)
CUDA_VISIBLE_DEVICES=6,7,8,9 python -m torch.distributed.launch --nproc_per_node=4 --master_port=33415 --use_env train_individual_imagenet1k_multidistill.py --model deit_small --des_model savit_small --expand_method weight_assignment --width_ratio 1.5 --self_distillation_adaptation stitch --data-path /home/user/datasets/ImageNet2012/Data/CLS-LOC --batchSize 320 --epochs 50 --warmup-epochs 5 --finetune ./checkpoint/ --output_dir  ./checkpoint/deit_small_assignment_1.5/ --pin-mem

### 参数assignment, 共享相同内存, savit_tiny, 参数和Flops从(5.5M, 1.08G)降低到(4.152M,0.817G)
CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.launch --nproc_per_node=4 --master_port=39025 --use_env train_individual_imagenet1k_multidistill.py --model deit_tiny --des_model savit_tiny --expand_method weight_assignment --self_distillation_adaptation stitch --data-path /home/user/datasets/ImageNet2012/Data/CLS-LOC --batchSize 640 --epochs 50 --warmup-epochs 5 --finetune ./checkpoint/ --output_dir  ./checkpoint/deit_tiny_assignment/ --pin-mem


## ImageNet上结果 baselines

### auto-learngene deit_base
CUDA_VISIBLE_DEVICES=9 python -m torch.distributed.launch --nproc_per_node=1 --master_port=37317 --use_env train_individual_imagenet1k_multidistill.py --model deit_base --expand_method auto-learngene --self_distillation_adaptation stitch --data-path /home/user/datasets/ImageNet2012/Data/CLS-LOC --batchSize 200 --epochs 50 --warmup-epochs 5 --finetune ./checkpoint/ --output_dir  ./checkpoint/deit_base_auto_learngene/ --pin-mem

### auto-learngene deit_small
CUDA_VISIBLE_DEVICES=9 python -m torch.distributed.launch --nproc_per_node=1 --master_port=33297 --use_env train_individual_imagenet1k_multidistill.py --model deit_small --expand_method auto-learngene --self_distillation_adaptation stitch --data-path /home/user/datasets/ImageNet2012/Data/CLS-LOC --batchSize 400 --epochs 50 --warmup-epochs 5 --finetune ./checkpoint/ --output_dir  ./checkpoint/deit_small_auto_learngene/ --pin-mem

### auto-learngene deit_tiny
CUDA_VISIBLE_DEVICES=9 python -m torch.distributed.launch --nproc_per_node=1 --master_port=38341 --use_env train_individual_imagenet1k_multidistill.py --model deit_tiny --expand_method auto-learngene --self_distillation_adaptation stitch --data-path /home/user/datasets/ImageNet2012/Data/CLS-LOC --batchSize 800 --epochs 50 --warmup-epochs 5 --finetune ./checkpoint/ --output_dir  ./checkpoint/deit_tiny_auto_learngene/ --pin-mem

### heuristic-learngene deit_base
CUDA_VISIBLE_DEVICES=9 python -m torch.distributed.launch --nproc_per_node=1 --master_port=37827 --use_env train_individual_imagenet1k_multidistill.py --model deit_base --expand_method heuristic-learngene --self_distillation_adaptation stitch --data-path /home/user/datasets/ImageNet2012/Data/CLS-LOC --batchSize 200 --epochs 50 --warmup-epochs 5 --finetune ./checkpoint/ --output_dir  ./checkpoint/deit_base_heuristic_learngene/ --pin-mem


## ImageNet上结果 不继承FFN

### 参数assignment, 共享相同内存, savit_base, 参数和Flops从(85.647M, 16.863G)降低到(64.385M,12.680G)
CUDA_VISIBLE_DEVICES=5,6,7,8 python -m torch.distributed.launch --nproc_per_node=4 --master_port=36727 --use_env train_individual_imagenet1k_multidistill.py --model deit_base_w/o_ffn --expand_method weight_assignment --self_distillation_adaptation stitch --data-path /home/user/datasets/ImageNet2012/Data/CLS-LOC --batchSize 200 --epochs 50 --warmup-epochs 5 --finetune ./checkpoint/ --output_dir  ./checkpoint/deit_base_wo_ffn_assignment/ --pin-mem


## 下游任务

### iNat-2019, 参数assignment, 继承FFN, savit_small
python -m torch.distributed.launch --nproc_per_node=4 --master_port=43325 --use_env train_individual_imagenet1k_multidistill.py --model deit_small --des_model savit_small --expand_method weight_assignment --self_distillation_adaptation stitch --data-set INAT19 --data-path /home/user/datasets/iNat19 --batchSize 320 --epochs 500 --warmup-epochs 10 --finetune ./checkpoint/ --output_dir  ./checkpoint/small_finetuning/inat/ --pin-mem

### iNat-2019, 参数assignment, 不继承FFN, savit_small
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --master_port=43342 --use_env train_individual_imagenet1k_multidistill.py --model deit_small_w/o_ffn --des_model savit_small --expand_method weight_assignment --self_distillation_adaptation stitch --data-set INAT19 --data-path /home/user/datasets/iNat19 --batchSize 320 --epochs 500 --warmup-epochs 10 --finetune ./checkpoint/ --output_dir  ./checkpoint/small_finetuning/inat_wo_ffn/ --pin-mem
