#!/usr/bin/env bash

gpus=0,1,2

data_name=LEVIR
net_G=ChangeFormerV6 #This is the best version
split=test
vis_root=/raid/parksonice/change_detection_3label/vis/
project_name=CD_ChangeFormerV6_LEVIR_test_final
checkpoints_root=/raid/parksonice/CD_ChangeFormerV6_LEVIR_b32_lr0.0002_adamw_train_val_150_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256/
checkpoint_name=best_ckpt.pt
img_size=256
embed_dim=256 #Make sure to change the embedding dim (best and default = 256)


CUDA_VISIBLE_DEVICES=0,1,2 python3 eval_cd.py --split ${split} --net_G ${net_G} --embed_dim ${embed_dim} --img_size ${img_size} --vis_root ${vis_root} --checkpoints_root ${checkpoints_root} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --n_class 4 --data_name ${data_name}


