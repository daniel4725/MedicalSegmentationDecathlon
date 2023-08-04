python3.6 train_models.py --experiment_name "Heart 1" --GPU 0 --data_type "Heart" --lr "0.0001" --dice_factor 1 --epochs 40 --wandb_callback --load2ram
python3.6 train_models.py --experiment_name "Hippocampus 1" --GPU 0 --data_type "Hippocampus" --lr "0.0001" --dice_factor 1 --epochs 40 --wandb_callback --load2ram
python3.6 train_models.py --experiment_name "Heart cross_entropy 1" --GPU 1 --data_type "Heart" --lr "0.0001" --dice_factor 0 --epochs 40 --wandb_callback --load2ram --runfromshell
python3.6 train_models.py --experiment_name "Hippocampus cross_entropy 1" --GPU 1 --data_type "Hippocampus" --lr "0.0001" --dice_factor 0 --epochs 40 --wandb_callback --load2ram --runfromshell


python3.6 train_models.py --experiment_name "Hippocampus aug" --GPU 0 --data_type "Hippocampus" --transform "hippo_augmentations" --class_weights 0.3526 11.7466 12.6686 --lr "0.0001" --dice_factor 1 --epochs 300 --wandb_callback --load2ram --runfromshell
python3.6 train_models.py --experiment_name "Heart aug" --GPU 1 --data_type "Heart" --transform "heart_augmentations"  --class_weights 0.5141 18.2512 --lr "0.0001" --dice_factor 1 --epochs 300 --wandb_callback --load2ram --runfromshell

python3.6 train_models.py --experiment_name "Hippocampus no aug" --GPU 2 --data_type "Hippocampus" --transform "hippo_reg_tform" --class_weights 0.3526 11.7466 12.6686 --lr "0.0001" --dice_factor 1 --epochs 300 --wandb_callback --load2ram --runfromshell
python3.6 train_models.py --experiment_name "Heart no aug" --GPU 3 --data_type "Heart" --transform "heart_reg_tform" --class_weights 0.5141 18.2512 --lr "0.0001" --dice_factor 1 --epochs 300 --wandb_callback --load2ram --runfromshell

python3.6 train_models.py --experiment_name "Hippocampus no aug" --GPU 1 --data_type "Hippocampus" --transform "hippo_reg_tform" --lr "0.0001" --dice_factor 0 --batch_size 6 --epochs 300 --wandb_callback --load2ram --runfromshell

python3.6 train_models.py --experiment_name "Hippocampus no aug" --GPU 2 --model "UNet_Short" --data_type "Hippocampus" --transform "hippo_reg_tform" --lr "0.0001" --dice_factor 0 --batch_size 6 --epochs 300 --wandb_callback --load2ram --runfromshell

# -------------------- 24.7  -------------------
python3.6 train_models.py --experiment_name "Hippocampus4test" --GPU 3 --save_cp_every 2 --init_features 32 --L2 "0.05" --dice_factor "0.75" --dropout "0.4" --model "UNet_Short" --data_type "Hippocampus" --transform "hippo_reg_tform" --lr "0.0001" --batch_size 8 --epochs 100 --wandb_callback --load2ram --runfromshell

# -------------------- 25.7  -------------------
python3.6 train_models.py --experiment_name "Heart_f0_1" --GPU 1 --data_fold 0 --save_cp_every 10 --init_features 32 --L2 "0.025" --dice_factor "1" --dropout "0.3" --class_weights 0.5 0.68 --model "UNet" --data_type "Heart" --transform "heart_best_aug" --lr "0.0001" --batch_size 4 --epochs 700 --wandb_callback --load2ram --runfromshell
python3.6 train_models.py --experiment_name "Heart_f1_1" --GPU 0 --data_fold 1 --save_cp_every 10 --init_features 32 --L2 "0.025" --dice_factor "1" --dropout "0.3" --class_weights 0.5 0.68 --model "UNet" --data_type "Heart" --transform "heart_best_aug" --lr "0.0001" --batch_size 4 --epochs 700 --wandb_callback --load2ram --runfromshell

# -------------------- 26.7  -------------------
python3.6 train_models.py --experiment_name "Heart_f2_1" --GPU 0 --data_fold 2 --save_cp_every 10 --init_features 32 --L2 "0.025" --dice_factor "1" --dropout "0.3" --class_weights 0.5 0.68 --model "UNet" --data_type "Heart" --transform "heart_best_aug" --lr "0.0001" --batch_size 4 --epochs 900 --wandb_callback --load2ram --runfromshell
python3.6 train_models.py --experiment_name "Heart_f3_1" --GPU 1 --data_fold 1 --save_cp_every 10 --init_features 32 --L2 "0.025" --dice_factor "1" --dropout "0.3" --class_weights 0.5 0.68 --model "UNet" --data_type "Heart" --transform "heart_best_aug" --lr "0.0001" --batch_size 4 --epochs 900 --wandb_callback --load2ram --runfromshell
python3.6 train_models.py --experiment_name "Heart_f4_1" --GPU 2 --data_fold 4 --save_cp_every 10 --init_features 32 --L2 "0.025" --dice_factor "1" --dropout "0.3" --class_weights 0.5 0.68 --model "UNet" --data_type "Heart" --transform "heart_best_aug" --lr "0.0001" --batch_size 4 --epochs 900 --wandb_callback --load2ram --runfromshell

# continue training
python3.6 train_models.py --experiment_name "Heart_f0_1" --GPU 1 --data_fold 0 --save_cp_every 10 --init_features 32 --L2 "0.025" --dice_factor "1" --dropout "0.3" --class_weights 0.5 0.68 --model "UNet" --data_type "Heart" --transform "heart_best_aug" --lr "0.0001" --batch_size 4 --epochs 200 --wandb_callback --load2ram --runfromshell --continue_from_checkpoint "True" --cp_epoch 700
python3.6 train_models.py --experiment_name "Heart_f3_1" --GPU 1 --data_fold 1 --save_cp_every 10 --init_features 32 --L2 "0.025" --dice_factor "1" --dropout "0.3" --class_weights 0.5 0.68 --model "UNet" --data_type "Heart" --transform "heart_best_aug" --lr "0.0001" --batch_size 4 --epochs 370 --wandb_callback --load2ram --runfromshell --continue_from_checkpoint "True" --cp_epoch 530

python3.6 train_models.py --experiment_name "Heart_f3_1" --GPU 0 --data_fold 3 --save_cp_every 10 --init_features 32 --L2 "0.025" --dice_factor "1" --dropout "0.3" --class_weights 0.5 0.68 --model "UNet" --data_type "Heart" --transform "heart_best_aug" --lr "0.0001" --batch_size 4 --epochs 900 --wandb_callback --load2ram --runfromshell
python3.6 train_models.py --experiment_name "Heart_f2_2" --GPU 2 --data_fold 2 --save_cp_every 10 --init_features 32 --L2 "0.025" --dice_factor "1" --dropout "0.3" --class_weights 0.5 0.68 --model "UNet" --data_type "Heart" --transform "heart_best_aug" --lr "0.0001" --batch_size 4 --epochs 900 --wandb_callback --load2ram --runfromshell
python3.6 train_models.py --experiment_name "Heart_f2_3" --GPU 2 --data_fold 2 --save_cp_every 10 --init_features 32 --L2 "0.025" --dice_factor "1" --dropout "0.3" --class_weights 0.5 0.68 --model "UNet" --data_type "Heart" --transform "heart_best_aug" --lr "0.0001" --batch_size 4 --epochs 900 --wandb_callback --load2ram --runfromshell

#  -----------------  27.7 ------------------
# hippocampus:
python3.8 train_models.py --experiment_name "Hippocampus_f0_1" --GPU 0 --data_fold 0 --save_cp_every 5 --init_features 32 --L2 "0.01" --dice_factor "1" --dropout "0.3" --class_weights 0.3526 11.7466 12.6686 --model "UNet" --data_type "Hippocampus" --transform "hippocampus_best_aug" --lr "0.0001" --batch_size 8 --epochs 300 --wandb_callback --load2ram --runfromshell
python3.8 train_models.py --experiment_name "Hippocampus_f1_1" --GPU 1 --data_fold 1 --save_cp_every 5 --init_features 32 --L2 "0.01" --dice_factor "1" --dropout "0.3" --class_weights 0.3526 11.7466 12.6686 --model "UNet" --data_type "Hippocampus" --transform "hippocampus_best_aug" --lr "0.0001" --batch_size 8 --epochs 300 --wandb_callback --load2ram --runfromshell
python3.8 train_models.py --experiment_name "Hippocampus_f2_1" --GPU 0 --data_fold 2 --save_cp_every 5 --init_features 32 --L2 "0.01" --dice_factor "1" --dropout "0.3" --class_weights 0.3526 11.7466 12.6686 --model "UNet" --data_type "Hippocampus" --transform "hippocampus_best_aug" --lr "0.0001" --batch_size 8 --epochs 300 --wandb_callback --load2ram --runfromshell
python3.8 train_models.py --experiment_name "Hippocampus_f3_1" --GPU 1 --data_fold 3 --save_cp_every 5 --init_features 32 --L2 "0.01" --dice_factor "1" --dropout "0.3" --class_weights 0.3526 11.7466 12.6686 --model "UNet" --data_type "Hippocampus" --transform "hippocampus_best_aug" --lr "0.0001" --batch_size 8 --epochs 300 --wandb_callback --load2ram --runfromshell

python3.8 train_models.py --experiment_name "Hippocampus_f0_1" --GPU 0 --data_fold 0 --save_cp_every 5 --init_features 32 --L2 "0.01" --dice_factor "1" --dropout "0.3" --class_weights 0.3526 11.7466 12.6686 --model "UNet" --data_type "Hippocampus" --transform "hippocampus_best_aug" --lr "0.0001" --batch_size 8 --epochs 300 --wandb_callback --load2ram --runfromshell
python3.8 train_models.py --experiment_name "Hippocampus_f1_1" --GPU 0 --data_fold 1 --save_cp_every 5 --init_features 32 --L2 "0.01" --dice_factor "1" --dropout "0.3" --class_weights 0.3526 11.7466 12.6686 --model "UNet" --data_type "Hippocampus" --transform "hippocampus_best_aug" --lr "0.0001" --batch_size 8 --epochs 300 --wandb_callback --load2ram --runfromshell
python3.8 train_models.py --experiment_name "Hippocampus_f2_1" --GPU 1 --data_fold 2 --save_cp_every 5 --init_features 32 --L2 "0.01" --dice_factor "1" --dropout "0.3" --class_weights 0.3526 11.7466 12.6686 --model "UNet" --data_type "Hippocampus" --transform "hippocampus_best_aug" --lr "0.0001" --batch_size 8 --epochs 300 --wandb_callback --load2ram --runfromshell
CUDA_VISIBLE_DEVICES=1 python3.8 train_models.py --experiment_name "Hippocampus_f3_1" --GPU 1 --data_fold 3 --save_cp_every 5 --init_features 32 --L2 "0.01" --dice_factor "1" --dropout "0.3" --class_weights 0.3526 11.7466 12.6686 --model "UNet" --data_type "Hippocampus" --transform "hippocampus_best_aug" --lr "0.0001" --batch_size 8 --epochs 300 --wandb_callback --load2ram --runfromshell
CUDA_VISIBLE_DEVICES=1 python3.8 train_models.py --experiment_name "Hippocampus_f4_1" --GPU 1 --data_fold 4 --save_cp_every 5 --init_features 32 --L2 "0.01" --dice_factor "1" --dropout "0.3" --class_weights 0.3526 11.7466 12.6686 --model "UNet" --data_type "Hippocampus" --transform "hippocampus_best_aug" --lr "0.0001" --batch_size 8 --epochs 300 --wandb_callback --load2ram --runfromshell

# rescale 10% and not 1%
python3.8 train_models.py --experiment_name "Hippocampus_f0_2" --GPU 0 --data_fold 0 --save_cp_every 5 --init_features 32 --L2 "0.01" --dice_factor "1" --dropout "0.3" --class_weights 0.3526 11.7466 12.6686 --model "UNet" --data_type "Hippocampus" --transform "hippocampus_best_aug" --lr "0.0001" --batch_size 8 --epochs 150 --wandb_callback --load2ram --runfromshell
python3.8 train_models.py --experiment_name "Hippocampus_f1_2" --GPU 0 --data_fold 1 --save_cp_every 5 --init_features 32 --L2 "0.01" --dice_factor "1" --dropout "0.3" --class_weights 0.3526 11.7466 12.6686 --model "UNet" --data_type "Hippocampus" --transform "hippocampus_best_aug" --lr "0.0001" --batch_size 8 --epochs 150 --wandb_callback --load2ram --runfromshell
python3.8 train_models.py --experiment_name "Hippocampus_f2_2" --GPU 0 --data_fold 2 --save_cp_every 5 --init_features 32 --L2 "0.01" --dice_factor "1" --dropout "0.3" --class_weights 0.3526 11.7466 12.6686 --model "UNet" --data_type "Hippocampus" --transform "hippocampus_best_aug" --lr "0.0001" --batch_size 8 --epochs 150 --wandb_callback --load2ram --runfromshell
CUDA_VISIBLE_DEVICES=1 python3.8 train_models.py --experiment_name "Hippocampus_f3_2" --GPU 1 --data_fold 3 --save_cp_every 5 --init_features 32 --L2 "0.01" --dice_factor "1" --dropout "0.3" --class_weights 0.3526 11.7466 12.6686 --model "UNet" --data_type "Hippocampus" --transform "hippocampus_best_aug" --lr "0.0001" --batch_size 8 --epochs 150 --wandb_callback --load2ram --runfromshell
CUDA_VISIBLE_DEVICES=1 python3.8 train_models.py --experiment_name "Hippocampus_f4_2" --GPU 1 --data_fold 4 --save_cp_every 5 --init_features 32 --L2 "0.01" --dice_factor "1" --dropout "0.3" --class_weights 0.3526 11.7466 12.6686 --model "UNet" --data_type "Hippocampus" --transform "hippocampus_best_aug" --lr "0.0001" --batch_size 8 --epochs 150 --wandb_callback --load2ram --runfromshell

# adding random flip to the 10% rescale
CUDA_VISIBLE_DEVICES=0 python3.8 train_models.py --experiment_name "Hippocampus_f0_3" --GPU 0 --data_fold 0 --save_cp_every 5 --init_features 32 --L2 "0.01" --dice_factor "1" --dropout "0.3" --class_weights 0.3526 11.7466 12.6686 --model "UNet" --data_type "Hippocampus" --transform "hippocampus_best_aug" --lr "0.0001" --batch_size 8 --epochs 150 --wandb_callback --load2ram --runfromshell
CUDA_VISIBLE_DEVICES=0 python3.8 train_models.py --experiment_name "Hippocampus_f1_3" --GPU 0 --data_fold 1 --save_cp_every 5 --init_features 32 --L2 "0.01" --dice_factor "1" --dropout "0.3" --class_weights 0.3526 11.7466 12.6686 --model "UNet" --data_type "Hippocampus" --transform "hippocampus_best_aug" --lr "0.0001" --batch_size 8 --epochs 150 --wandb_callback --load2ram --runfromshell
CUDA_VISIBLE_DEVICES=1 python3.8 train_models.py --experiment_name "Hippocampus_f2_3" --GPU 1 --data_fold 2 --save_cp_every 5 --init_features 32 --L2 "0.01" --dice_factor "1" --dropout "0.3" --class_weights 0.3526 11.7466 12.6686 --model "UNet" --data_type "Hippocampus" --transform "hippocampus_best_aug" --lr "0.0001" --batch_size 8 --epochs 150 --wandb_callback --load2ram --runfromshell
CUDA_VISIBLE_DEVICES=1 python3.8 train_models.py --experiment_name "Hippocampus_f3_3" --GPU 1 --data_fold 3 --save_cp_every 5 --init_features 32 --L2 "0.01" --dice_factor "1" --dropout "0.3" --class_weights 0.3526 11.7466 12.6686 --model "UNet" --data_type "Hippocampus" --transform "hippocampus_best_aug" --lr "0.0001" --batch_size 8 --epochs 150 --wandb_callback --load2ram --runfromshell
CUDA_VISIBLE_DEVICES=1 python3.8 train_models.py --experiment_name "Hippocampus_f4_3" --GPU 1 --data_fold 4 --save_cp_every 5 --init_features 32 --L2 "0.01" --dice_factor "1" --dropout "0.3" --class_weights 0.3526 11.7466 12.6686 --model "UNet" --data_type "Hippocampus" --transform "hippocampus_best_aug" --lr "0.0001" --batch_size 8 --epochs 150 --wandb_callback --load2ram --runfromshell


