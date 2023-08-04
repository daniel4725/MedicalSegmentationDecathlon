import pandas as pd
import numpy as np
import torch.nn as nn
import time
import os
from utils import *
from train_and_eval import train_model
from models import *
from argparse import ArgumentParser, Namespace
import wandb
from Dataset import *
from transform_and_augment import *

# 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
def init_wandb(args, class_weights):
    wandb.init(
        project=args.project_name,
        name=args.experiment_name,
        config={
            "GPU": args.GPU,
            "data_type": args.data_type,
            "data_fold": args.data_fold,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "model": args.model,
            "init_features": args.init_features,
            "lr": args.lr,
            "L2_lambda": args.L2,
            "dropout": args.dropout,
            "dice_factor": args.dice_factor,
            "class_weights": class_weights,
            "checkpoint_dir": args.checkpoint_dir,
            "continue_from_checkpoint": args.continue_from_checkpoint,
            "cp_epoch": args.cp_epoch,
            "transform": args.transform
        })

# python3.6 train_models.py --wandb_callback True --experiment_name "baseline_L2-0.1 M 1" --GPU 2 --model "AgePredModel" --data_type M --L2 0.1 --epochs 50
if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--wandb_callback", action='store_true')
    parser.add_argument("--sweeping", action='store_true')
    parser.add_argument("--experiment_name", default="0 tst")
    parser.add_argument("--GPU", default=3)

    # model
    parser.add_argument("--model", default="UNet")
    parser.add_argument("--init_features", default=32)
    parser.add_argument("--L2", default=0)
    parser.add_argument("--dropout", default=0.0)
    parser.add_argument("--lr", default=0.0001)
    parser.add_argument("--dice_factor", default=1)
    parser.add_argument("--class_weights",  nargs="+", type=float, default=[0])  # [0] is auto weight calc
    #  [0.3526, 11.7466, 12.6686]  hippocampus

    # data
    parser.add_argument("--epochs", default=40)
    parser.add_argument("--data_type", choices=['Hippocampus', 'Heart'], default="Heart")
    parser.add_argument("--data_fold", type=int, choices=[0, 1, 2, 3, 4], default=0)
    parser.add_argument("--batch_size", default=6)
    parser.add_argument("--transform", default=None)
    parser.add_argument('--load2ram', action='store_true')

    # checkpoints save dir
    parser.add_argument("--checkpoint_dir", default="/media/rrtammyfs/Users/daniel/DeepVisionProject/checkpoints")
    parser.add_argument("--save_cp_every", default=1)

    # resume learning from checkpoint option
    parser.add_argument("--continue_from_checkpoint", default=None)
    parser.add_argument("--cp_epoch", default=None)

    # project name dont change!
    parser.add_argument("--project_name", default="DeepVisionProject-MSD")

    # running from here or from shell :
    parser.add_argument('--runfromshell', action='store_true')

    args = parser.parse_args()

    if not args.runfromshell:
        print("Not running from Shell")
        args.wandb_callback = False
        args.data_type = 'Hippocampus'
        args.class_weights = [0.3526, 11.7466, 12.6686]   # [0.3526, 11.7466, 12.6686]  [0.5169, 15.3180]
        args.model = 'UNet'  #'UNet_Short'
        args.init_features = 32
        args.batch_size = 8
        args.dropout = 0.2
        args.transform = "hippo_rotate_aug"
        num_workers = 0
    else:
        num_workers = 3
        print("Running from Shell")


    # ---------  wandb ----------
    # wandb.login(key='eb1e510a4bed996a9dac07bf3d3a2bda00cb113d')  # bio-lab
    sweeping = args.sweeping
    if sweeping:
        wandb.login(key='588ccf0f1631d995e385af0b2675f0214d2fb9ff')  # team-dd
        wandb.init()
        config = {key: wandb.config[key] for key in wandb.config.keys()}
        for key in config.keys():
            if config[key] == "None":
                config[key] = None
        args = Namespace(**config)
        wandb_callback = True
    elif args.wandb_callback:
        wandb.login(key='588ccf0f1631d995e385af0b2675f0214d2fb9ff')  # team-dd
        wandb.init(project=args.project_name,
                   name=args.experiment_name,
                   config=vars(args))
        wandb_callback = True
    else:
        wandb_callback = False
    # --------------------------

    # Select GPU
    if sweeping:
        device = "cuda"
    else:
        device = set_GPU(args.GPU)



    # create the checkpoints dir
    if sweeping:
        # experiment_name = os.path.join(args.sweep_name, wandb.run.name)
        experiment_name = args.sweep_name
    else:
        print("experiment name: ", args.experiment_name)
        print("epochs: ", args.epochs)
        print("data type: ", args.data_type)
        experiment_name = args.experiment_name
    cp_base_dir = args.checkpoint_dir
    cp_dir = os.path.join(cp_base_dir, experiment_name)
    os.makedirs(cp_dir, exist_ok=True)

    # set variables
    transform = args.transform
    if transform is not None:
        transform = locals()[transform]
    if args.data_type == "Heart":
        valid_transform = heart_valid_tform
        valid_batch_size = 2
    else:
        valid_transform = hippo_reg_tform
        valid_batch_size = 2

    fold = int(args.data_fold)
    batch_size = int(args.batch_size)
    init_features = int(args.init_features)
    epochs = int(args.epochs)
    save_cp_every = int(args.save_cp_every)
    lr = float(args.lr)
    L2_lambda = float(args.L2)
    dropout = float(args.dropout)
    dice_factor = float(args.dice_factor)

    #  create the relevant data loaders:
    # valid_batch_size = batch_size
    # if args.data_type == "Heart":
    #     valid_batch_size = 1
    train_ds = DecDataset(data_type=args.data_type, tr_val_tst="train", fold=fold,
                          base_data_dir='data', transform=transform, load2ram=args.load2ram)
    if args.data_type == "Heart" and sweeping:
        train_ds.heart_crop_prob = float(args.heart_crop_prob)
    valid_ds = DecDataset(data_type=args.data_type, tr_val_tst="valid", fold=fold,
                          base_data_dir='data', transform=valid_transform, load2ram=args.load2ram)
    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(dataset=valid_ds, batch_size=valid_batch_size, shuffle=True, num_workers=num_workers)

    # create the model
    if args.data_type == "Heart":
        num_classes = 2
    else:
        num_classes = 3

    if args.class_weights is None:
        class_weights = None
    elif len(args.class_weights) <= 1:  # compute automatically the class weights
        if args.data_type == "Heart":
            weight_calc_tform = heart_valid_tform   # valid tform takes the whole image
        else:
            weight_calc_tform = hippo_reg_tform
        print("Calculating class weights...")
        ds1 = DecDataset(data_type=args.data_type, tr_val_tst="train", fold=fold,
                         base_data_dir='data', transform=weight_calc_tform, load2ram=False)
        ds2 = DecDataset(data_type=args.data_type, tr_val_tst="valid", fold=fold,
                         base_data_dir='data', transform=weight_calc_tform, load2ram=False)
        loader1 = DataLoader(dataset=ds1, batch_size=batch_size, shuffle=True)
        loader2 = DataLoader(dataset=ds2, batch_size=valid_batch_size, shuffle=False)
        class_weights = get_class_weight(loader1, loader2, num_classes).to(device)
        print("Class weights are: ", class_weights)
    else:
        class_weights = torch.Tensor(args.class_weights).to(device)
        assert len(class_weights) == num_classes, "Number of classes is not the same as the number of weights!!!"

    model = locals()[args.model](num_classes=num_classes, init_features=init_features, dropout=dropout)
    count_trainable_params(model)
    model.to(device)
    criterion = DiceAndCrossEntropyLossMix(weight=class_weights, dice_factor=dice_factor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=L2_lambda)

    if args.continue_from_checkpoint is None:
        checkpoint = None
    else:
        checkpoint = load_checkpoint(os.path.join(cp_dir, "{}_e{}".format(model.model_name, str(args.cp_epoch))), model, optimizer)

    print("Starts the training...\n\n\n\n")

    train_model(model, optimizer, criterion,
                train_loader=train_loader,
                valid_loader=valid_loader,
                num_classes=num_classes,
                epochs=epochs, device=device, continue_from_checkpoint=checkpoint,
                save_cp_every=save_cp_every, cp_dir=cp_dir,
                wandb_callback=wandb_callback)

    if wandb_callback:
        wandb.finish()



