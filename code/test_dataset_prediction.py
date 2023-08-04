import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import time
import os
from utils import *
from models import *
import wandb
from Dataset import *
from argparse import ArgumentParser
from datetime import datetime

def wandb_log_imgNmask(input_img, y_pred, title="Test set segmentation"):
    class_labels = {
        0: "Background",
        1: "class1",
        2: "class2"
    }
    input_img = np.array(input_img.cpu())  # 1 channel 3d image
    input_img /= input_img.max()

    wandb_images_lst = []
    for i in range(input_img.shape[2]//2 - 10, input_img.shape[2]//2 + 10):  # change shape index and i location to get different slices
        mask_img = wandb.Image(input_img[..., i], masks={
            "predictions": {
                "mask_data": y_pred[..., i],
                "class_labels": class_labels
            }
        })
        wandb_images_lst.append(mask_img)
    wandb.log({title: wandb_images_lst})

def predict_test_set(model, test_loader, device, save_dir, tt_aug=None, num_of_aug=40,
                     wandb_callback=False, wandb_args=None):
    if wandb_callback:
        assert wandb_args  # the args need to be not None
        wandb.login(key='588ccf0f1631d995e385af0b2675f0214d2fb9ff')  # team-dd
        wandb.init(project=wandb_args["project_name"],
                   name=wandb_args["test_name"],
                   config=wandb_args["config"])

    model.eval()
    with torch.no_grad():
        print(f"num of test samples: {len(test_loader)}")
        for batch_idx, (x_batch, x_names) in tqdm(enumerate(test_loader)):
            assert x_batch.shape[0] == 1   # the batch size must be 1

            # transform the image & predict the batch
            img = np.array(x_batch[0])
            x_batch, nonpadded_crop = test_reg_tform(img)
            x_batch = x_batch.float().to(device)
            output = model(x_batch)
            if tt_aug:
                for _ in range(num_of_aug):
                    x_batch = tt_aug_tform(img)
                    x_batch = x_batch.float().to(device)
                    output += model(x_batch)

            y_pred = torch.argmax(output.data, 1)  # the maximum of each channel is the classification
            y_pred = np.array(y_pred.cpu()).astype('uint8')

            img, y_pred = crop2regsize(x_batch[0, 0], y_pred[0], nonpadded_crop)  # one image only

            if wandb_callback:
                wandb_log_imgNmask(img, y_pred)

            # save the predicted image
            name = x_names[0]
            y_pred = nib.nifti1.Nifti1Image(y_pred, None)
            nib.save(y_pred, os.path.join(save_dir, name))

# --------------------  test data transforms -------------------
def padd_test_img(img):
    """ padds the image to the proper size for the network
        each axis has to be in the size of a multiplication of 16  and returns the cropping values"""
    target_shape = calculate_target_shape(img.shape)
    padded_img = torch.zeros(target_shape)
    p_y, p_x, p_z = target_shape
    i_y, i_x, i_z = img.shape

    x_diff = p_x - i_x
    y_diff = p_y - i_y
    z_diff = p_z - i_z

    start_x, stop_x = int(np.ceil(x_diff / 2)), p_x - int(np.floor(x_diff / 2))
    start_y, stop_y = int(np.ceil(y_diff / 2)), p_y - int(np.floor(y_diff / 2))
    start_z, stop_z = int(np.ceil(z_diff / 2)), p_z - int(np.floor(z_diff / 2))

    nonpadded_crop = [start_y, stop_y, start_x, stop_x, start_z, stop_z]
    # crop enc in the center
    try:
        padded_img[start_y: stop_y, start_x: stop_x, start_z: stop_z] = img
    except:
        raise ValueError("Error in test padding in Dataset pad func!!!!!!!!!!!!!!!")
    return padded_img, nonpadded_crop

def closest_ceil_16_divisor(number):
    if number % 16 == 0:
        return number
    else:
        return ((number // 16) + 1) * 16

def calculate_target_shape(img_shape):
    target_shape = []
    for axis in img_shape:
        target_shape.append(closest_ceil_16_divisor(axis))
    return target_shape

def test_reg_tform(img):
    img = img.copy()
    img = (img - img.mean()) / img.std()   # normalize img
    img = torch.tensor(img)
    img, nonpadded_crop = padd_test_img(img)
    return img[None][None], nonpadded_crop

def tt_aug_tform(img):
    img = img.copy()
    sigmas = [0, 0.001, 0.005, 0.01, 0.1]
    probs = [0.2, 0.2, 0.2, 0.2, 0.2]
    sigma = np.random.choice(sigmas, p=probs)

    img = (img - img.mean()) / img.std()   # normalize img
    # img, label = rand_rescale_3d(img, label, scale_prct=0.1, num_classes=3)
    img = random_noise(img, mode='gaussian', mean=0, var=sigma, clip=False)
    img = torch.tensor(img)
    img, nonpadded_crop = padd_test_img(img)
    return img[None][None]



def crop2regsize(img, pred, nonpadded_crop):
    y_start, y_stop, x_start, x_stop, z_start, z_stop = nonpadded_crop
    img = img[y_start: y_stop, x_start: x_stop, z_start: z_stop]
    pred = pred[y_start: y_stop, x_start: x_stop, z_start: z_stop]
    return img, pred

if __name__ == '__main__':
# --init_features 32 --dropout "0.3" --model "UNet" --data_type "Heart"
# --init_features 32 --dropout "0.3" --model "UNet" --data_type "Hippocampus"

    # Select GPU
    device = set_GPU(1)
    data_type = "Heart"  # Hippocampus, Heart
    test_name = data_type + " " + "tt aug"
    wandb_callback = True
    test_time_aug = True

    model_type = "UNet"  # UNet, UNet_Short
    init_features = 32
    dropout = 0.3
    experiments2load = {  # "experiment name": epoch number
        # "Hippocampus_f0_2": 55,
        # "Hippocampus_f1_2": 115,
        # "Hippocampus_f2_2": 60,
        # "Hippocampus_f3_2": 55,
        # "Hippocampus_f4_2": 90

        "Heart_f0_1": 720,
        "Heart_f1_2": 460,
        "Heart_f2_2": 580,
        "Heart_f3_1": 740,
        "Heart_f4_1": 510
    }

# --------------------------------------------------------------------------------------------------------
    experiment_names = [exp_name for exp_name in experiments2load.keys()]
    cp_epochs_to_load = [epoch for epoch in experiments2load.values()]
    config = {"experiment_names": experiment_names,
              "cp_epochs_to_load": cp_epochs_to_load,
              "model_type": model_type,
              "test_time_aug": test_time_aug}

    print(experiment_names)
    print(cp_epochs_to_load)

    # set variables
    cp_base_dir = "/media/rrtammyfs/Users/daniel/DeepVisionProject/checkpoints"
    save_dir = "/media/rrtammyfs/Users/daniel/DeepVisionProject/test_predictions"
    save_dir = os.path.join(save_dir, test_name)
    os.makedirs(save_dir, exist_ok=True)
    wandb_args = {"test_name": test_name, "config": config, "project_name": "DeepVisionProject-MSD"}
    transform = None
    batch_size = 1
    num_workers = 0
    if data_type == "Heart":
        num_classes = 2
    else:
        num_classes = 3

    # create the ensemble model
    models_lst = []
    for _ in range(len(experiment_names)):
        tmp_model = locals()[model_type](num_classes=num_classes, init_features=init_features, dropout=dropout)
        models_lst.append(tmp_model)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(models_lst[0].parameters(), lr=0.00015, betas=(0.9, 0.999))

    # load the models from their checkpoints
    model_name = models_lst[0].model_name
    for i in tqdm(range(len(experiment_names)),  desc=f'Loading models from checkpoints: '):
        cp_dir = os.path.join(cp_base_dir, experiment_names[i])
        checkpoint = load_checkpoint(os.path.join(cp_dir, f"{model_name}_e{str(cp_epochs_to_load[i])}"),
                                     models_lst[i], optimizer)
    model = EnsembleModel(models_lst, device=device)

    test_ds = DecDataset(data_type=data_type, tr_val_tst="test",
                         base_data_dir='data', transform=None, load2ram=False)
    test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False)

    predict_test_set(model, test_loader=test_loader, device=device, save_dir=save_dir,
                     tt_aug=test_time_aug, wandb_callback=wandb_callback, wandb_args=wandb_args)

    if wandb_callback:
        wandb.finish()
