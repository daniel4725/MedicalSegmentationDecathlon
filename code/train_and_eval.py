import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import time
import os
import wandb
from utils import *
from sklearn.metrics import f1_score, jaccard_similarity_score
from tensorflow.keras.metrics import MeanIoU
import cv2


def train_model(model, optimizer, criterion, train_loader, valid_loader, epochs, device,
                num_classes, continue_from_checkpoint=None, save_cp_every=1000, cp_dir=None,
                wandb_callback=None):
    IoU = MeanIoU(num_classes=num_classes)

    if cp_dir is None:
        raise ValueError("please input cp_dir to train_model function !!")

    if continue_from_checkpoint is None:
        train_loss_lst = []
        valid_loss_lst = []
        start_epoch = 1
    else:
        _, epoch, loss, _, _ = continue_from_checkpoint
        start_epoch = epoch + 1
        train_loss_lst, valid_loss_lst = loss

    smooth_dice_target_train = []
    smooth_dice_target_valid = []
    smooth_number = 10

    show_timer = time.time()  # timer for progress printing
    train_start = time.time()  # timing the training
    train_size = len(train_loader)
    for epoch in range(start_epoch, start_epoch + epochs):
        epoch_time = time.time()  # timing each epoch
        model.train()

        # ------------ training data --------------------
        epoch_loss = 0
        num_samples = 0

        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.float().to(device), y_batch.to(device).long()

            # update parameters and calculate the loss
            optimizer.zero_grad()
            output = model(x_batch)
            loss, dice_loss, cross_entropy_loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            num_samples += len(output)
            epoch_loss += loss.item() * len(output)

            # evaluation metrics
            IoU.reset_states()
            dice_score, IOU_per_class, _ = eval_metrics(output, y_batch, IoU)

            # ---------- wandb --------------
            if wandb_callback:
                dice_dict = {f"train/Dice_class_{i}": score for i, score in enumerate(dice_score)}
                smooth_dice_target_train.append(np.mean(list(dice_dict.values())[1:]))
                if len(smooth_dice_target_train) > smooth_number:
                    smooth_dice_target_train.pop(0)
                dice_dict.update({"train/Dice_target_smooth": np.mean(smooth_dice_target_train)})
                dice_dict.update({"train/Dice_target": smooth_dice_target_train[-1]})
                dice_dict.update({"train/Dice_mean": dice_score.mean()})
                IOU_dict = {f"train/IoU_class_{i}": score for i, score in enumerate(IOU_per_class)}
                IOU_dict.update({"train/IoU_mean": IOU_per_class.mean()})
                metrics = {"train/loss": loss.item(),
                           "train/Dice_loss": 1 - dice_score.mean(),
                           "train/CrossEntropy_loss": cross_entropy_loss.item(),
                           "train/epoch": epoch - 1
                           }
                metrics.update(dice_dict)
                metrics.update(IOU_dict)
                wandb.log(metrics)
                if batch_idx == train_size - 1:
                    wandb_log_img(x_batch[0], y_batch[0], output[0], title="segmentation train")
            # --------------------------------

            if time.time() - show_timer > 1 or (batch_idx == train_size - 1):  # enters every 1 sec,last batch
                show_timer = time.time()
                print(f"\rEpoch {epoch}:  [{batch_idx + 1}/{train_size}]"
                      f'train loss: {epoch_loss / num_samples:.3f}'
                      f'\tDice: {dice_score.round(3)}, class IoU: {IOU_per_class.round(3)}', end='')

        # print in the end of the training epoch and append the losses
        print(f'\rEpoch {epoch}:\ttrain loss: {epoch_loss / num_samples:.3f}'
              f'\tDice: {dice_score.round(3)}, class IoU: {IOU_per_class.round(3)}')
        train_loss_lst.append(epoch_loss / num_samples)

        # ------------ validation data --------------------
        del x_batch, y_batch, output   # to save space before calling the evaluation func

        model.eval()  # evaluating the validation data
        valid_losses, valid_dice_score, valid_IOU_per_class = evaluate(model, valid_loader, criterion,
                                                                       "validation", device, IoU)
        valid_loss = valid_losses["loss"]
        valid_loss_lst.append(valid_loss)

        # ---------- wandb --------------
        if wandb_callback:
            val_dice_dict = {f"val/Dice_class_{i}": score for i, score in enumerate(valid_dice_score)}
            smooth_dice_target_valid.append(np.mean(list(val_dice_dict.values())[1:]))
            if len(smooth_dice_target_valid) > smooth_number:
                smooth_dice_target_valid.pop(0)
            val_dice_dict.update({"val/Dice_target_smooth": np.mean(smooth_dice_target_valid)})
            val_dice_dict.update({"val/Dice_target": smooth_dice_target_valid[-1]})
            val_dice_dict.update({"val/Dice_mean": valid_dice_score.mean()})
            val_IOU_dict = {f"val/IoU_class_{i}": score for i, score in enumerate(valid_IOU_per_class)}
            val_IOU_dict.update({"val/IoU_mean": valid_IOU_per_class.mean()})
            val_metrics = {"val/loss": valid_losses["loss"],
                           "val/Dice_loss": 1 - valid_dice_score.mean(),
                           "val/CrossEntropy_loss": valid_losses["cross_entropy"],
                           }
            val_metrics.update(val_dice_dict)
            val_metrics.update(val_IOU_dict)
            metrics["train/epoch"] = epoch
            metrics.update(val_metrics)
            wandb.log(metrics)
        # --------------------------------

        if epoch == (start_epoch + epochs - 2):  # at the end of the training, log full img
            evaluate(model, valid_loader, criterion, "validation", device, IoU, log_all_img=True)

        print(f'\r\t\t\tvalid loss: {valid_loss:.3f}  Dice: {valid_dice_score.round(3)}, '
              f'class IoU: {valid_IOU_per_class.round(3)}'
              f'\tepoch time: {show_time(time.time() - epoch_time)}\n')

        if (epoch % save_cp_every) == 0:
            path = os.path.join(cp_dir, model.model_name + "_e" + str(epoch))
            save_checkpoint(
                path,
                model,
                optimizer,
                epoch,
                loss=(train_loss_lst, valid_loss_lst)
            )

    print(f'Finished Training: {show_time(time.time() - train_start)}')
    return train_loss_lst, valid_loss_lst


def evaluate(model, data_loader, criterion, data_type, device, IoU, log_all_img=False):
    model.eval()
    epoch_loss = 0
    epoch_dice_loss = 0
    epoch_cross_entropy_loss = 0
    num_samples = 0
    IoU.reset_states()
    data_size = len(data_loader)
    epoch_dice = np.zeros(IoU.num_classes)
    epoch_IOU = np.zeros(IoU.num_classes)

    show_timer = time.time()  # timer for progress printing
    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(data_loader):
            x_batch, y_batch = x_batch.float().to(device), y_batch.to(device).long()

            output = model(x_batch)
            loss, dice_loss, cross_entropy_loss = criterion(output, y_batch)
            num_samples += len(output)
            epoch_loss += loss.item() * len(output)
            epoch_dice_loss += dice_loss.item() * len(output)
            epoch_cross_entropy_loss += cross_entropy_loss.item() * len(output)

            dice_score, IOU_per_class, mean_IOU = eval_metrics(output, y_batch, IoU)
            epoch_dice += dice_score * len(output)

            if (data_type == 'validation') and wandb.run and (batch_idx in [0, 1]):
                if log_all_img and (batch_idx == 0):
                    wandb_log_all_img(x_batch[0], y_batch[0], output[0])
                else:
                    wandb_log_img(x_batch[0], y_batch[0], output[0])

            if time.time() - show_timer > 1 or (batch_idx == data_size - 1):  # enters every 1 sec,last batch
                show_timer = time.time()
                print(f"\r{data_type}:  [{batch_idx + 1}/{data_size}]"
                      f' loss: {epoch_loss / num_samples:.3f}'
                      f'\tDice: {(epoch_dice / num_samples).round(3)}, class IoU: {IOU_per_class.round(3)}', end='')
    loss = epoch_loss / num_samples
    dice_loss = epoch_dice_loss / num_samples
    cross_entropy_loss = epoch_cross_entropy_loss / num_samples
    valid_losses = {"loss": loss, "dice": dice_loss, "cross_entropy": cross_entropy_loss}
    return valid_losses, epoch_dice / num_samples, IOU_per_class


def eval_metrics(output, y_batch, IoU, epsilon=1e-7):
    # evaluation metrics
    y_pred = torch.argmax(output.data, 1)  # the maximum of each channel is the classification
    y_pred, y_batch = np.array(y_pred.cpu()), np.array(y_batch.cpu())

    # ignore the -100 labels of the padding
    y_pred = y_pred[y_batch != -100]
    y_batch = y_batch[y_batch != -100]

    # Dice/F1 score
    dice_score = f1_score(y_batch, y_pred, average=None)

    # IOU
    IoU.update_state(y_batch, y_pred)
    intersection = np.diagonal(IoU.get_weights()[0])
    union = IoU.get_weights()[0].sum(axis=1) + IoU.get_weights()[0].sum(axis=0) - intersection
    IOU_per_class = intersection / (union + epsilon)  # to avoid dividing by 0
    mean_IOU = IOU_per_class.mean()
    return dice_score, IOU_per_class, mean_IOU


def wandb_log_img(input_img, gt_image, output, title="Segmentations validation"):
    class_labels = {
        0: "Background",
        1: "class1",
        2: "class2"
    }
    input_img = np.array(input_img[0].cpu())  # 1 channel 3d image
    input_img /= input_img.max()
    y_pred = np.array(torch.argmax(output.data, 0).cpu())  # the maximum of each channel is the classification
    gt_image = np.array(gt_image.cpu())
    ys, xs, zs = np.where(gt_image != -100)
    min_x, min_y, min_z = xs.min(), ys.min(), zs.min()
    max_x, max_y, max_z = xs.max(), ys.max(), zs.max()

    gt_image = gt_image[min_y: max_y + 1, min_x: max_x + 1, min_z: max_z + 1]
    input_img = input_img[min_y: max_y + 1, min_x: max_x + 1, min_z: max_z + 1]
    y_pred = y_pred[min_y: max_y + 1, min_x: max_x + 1, min_z: max_z + 1]

    wandb_images_lst = []
    for i in [input_img.shape[2] // 2 - 4,
              input_img.shape[2] // 2 + 4]:  # change shape index and i location to get different slices
        mask_img = wandb.Image(input_img[..., i], masks={
            "predictions": {
                "mask_data": y_pred[..., i],
                "class_labels": class_labels
            },
            "ground_truth": {
                "mask_data": gt_image[..., i],
                "class_labels": class_labels
            }
        })
        wandb_images_lst.append(mask_img)
    wandb.log({title: wandb_images_lst})


def wandb_log_all_img(input_img, gt_image, output, title="final full img validation"):
    class_labels = {
        0: "Background",
        1: "class1",
        2: "class2"
    }
    input_img = np.array(input_img[0].cpu())  # 1 channel 3d image
    input_img /= input_img.max()
    y_pred = np.array(torch.argmax(output.data, 0).cpu())  # the maximum of each channel is the classification
    gt_image = np.array(gt_image.cpu())
    ys, xs, zs = np.where(gt_image != -100)
    min_x, min_y, min_z = xs.min(), ys.min(), zs.min()
    max_x, max_y, max_z = xs.max(), ys.max(), zs.max()

    gt_image = gt_image[min_y: max_y + 1, min_x: max_x + 1, min_z: max_z + 1]
    input_img = input_img[min_y: max_y + 1, min_x: max_x + 1, min_z: max_z + 1]
    y_pred = y_pred[min_y: max_y + 1, min_x: max_x + 1, min_z: max_z + 1]

    wandb_images_lst = []
    for i in range(input_img.shape[2]//2 - 35, input_img.shape[2]//2 + 35):  # change shape index and i location to get different slices
        mask_img = wandb.Image(input_img[..., i], masks={
            "predictions": {
                "mask_data": y_pred[..., i],
                "class_labels": class_labels
            },
            "ground_truth": {
                "mask_data": gt_image[..., i],
                "class_labels": class_labels
            }
        })
        wandb_images_lst.append(mask_img)
    wandb.log({title: wandb_images_lst})


if __name__ == '__main__':
    pass
