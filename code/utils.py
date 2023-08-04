import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import time
import os
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight



def set_GPU(gpu_num):
    GPU_ID = str(gpu_num)
    print('GPU USED: ' + GPU_ID)
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # use GPU if runes on one
    return device


def plotHist(training_output, model_name="", slice=(None, None)):
    """ plots the learning curves w.r.t the epochs """

    train_loss_lst, train_mae_lst, valid_loss_lst, valid_mae_lst = training_output

    x = [i+1 for i in range(len(train_loss_lst))]

    # plot loss
    fig, ax = plt.subplots()
    ax.set(xlabel='epoch number', ylabel='MSE',
           title=model_name+': loss over epochs')
    ax.plot(x[slice[0]:slice[1]], train_loss_lst[slice[0]:slice[1]], c='b')
    ax.plot(np.NaN, np.NaN, c='b', label='train')
    ax.plot(x[slice[0]:slice[1]], valid_loss_lst[slice[0]:slice[1]], c='r')
    ax.plot(np.NaN, np.NaN, c='r', label='validation')
    ax.legend(loc=1)
    ax.grid()

    # plot mean average error
    fig, ax = plt.subplots()
    ax.set(xlabel='epoch number', ylabel='MAE',
           title=model_name+': MAE over epochs')
    ax.plot(x[slice[0]:slice[1]], train_mae_lst[slice[0]:slice[1]], c='b')
    ax.plot(np.NaN, np.NaN, c='b', label='train')
    ax.plot(x[slice[0]:slice[1]], valid_mae_lst[slice[0]:slice[1]], c='r')
    ax.plot(np.NaN, np.NaN, c='r', label='validation')
    ax.legend(loc=1)
    ax.grid()

    plt.show()


def save_checkpoint(path, model, optimizer, epoch, description="No description", loss=(), accuracy=(), other=()):
    """ saves the current state of the model and optimizer and the training progress"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'description': description,
        'epoch': epoch,
        'loss': loss,
        'accuracy': accuracy,
        'other': other
    }, path)


def load_checkpoint(path, model, optimizer):
    """ loads the state of the model and optimizer and the training progress"""
    cp = torch.load(path)
    model.load_state_dict(cp['model_state_dict'])
    optimizer.load_state_dict(cp['optimizer_state_dict'])
    return cp['description'], cp['epoch'], cp['loss'], cp['accuracy'], cp['other']


def show_time(seconds):
    time = int(seconds)
    day = time // (24 * 3600)
    time = time % (24 * 3600)
    hour = time // 3600
    time %= 3600
    minutes = time // 60
    time %= 60
    seconds = time
    if day != 0:
        return "%dD %dH %dM %dS" % (day, hour, minutes, seconds)
    elif day == 0 and hour != 0:
        return "%dH %dM %dS" % (hour, minutes, seconds)
    elif day == 0 and hour == 0 and minutes != 0:
        return "%dM %dS" % (minutes, seconds)
    else:
        return "%dS" % (seconds)


def normalize_minmax(img):
    img = img - img.min()
    img = img / img.max()
    return img


def normalize_stdmean(img):
    img = img - img.mean()
    img = img / img.std()
    return img


def get_class_weight(train_loader, valid_loader, num_classes):
    classes = [i for i in range(num_classes)]
    labels = []
    for i in range(len(train_loader.dataset)):
        _, label = train_loader.dataset.__getitem__(i)
        label = label[label != -100]
        labels.append(label)
    for i in range(len(valid_loader.dataset)):
        _, label = valid_loader.dataset.__getitem__(i)
        label = label[label != -100]
        labels.append(label)
    all_labels = np.concatenate(labels)
    return torch.Tensor(compute_class_weight('balanced', classes, all_labels))

    # class_weights = compute_class_weight('balanced', classes,
    #                                      np.argmax(y_train, axis=-1).flatten())

def count_trainable_params(model):
    cnt = 0
    for p in model.parameters():
        if p.requires_grad:
            cnt += np.prod(p.shape)
    print(f"model's trainable parameters: {cnt}")

# def IOU(output, seg):
#     I = 0
#     U = 0
#     for label in torch.unique(seg):
#         a = output == label
#         b = seg == label
#         I += (a * b).sum()
#         U += a + b
#     return I/U


class DiceLoss(nn.Module):
    def __init__(self, weight=None):
        super(DiceLoss, self).__init__()
        self.weight = weight
        if self.weight is not None:
            self.weight = weight / weight.sum()
            self.classes = [i for i in range(len(weight))]

    def forward(self, output, target, smooth=1e-7):
        if self.weight is None:
            self.classes = torch.unique(target)
            self.classes = self.classes[self.classes != -100]
            self.weight = [1 / len(self.classes)] * len(self.classes)
        loss = 0
        for c, class_weight in zip(self.classes, self.weight):
            class_mask = target == c
            intersection = (class_mask * output[:, c])[target != -100].sum()
            denominator = (class_mask + output[:, c])[target != -100].sum()
            class_loss = 1 - (2 * intersection + smooth) / (denominator + smooth)
            loss += class_loss * class_weight

        return loss


class DiceAndCrossEntropyLossMix(nn.Module):
    def __init__(self, dice_factor=0.5, weight=None):
        super(DiceAndCrossEntropyLossMix, self).__init__()
        self.dice_loss = DiceLoss(weight=weight)
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=weight)
        self.dice_factor = dice_factor
        self.weight = weight

    def forward(self, output, target):
        dice = self.dice_loss(output, target)
        cross_entropy = self.cross_entropy_loss(output, target)
        return self.dice_factor * dice + (1 - self.dice_factor) * cross_entropy, dice, cross_entropy

if __name__ == '__main__':
    pass