import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import json
import nibabel as nib
import torchvision.transforms as transforms
from skimage.util import random_noise
import scipy.ndimage

def padd_imgNlabel(img, label, target_shape):
    """ padds the image and the label with -100 to the proper size for the network
        each axis has to be in the size of a multiplication of 16 """
    padded_img = torch.zeros(target_shape)
    padded_label = torch.zeros(target_shape) - 100  # the loss ignores -100 padding
    p_x, p_y, p_z = target_shape
    i_x, i_y, i_z = img.shape
    x_diff = p_x - i_x
    if x_diff < 0:
        x_diff = -x_diff
        start_x, stop_x = int(np.ceil(x_diff / 2)), i_x - int(np.floor(x_diff / 2))
        img = img[start_x: stop_x, :, :]
        label = label[start_x: stop_x, :, :]
        start_x, stop_x = 0, p_x
    else:
        start_x, stop_x = int(np.ceil(x_diff / 2)), p_x - int(np.floor(x_diff / 2))

    y_diff = p_y - i_y
    if y_diff < 0:
        y_diff = -y_diff
        start_y, stop_y = int(np.ceil(y_diff / 2)), i_y - int(np.floor(y_diff / 2))
        img = img[:, start_y: stop_y, :]
        label = label[:, start_y: stop_y, :]
        start_y, stop_y = 0, p_y
    else:
        start_y, stop_y = int(np.ceil(y_diff / 2)), p_y - int(np.floor(y_diff / 2))

    z_diff = p_z - i_z
    if z_diff < 0:
        z_diff = -z_diff
        start_z, stop_z = int(np.ceil(z_diff / 2)), i_z - int(np.floor(z_diff / 2))
        img = img[:, :, start_z: stop_z]
        label = label[:, :, start_z: stop_z]
        start_z, stop_z = 0, p_z
    else:
        start_z, stop_z = int(np.ceil(z_diff / 2)), p_z - int(np.floor(z_diff / 2))

    # crop enc in the center
    try:
        padded_img[start_x: stop_x, start_y: stop_y, start_z: stop_z] = img
        padded_label[start_x: stop_x, start_y: stop_y, start_z: stop_z] = label
    except:
        raise ValueError("Error in -100 padding in Dataset pad func!!!!!!!!!!!!!!!")
    return padded_img, padded_label



def rand_crop_heart(img, label, heart_crop_prob=0.8):
    """ heart_crop_prob chance that the crop will be exactly around the heart"""
    crop_around_the_heart = (torch.randint(0, 100, (1, )).item()/100 < heart_crop_prob)  # cropping close to the heart or not necesserly
    if crop_around_the_heart:
        start_y = torch.randint(30, 120, (1, )).item()
        start_x = torch.randint(50, 120, (1, )).item()
        # start_z = torch.randint(0, img.shape[2] - 89, (1, )).item()
    else:
        start_y = torch.randint(0, img.shape[0] - 144, (1, )).item()
        start_x = torch.randint(0, img.shape[1] - 144, (1, )).item()
        # start_z = torch.randint(0, img.shape[2] - 89, (1, )).item()
    img = img[start_y: start_y + 144, start_x: start_x + 144, :]
    # img = img[crop_y: 320 - crop_y, start_x: start_x + 120, start_z: start_z + 96]
    # label = label[crop_y: 320 - crop_y, start_x: start_x + 120, start_z: start_z + 96]
    label = label[start_y: start_y + 144, start_x: start_x + 144, :]

    # ys, xs, zs = np.where(label)
    # min_x, min_y, min_z = xs.min(), ys.min(), zs.min()
    # max_x, max_y, max_z = xs.max(), ys.max(), zs.max()

    return img, label


def rand_3d_flip(img, label):
    """ randomly flips the img and label (p = 0.5 for each axis) """
    flip_x, flip_y, flip_z = torch.randint(2, (3, ))  # 3 ints that are 0 or 1
    dims = []
    if flip_x:
        dims.append(0)
    if flip_y:
        dims.append(1)
    if flip_z:
        dims.append(2)
    return torch.flip(img, dims=dims), torch.flip(label, dims=dims)


def rand_3d_rotate(img, label, max_angle, num_classes, reshape=True):
    num_classes -= 1  # no background
    xyz = [(1, 2), (0, 2), (0, 1)]
    angles = torch.randint(-max_angle, max_angle, (3,))  # random angle for each axis
    axes = torch.randint(2, (3,))  # 1 if the axis is transformed

    # angles = np.array([45, 45, 45])  # random angle for each axis
    # axes = np.array([1, 1, 1])  # 1 if the axis is transformed

    rotations = angles * axes
    img_mask = (img != 0).astype('float')
    for idx, axes in enumerate(xyz):  # for each rotation axes
        if rotations[idx]:
            # label rotation
            class1 = (label == 1).astype('float')
            class1 = scipy.ndimage.rotate(class1, rotations[idx], mode='constant', cval=0, axes=axes, reshape=reshape)
            label_rot_tmp = np.zeros_like(class1)
            label_rot_tmp[class1 > 0.5] = 1
            for c in range(num_classes - 1):
                c += 2
                classi = (label == c).astype('float')
                classi = scipy.ndimage.rotate(classi, rotations[idx], mode='constant', cval=0, axes=axes, reshape=reshape)
                label_rot_tmp[classi > 0.5] = c
            label = label_rot_tmp

            img_mask = scipy.ndimage.rotate(img_mask, rotations[idx], mode='constant', cval=0, axes=axes, reshape=reshape)
            img = scipy.ndimage.rotate(img, rotations[idx], mode='constant', cval=0, axes=axes, reshape=reshape)
            img_mask[img_mask < 0.5] = 0
            img_mask[img_mask >= 0.5] = 1
            img[img_mask < 0.5] = 0
            label[img_mask < 0.5] = 0

    return img, label


def rand_rescale_3d(img, label, num_classes, scale_prct=0.1):
    """ random rescale the image with random scale (maximum 1+scale_prct and minimum 1-scale_prct)"""
    num_classes -= 1  # without the background
    scale = 1 + (torch.rand((1,)).item() * 2 * scale_prct) - scale_prct  # random the scale
    img = scipy.ndimage.zoom(img, (scale, scale, scale))

    class1 = (label == 1).astype('float')
    class1 = np.round(scipy.ndimage.zoom(class1, (scale, scale, scale)))
    label_scaled_tmp = np.zeros_like(class1)
    label_scaled_tmp[class1 > 0.5] = 1
    for c in range(num_classes - 1):
        c += 2
        classi = (label == c).astype('float')
        classi = np.round(scipy.ndimage.zoom(classi, (scale, scale, scale)))
        label_scaled_tmp[classi > 0.5] = c

    return img, label_scaled_tmp


def rand_rescale_3d_gpu(img, label, num_classes, scale_prct=0.1):
    """ random rescale the image with random scale (maximum 1+scale_prct and minimum 1-scale_prct)"""
    num_classes -= 1  # without the background
    scale = 1 + (torch.rand((1,)).item() * 2 * scale_prct) - scale_prct  # random the scale
    # img = scipy.ndimage.zoom(img, (scale, scale, scale))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # use GPU if runes on one
    dx = torch.linspace(-1, 1, int(scale*img.shape[1]))
    dy = torch.linspace(-1, 1, int(scale*img.shape[0]))
    dz = torch.linspace(-1, 1, int(scale*img.shape[2]))
    meshx, meshy, meshz = torch.meshgrid((dx, dy, dz))
    grid = torch.stack((meshx, meshy, meshz), 3)
    grid = grid.unsqueeze(0)  # add batch dim
    grid = grid.to(device)

    img = torch.Tensor(img[None][None]).to(device)
    img = torch.nn.functional.grid_sample(img, grid, align_corners=True)[0, 0]
    img = np.array(img.to("cpu"))

    class1 = (label == 1).astype('float')
    class1 = torch.Tensor(class1[None][None]).to(device)
    class1 = torch.nn.functional.grid_sample(class1, grid, align_corners=True)[0, 0]
    class1 = np.array(class1.to("cpu"))
    # class1 = np.round(scipy.ndimage.zoom(class1, (scale, scale, scale)))
    label_scaled_tmp = np.zeros_like(class1)
    label_scaled_tmp[class1 > 0.5] = 1
    for c in range(num_classes - 1):
        c += 2
        classi = (label == c).astype('float')
        classi = torch.Tensor(classi[None][None]).to(device)
        classi = torch.nn.functional.grid_sample(classi, grid, align_corners=True)[0, 0]
        classi = np.array(classi.to("cpu"))
        # classi = np.round(scipy.ndimage.zoom(classi, (scale, scale, scale)))
        label_scaled_tmp[classi > 0.5] = c

    return img, label_scaled_tmp


def heart_crop_and_norm(img, label, crop_y=72):
    img, label = img[crop_y: img.shape[0] - crop_y], label[crop_y: img.shape[0] - crop_y]
    img = (img - img.mean()) / img.std()   # normalize img
    img, label = rand_crop_heart(img, label, heart_crop_prob=0)
    return img, label

# -------------------------------------------------------------------------------

def hippo_all_augmentations(img, label, g_noise=(0, 0.05)):
    img, label = rand_rescale_3d(img, label, scale_prct=0.1, num_classes=3)
    img = random_noise(img, mode='gaussian', mean=g_noise[0], var=g_noise[1], clip=False)

    img, label = rand_3d_rotate(img, label, max_angle=20, num_classes=3, reshape=True)
    img, label = torch.tensor(img), torch.tensor(label)
    img, label = rand_3d_flip(img, label)
    img, label = padd_imgNlabel(img, label, target_shape=(96, 96, 96))
    img = (img - img.mean()) / img.std()   # normalize img
    return img, label


def hippo_basic_aug(img, label):
    img, label = torch.tensor(img), torch.tensor(label)
    img, label = rand_3d_flip(img, label)
    img, label = padd_imgNlabel(img, label, target_shape=(64, 64, 64))
    img = (img - img.mean()) / img.std()   # normalize img
    return img, label


def hippo_noise_aug(img, label, g_noise=(0, 0.1)):
    img = random_noise(img, mode='gaussian', mean=g_noise[0], var=g_noise[1], clip=False)
    return hippo_basic_aug(img, label)


def hippo_rescale_aug(img, label):
    img, label = rand_rescale_3d(img, label, scale_prct=0.01, num_classes=3)
    return hippo_basic_aug(img, label)


def hippo_rotate_aug(img, label):
    img, label = rand_3d_rotate(img, label, max_angle=10, num_classes=3, reshape=True)
    return hippo_basic_aug(img, label)


def hippo_reg_tform(img, label):
    img = (img - img.mean()) / img.std()   # normalize img
    img, label = torch.tensor(img), torch.tensor(label)
    img, label = padd_imgNlabel(img, label, target_shape=(64, 64, 64))
    return img, label


def heart_augmentations(img, label, g_noise=(0, 0.05)):
    img, label = rand_crop_heart(img, label)
    img, label = rand_rescale_3d(img, label, scale_prct=0.05, num_classes=3)
    img = random_noise(img, mode='gaussian', mean=g_noise[0], var=g_noise[1], clip=False)

    img, label = rand_3d_rotate(img, label, max_angle=20, num_classes=3, reshape=True)
    img, label = torch.tensor(img), torch.tensor(label)
    img, label = rand_3d_flip(img, label)
    img, label = padd_imgNlabel(img, label, target_shape=(96, 96, 144))
    img = (img - img.mean()) / img.std()   # normalize img

    # img[img != 0] = (img[img != 0] - img[img != 0].mean())/img[img != 0].std()

    # transforms.functional.adjust_brightnes    s()
    # transforms.functional.adjust_contrast()

    return img, label


def heart_reg_tform(img, label, heart_crop_prob=0., crop_y=72):
    img, label = img[crop_y: img.shape[0] - crop_y], label[crop_y: img.shape[0] - crop_y]
    img = (img - img.mean()) / img.std()   # normalize img
    img, label = rand_crop_heart(img, label, heart_crop_prob=heart_crop_prob)
    img, label = torch.tensor(img), torch.tensor(label)
    # img, label = padd_imgNlabel(img, label, target_shape=(96, 96, 96))
    img, label = padd_imgNlabel(img, label, target_shape=(144, 144, 144))
    return img, label


def heart_valid_tform(img, label, crop_y=72):
    img, label = img[crop_y: img.shape[0] - crop_y], label[crop_y: img.shape[0] - crop_y]
    img, label = torch.tensor(img), torch.tensor(label)
    img, label = padd_imgNlabel(img, label, target_shape=(320 - crop_y*2, 320, 144))
    img = (img - img.mean()) / img.std()   # normalize img
    return img, label


def heart_flip_aug(img, label):
    img, label = heart_crop_and_norm(img, label)
    img, label = torch.tensor(img), torch.tensor(label)
    img, label = rand_3d_flip(img, label)
    img, label = padd_imgNlabel(img, label, target_shape=(144, 144, 144))
    return img, label


def heart_noise_aug(img, label):
    sigmas = [0, 0.001, 0.005, 0.01]
    probs = [0.4, 0.2, 0.2, 0.2]
    sigma = np.random.choice(sigmas, p=probs)
    img, label = heart_crop_and_norm(img, label)
    img = random_noise(img, mode='gaussian', mean=0, var=sigma, clip=False)
    img, label = torch.tensor(img), torch.tensor(label)
    img, label = padd_imgNlabel(img, label, target_shape=(144, 144, 144))
    return img, label


def heart_rescale5prct_aug(img, label):
    img, label = heart_crop_and_norm(img, label)
    img, label = rand_rescale_3d(img, label, scale_prct=0.05, num_classes=2)
    img, label = torch.tensor(img), torch.tensor(label)
    img, label = padd_imgNlabel(img, label, target_shape=(144, 144, 144))
    return img, label


def heart_rescale10prct_aug(img, label):
    img, label = heart_crop_and_norm(img, label)
    img, label = rand_rescale_3d(img, label, scale_prct=0.1, num_classes=2)
    img, label = torch.tensor(img), torch.tensor(label)
    img, label = padd_imgNlabel(img, label, target_shape=(144, 144, 144))
    return img, label


def heart_best_aug(img, label):
    sigmas = [0, 0.001, 0.005, 0.01, 0.1]
    probs = [0.2, 0.2, 0.2, 0.2, 0.2]
    sigma = np.random.choice(sigmas, p=probs)
    # sigma = 1

    img, label = heart_crop_and_norm(img, label)
    img, label = rand_rescale_3d(img, label, scale_prct=0.05, num_classes=2)
    img = random_noise(img, mode='gaussian', mean=0, var=sigma, clip=False)
    img, label = torch.tensor(img), torch.tensor(label)
    img, label = padd_imgNlabel(img, label, target_shape=(144, 144, 144))
    return img, label


def hippocampus_best_aug(img, label):
    sigmas = [0, 0.001, 0.005, 0.01, 0.1]
    probs = [0.2, 0.2, 0.2, 0.2, 0.2]
    sigma = np.random.choice(sigmas, p=probs)
    # sigma = 1

    img = (img - img.mean()) / img.std()   # normalize img
    img, label = rand_rescale_3d(img, label, scale_prct=0.1, num_classes=3)
    img = random_noise(img, mode='gaussian', mean=0, var=sigma, clip=False)
    img, label = torch.tensor(img), torch.tensor(label)
    # img, label = rand_3d_flip(img, label)
    img, label = padd_imgNlabel(img, label, target_shape=(64, 64, 64))
    return img, label


if __name__ == "__main__":
 pass