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
from transform_and_augment import *

def make_video(img3d, seg3d):
    x, y, z = img3d.shape
    video = cv2.VideoWriter("video" + '.avi', cv2.VideoWriter_fourcc(*'DIVX'), 10, (x, y))
    for i in range(z):
        img_slice = img3d[:, :, i]
        img_slice = np.stack([img_slice, img_slice, img_slice], axis=2)
        seg_slice = seg3d[:, :, i]
        seg_slice = np.stack([seg_slice == 2, seg_slice * 0, seg_slice == 1], axis=2)

        alpha = 0.8
        img_slice = cv2.addWeighted(img_slice, alpha, seg_slice, 1 - alpha, 0.0)
        img_slice *= 255
        present = img_slice
        # present = cv2.resize(img_slice, (0, 0), fx=factor, fy=factor)
        cv2.imshow("s", np.array(present).astype('uint8'))
        video.write(np.array(present).astype('uint8'))
        cv2.waitKey(30)

    video.release()


def get_dataloaders(batch_size, data_type, base_data_dir='data', fold=0, num_workers=0,
                    transform=None, load2ram=True):
    """ creates the train test and validation data sets and creates their data loaders"""
    train_ds = DecDataset(data_type=data_type, tr_val_tst="train", fold=fold,
                          base_data_dir=base_data_dir, transform=transform, load2ram=load2ram)
    valid_ds = DecDataset(data_type=data_type, tr_val_tst="valid", fold=fold,
                          base_data_dir=base_data_dir, transform=transform, load2ram=load2ram)
    # test_ds = DecDataset(data_type=data_type, tr_val_tst="test", fold=fold,
    #                      base_data_dir=base_data_dir, transform=transform, load2ram=load2ram)

    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(dataset=valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, valid_loader  # , test_loader


class DecDataset(Dataset):
    def __init__(self, data_type, tr_val_tst, fold=0, base_data_dir="data",
                 transform=None, load2ram=False):
        self.tr_val_tst = tr_val_tst
        self.transform = transform
        self.data_type = data_type
        self.need_heart_crop = False

        assert fold in [0, 1, 2, 3, 4]

        if data_type == "Hippocampus":
            data_dir = os.path.join(base_data_dir, "Task04_Hippocampus")
            if transform is None:
                if tr_val_tst != "test":
                    self.transform = hippo_reg_tform
        elif data_type == "Heart":
            data_dir = os.path.join(base_data_dir, "Task02_Heart")
            if transform is None:
                if tr_val_tst == "train":
                    self.transform = heart_reg_tform
                    self.need_heart_crop = True
                elif tr_val_tst == "valid":
                    self.transform = heart_valid_tform
        else:
            raise ValueError("data_type not available!!")
        self.data_dir = data_dir

        ds_json_path = os.path.join(data_dir, "dataset.json")
        with open(ds_json_path) as json_file:
            ds_json = json.load(json_file)
        self.ds_json = ds_json

        np.random.seed(0)  # to repeat the same data split
        np.random.shuffle(ds_json["training"])

        # split the data to the relevant fold
        val_size = len(ds_json["training"])//5  # 20% validation
        val_data = np.copy(ds_json["training"][fold * val_size:(fold + 1) * val_size])
        train_data = list(np.copy(ds_json["training"]))
        for element in val_data:
            if element in train_data:
                train_data.remove(element)

        if tr_val_tst == 'valid':
            self.data = val_data
        elif tr_val_tst == "train":
            self.data = train_data
        elif tr_val_tst == "test":
            self.data = ds_json["test"]
        else:
            raise ValueError("tr_val_tst error!!")

        self.data_in_ram = False
        if load2ram:
            self.load_data2ram()
            self.data_in_ram = True

    def load_data2ram(self):
        # loads the data to the ram (make a list of all the loaded data)
        data_lst = []
        if self.tr_val_tst == "test":
            raise ValueError("Test Data dont need to be loaded to the ram")
        else:
            for sample in tqdm(self.data, desc=f'Loading {self.tr_val_tst} data to ram: '):
                img_path = os.path.join(self.data_dir, sample['image'][2:])
                label_path = os.path.join(self.data_dir, sample['label'][2:])
                img = nib.load(img_path).get_fdata()
                label = nib.load(label_path).get_fdata()
                data_lst.append((img, label))
        self.data = data_lst

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.tr_val_tst == 'test':  # if the dataset is test ds
            # we need to load the data from the data dir
            img_path_ending = self.data[index][2:]
            img_path = os.path.join(self.data_dir, img_path_ending)
            img = np.array(nib.load(img_path).get_fdata())
            img_name = img_path_ending.split('/')[1]

            return img, img_name

        else:   # if the dataset is validation or train
            if self.data_in_ram:   # the data is a list alredy
                img, label = self.data[index]
                img, label = img.copy(), label.copy()
            else:     # we need to load the data from the data dir
                img_path = os.path.join(self.data_dir, self.data[index]['image'][2:])
                label_path = os.path.join(self.data_dir, self.data[index]['label'][2:])
                img = nib.load(img_path).get_fdata()
                label = nib.load(label_path).get_fdata()
            if self.transform is not None:
                if self.need_heart_crop:
                    img, label = self.transform(img, label, 0)  #self.heart_crop_prob)
                else:
                    img, label = self.transform(img, label)

            return img[None, ...], label


def make_empty_results(base_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for img_name in os.listdir(base_dir):
        img = nib.load(os.path.join(base_dir, img_name))
        res_img = np.zeros(img.shape[:3], dtype='uint8')
        res_img = nib.nifti1.Nifti1Image(res_img, None)
        nib.save(res_img, os.path.join(dest_dir, img_name))


if __name__ == "__main__":
    from utils import set_GPU
    # Hippocampus
    # make_empty_results("/media/rrtammyfs/Users/daniel/transfer folder/Task05_Prostate",
    #                    "/media/rrtammyfs/Users/daniel/transfer folder/Task52")
    set_GPU(3)
    data_type = "Hippocampus"
    transform = hippocampus_best_aug   # hippo_augmentations  heart_reg_tform, heart_valid_tform
    ds = DecDataset(data_type=data_type, tr_val_tst='train', fold=0,
                    base_data_dir="data", transform=transform, load2ram=False)

    if data_type == "Heart":
        size_factor = 2
    else:
        size_factor = 5

    # ------------------ for test data only -------------------
    # for index in range(len(ds)):
    #     img, name, nonpadded_crop = ds.__getitem__(index)
    #     print(name, "shape: ", img.shape)
    #     img = img[0]
    #     img = img - img.min()
    #     img = img / img.max()
    #     for i in range(img.shape[2]):  # change shape index and i location to get different slices
    #         # make_video(img, seg)
    #         img_slice = img[:, :, i]
    #         img_slice = np.stack([img_slice, img_slice, img_slice], axis=2)
    #         img_slice *= 255
    #         # present = cv2.resize(img_slice, (0, 0), fx=10, fy=10)
    #         present = cv2.resize(img_slice, (0, 0), fx=size_factor, fy=size_factor)
    #         # cv2.imshow("s", np.array(present).astype('uint8'))
    #         # k = cv2.waitKey(30)
    # ----------------------------------------------------------------------------------------------

    # shapes = []
    # for i in tqdm(range(len(ds))):
    #     x, y = ds.__getitem__(i)
    #     shapes.append(x.shape)
    # shapes_set = set(shapes)

    save_slices = []
    for index in range(len(ds)):
        img, seg = ds.__getitem__(index)
        img = img[0]
        img = img - img.min()
        img = img / img.max()
        # angle = 45
        # axes = (1, 2)
        # class1 = (seg == 1).astype('float')
        # # class2 = (seg == 2).astype('float')
        # img_rot = scipy.ndimage.rotate(img, angle, mode='constant', cval=-100, axes=axes, reshape=True)
        # class1 = scipy.ndimage.rotate(class1, angle, mode='constant', cval=-100, axes=axes, reshape=True)
        # # class2 = scipy.ndimage.rotate(class2, angle, mode='constant', axes=axes, reshape=True)
        # seg_rot = np.zeros_like(class1)
        # seg_rot[class1 > 0.5] = 1
        # rand_3d_rotate(np.array(img), np.array(seg), 30)

        for i in range(img.shape[2]):  # change shape index and i location to get different slices
            # make_video(img, seg)
            img_slice = img[:, :, i]
            img_slice = np.stack([img_slice, img_slice, img_slice], axis=2)
            seg_slice = seg[:, :, i]
            seg_slice = np.stack([seg_slice == 2, seg_slice * 0, seg_slice == 1], axis=2)
            # seg_slice = np.stack([(seg_slice == 2) | (seg_slice == 1), seg_slice == -100, seg_slice * 0], axis=2)

            alpha = 0.8
            img_slice = cv2.addWeighted(img_slice, alpha, seg_slice, 1 - alpha, 0.0)
            img_slice *= 255
            # present = cv2.resize(img_slice, (0, 0), fx=10, fy=10)
            present = cv2.resize(img_slice, (0, 0), fx=size_factor, fy=size_factor)
            cv2.imshow("s", np.array(present).astype('uint8'))

            # img_slice = img_rot[:, :, i]
            # img_slice = np.stack([img_slice, img_slice, img_slice], axis=2)
            # seg_slice = seg_rot[:, :, i]
            # seg_slice = np.stack([seg_slice == 2, seg_slice * 0, seg_slice == 1], axis=2)
            #
            # alpha = 0.8
            # img_slice = cv2.addWeighted(img_slice, alpha, seg_slice, 1 - alpha, 0.0)
            # img_slice *= 255
            # # present = cv2.resize(img_slice, (0, 0), fx=10, fy=10)
            # present = cv2.resize(img_slice, (0, 0), fx=size_factor, fy=size_factor)
            # cv2.imshow("rotated", np.array(present).astype('uint8'))


            k = cv2.waitKey(30)
            if k == ord("s"):
                save_slices.append(img_slice)


    data_loaders = get_dataloaders(batch_size=1, data_type="Hippocampus", base_data_dir='data',
                                   fold=0, num_workers=0, transform=None, load2ram=True)
    train_loader, valid_loader, test_loader = data_loaders

    # data_loaders = get_dataloaders(batch_size=1, data_type="Heart", base_data_dir='data',
    #                                fold=0, num_workers=0, transform=None, load2ram=True)
    # train_loader, valid_loader, test_loader = data_loaders


    save_slices = []
    for x_batch, y_batch in train_loader:
        for img, seg in zip(x_batch, y_batch):
            img = img[0]
            img = img - img.min()
            img = img / img.max()

            for i in range(img.shape[0]):  # change shape index and i location to get different slices
                img_slice = img[i, :, :]
                img_slice = np.stack([img_slice, img_slice, img_slice], axis=2)
                seg_slice = seg[i, :, :]
                seg_slice = np.stack([seg_slice == 2, seg_slice == -100, seg_slice == 1], axis=2)

                alpha = 0.8
                img_slice = cv2.addWeighted(img_slice, alpha, seg_slice, 1-alpha, 0.0)
                img_slice *= 255
                present = cv2.resize(img_slice, (0, 0), fx=10, fy=10)
                # present = cv2.resize(img_slice, (0, 0), fx=2, fy=2)
                cv2.imshow("s", np.array(present).astype('uint8'))
                k = cv2.waitKey(0)
                if k == ord("s"):
                    save_slices.append(img_slice)


    print(2)

