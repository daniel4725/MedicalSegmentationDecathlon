import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import os
import torch.nn.functional as F


class EnsembleModel(nn.Module):
    def __init__(self, models_lst, weights=None, device='cpu'):
        super(EnsembleModel, self).__init__()
        self.models_lst = nn.ModuleList(models_lst)
        self.weights = weights
        if self.weights is None:
            self.weights = torch.Tensor([1/len(models_lst)] * len(models_lst))
        self.device = device
        self.weights.to(device)

    def forward(self, x):
        self.models_lst[0].to(self.device)
        out = self.weights[0] * self.models_lst[0](x)
        self.models_lst[0].to('cpu')
        for model, w in zip(self.models_lst[1:], self.weights[1:]):
            model.to(self.device)
            out += model(x) * w
            model.to('cpu')
        return out


class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=3, init_features=32, dropout=0):
        super(UNet, self).__init__()
        self.model_name = "UNet"
        self.dropout = dropout
        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose3d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv3d(
            in_channels=features, out_channels=num_classes, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(F.dropout3d(self.pool3(enc3), p=self.dropout))

        bottleneck = self.bottleneck(F.dropout3d(self.pool4(enc4), p=self.dropout))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = F.dropout3d(dec4, p=self.dropout)
        # dec4 = self.crop_enc_and_cat(dec4, enc4)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = F.dropout3d(dec3, p=self.dropout)
        # dec3 = self.crop_enc_and_cat(dec3, enc3)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        # dec2 = self.crop_enc_and_cat(dec2, enc2)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        # dec1 = self.crop_enc_and_cat(dec1, enc1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    # def crop_enc_and_cat(self, dec, enc):
    #     # e_x, e_y, e_z = enc.shape[2:]
    #     # d_x, d_y, d_z = dec.shape[2:]
    #     # x_diff = e_x - d_x
    #     # y_diff = e_y - d_y
    #     # z_diff = e_z - d_z
    #     # start_x, stop_x = int(np.ceil(x_diff/2)), e_x - int(np.floor(x_diff/2))
    #     # start_y, stop_y = int(np.ceil(y_diff/2)), e_y - int(np.floor(y_diff/2))
    #     # start_z, stop_z = int(np.ceil(z_diff/2)), e_z - int(np.floor(z_diff/2))
    #     # # crop enc in the center
    #     # enc = enc[..., start_x: stop_x, start_y: stop_y, start_z: stop_z]
    #     return torch.cat((dec, enc), dim=1)

    @staticmethod
    def _block(in_channels, features, name):
        conv1 = nn.Conv3d(in_channels=in_channels, out_channels=features, kernel_size=3,
                          padding=1, bias=False)
        batch_norm1 = nn.BatchNorm3d(num_features=features)
        relu1 = nn.ReLU(inplace=True)
        conv2 = nn.Conv3d(in_channels=features, out_channels=features, kernel_size=3,
                          padding=1, bias=False)
        batch_norm2 = nn.BatchNorm3d(num_features=features)
        relu2 = nn.ReLU(inplace=True)
        return nn.Sequential(conv1, batch_norm1, relu1, conv2, batch_norm2, relu2)


class UNet_Short(nn.Module):
    def __init__(self, in_channels=1, num_classes=3, init_features=32, dropout=0):
        super(UNet_Short, self).__init__()
        self.model_name = "UNet"
        self.dropout = dropout
        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 4, features * 8, name="bottleneck")

        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv3d(
            in_channels=features, out_channels=num_classes, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(F.dropout3d(self.pool2(enc2), p=self.dropout))

        bottleneck = self.bottleneck(F.dropout3d(self.pool3(enc3), p=self.dropout))

        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = F.dropout3d(dec3, p=self.dropout)
        # dec3 = self.crop_enc_and_cat(dec3, enc3)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        # dec2 = self.crop_enc_and_cat(dec2, enc2)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        # dec1 = self.crop_enc_and_cat(dec1, enc1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    # def crop_enc_and_cat(self, dec, enc):
    #     # e_x, e_y, e_z = enc.shape[2:]
    #     # d_x, d_y, d_z = dec.shape[2:]
    #     # x_diff = e_x - d_x
    #     # y_diff = e_y - d_y
    #     # z_diff = e_z - d_z
    #     # start_x, stop_x = int(np.ceil(x_diff/2)), e_x - int(np.floor(x_diff/2))
    #     # start_y, stop_y = int(np.ceil(y_diff/2)), e_y - int(np.floor(y_diff/2))
    #     # start_z, stop_z = int(np.ceil(z_diff/2)), e_z - int(np.floor(z_diff/2))
    #     # # crop enc in the center
    #     # enc = enc[..., start_x: stop_x, start_y: stop_y, start_z: stop_z]
    #     return torch.cat((dec, enc), dim=1)

    @staticmethod
    def _block(in_channels, features, name):
        conv1 = nn.Conv3d(in_channels=in_channels, out_channels=features, kernel_size=3,
                          padding=1, bias=False)
        batch_norm1 = nn.BatchNorm3d(num_features=features)
        relu1 = nn.ReLU(inplace=True)
        conv2 = nn.Conv3d(in_channels=features, out_channels=features, kernel_size=3,
                          padding=1, bias=False)
        batch_norm2 = nn.BatchNorm3d(num_features=features)
        relu2 = nn.ReLU(inplace=True)
        return nn.Sequential(conv1, batch_norm1, relu1, conv2, batch_norm2, relu2)


if __name__ == "__main__":
    from Dataset import *
    from torch.utils.data import Dataset, DataLoader

    # Select GPU
    GPU_ID = '0'
    print('GPU USED: ' + GPU_ID)
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # use GPU if runes on one

    # unet = UNet()
    unet = UNet_Short()
    unet.to(device)

    # Hippocampus  Heart
    data_type = "Heart"
    #  create the relevant data loaders:
    train_ds = DecDataset(data_type=data_type, tr_val_tst="train", fold=0,
                          base_data_dir='data', transform=None, load2ram=False)
    valid_ds = DecDataset(data_type=data_type, tr_val_tst="valid", fold=0,
                          base_data_dir='data', load2ram=False)
    train_loader = DataLoader(dataset=train_ds, batch_size=6, shuffle=True)
    valid_loader = DataLoader(dataset=valid_ds, batch_size=6, shuffle=False)

    for x, y in tqdm(train_loader, "passing trainind data"):
        x = x.float().to(device)
        out = unet(x)