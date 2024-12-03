from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
import torch


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def hr_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(56),
        transforms.ToTensor()
    ])


def lr_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(14),
        transforms.ToTensor()
    ])


def hr_restore_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(56, interpolation=Image.BICUBIC),
        transforms.ToTensor()
    ])


def display_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(400),
        transforms.CenterCrop(400),
        transforms.ToTensor()
    ])


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dense_dataset_dir, sparse_detaset_dir, transform=False):
        super(TrainDatasetFromFolder, self).__init__()
        self.hr_rssi_values = read_hr_rssi_file(dense_dataset_dir, transform)
        self.lr_rssi_values = read_lr_rssi_file(sparse_detaset_dir, transform)
        if transform:
            self.hr_transform = hr_transform()
            self.lr_transform = lr_transform()
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            hr_image = self.hr_transform(self.hr_rssi_values[index])
            lr_image = self.lr_transform(self.lr_rssi_values[index])
        else:
            hr_image = torch.from_numpy(self.hr_rssi_values[index]).float()
            lr_image = torch.from_numpy(self.lr_rssi_values[index]).float()
        return lr_image, hr_image

    def __len__(self):
        return len(self.hr_rssi_values)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dense_dataset_dir, sparse_detaset_dir, transform=False):
        super(ValDatasetFromFolder, self).__init__()
        self.hr_rssi_values = read_hr_rssi_file(dense_dataset_dir, transform)
        self.lr_rssi_values = read_lr_rssi_file(sparse_detaset_dir, transform)
        if transform:
            self.hr_transform = hr_transform()
            self.lr_transform = lr_transform()
            self.hr_restore_transform = hr_restore_transform()
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            hr_image = self.hr_transform(self.hr_rssi_values[index])
            lr_image = self.lr_transform(self.lr_rssi_values[index])
            hr_restore_image = self.hr_restore_transform(self.lr_rssi_values[index])
        else:
            hr_image = torch.from_numpy(self.hr_rssi_values[index]).float()
            lr_image = torch.from_numpy(self.lr_rssi_values[index]).float()
            hr_restore_image = torch.from_numpy(self.lr_rssi_values[index]).float()
        return lr_image, hr_restore_image, hr_image

    def __len__(self):
        return len(self.hr_rssi_values)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
        self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = transforms.Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        return image_name, transforms.ToTensor()(lr_image), transforms.ToTensor()(hr_restore_img), transforms.ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)
    

class TrainDatasetFromFiles(Dataset):
    def __init__(self, dense_dataset_dir, sparse_detaset_dir, preprocess_dataset_dir, transform=False, in_size='lr'):
        super(TrainDatasetFromFiles, self).__init__()
        self.hr_rssi_values = read_hr_rssi_file(dense_dataset_dir, transform)
        self.lr_rssi_values = read_lr_rssi_file(sparse_detaset_dir, transform)
        if transform:
            self.hr_transform = hr_transform()
            if in_size == 'lr':
                self.preprocess_rssi_values = read_lr_rssi_file(preprocess_dataset_dir, transform)
                self.preprocess_transform = hr_restore_transform()
            else:
                self.preprocess_rssi_values = read_hr_rssi_file(preprocess_dataset_dir, transform)
                self.preprocess_transform = hr_transform()
            self.hr_restore_transform = hr_restore_transform()
        else:
            self.preprocess_rssi_values = read_hr_rssi_file(preprocess_dataset_dir, transform)
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            hr_image = self.hr_transform(self.hr_rssi_values[index])
            lr_image = self.preprocess_transform(self.preprocess_rssi_values[index])
            hr_restore_image = self.hr_restore_transform(self.lr_rssi_values[index])
        else:
            hr_image = torch.from_numpy(self.hr_rssi_values[index]).float()
            lr_image = torch.from_numpy(self.preprocess_rssi_values[index]).float()
            hr_restore_image = torch.from_numpy(self.lr_rssi_values[index]).float()
        return lr_image, hr_restore_image, hr_image

    def __len__(self):
        return len(self.hr_rssi_values)


def read_hr_rssi_file(rssi_file, transform):
    df = pd.read_csv(rssi_file)
    df_numpy = df.values[:, :56]
    n = df.shape[0] // 56
    if transform:
        rssi_values = np.zeros((n, 56, 56, 1), dtype=np.uint8)
        for indx in range(n):
            rssi_value = df_numpy[56*indx:56*(indx+1)]
            rssi_value = rssi_value.reshape(rssi_value.shape[0],rssi_value.shape[1],1)
            rssi_values[indx] = rssi_value
    else:
        rssi_values = np.zeros((n, 1, 56, 56))
        for indx in range(n):
            rssi_value = df_numpy[56*indx:56*(indx+1)]
            rssi_value = rssi_value.reshape(1,rssi_value.shape[0],rssi_value.shape[1])
            rssi_values[indx] = rssi_value
    return rssi_values


def read_lr_rssi_file(rssi_file, transform, x4=False):
    df = pd.read_csv(rssi_file)
    df_numpy = df.values[:, :14]
    n = df.shape[0] // 14
    if transform:
        rssi_values = np.zeros((n, 14, 14, 1), dtype=np.uint8)
        for indx in range(n):
            rssi_value = df_numpy[14*indx:14*(indx+1)]
            rssi_value = rssi_value.reshape(rssi_value.shape[0],rssi_value.shape[1],1)
            rssi_values[indx] = rssi_value
    else:
        rssi_values = np.zeros((n, 1, 14, 14))
        for indx in range(n):
            rssi_value = df_numpy[14*indx:14*(indx+1)]
            rssi_value = rssi_value.reshape(1,rssi_value.shape[0],rssi_value.shape[1])
            rssi_values[indx] = rssi_value
        if x4:
            rssi_values = np.repeat(np.repeat(rssi_values,4,axis=2),4,axis=3)
    return rssi_values



if __name__ == "__main__":
    pass