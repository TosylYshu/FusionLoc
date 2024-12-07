from skimage.io import imread
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from os.path import join
import numpy as np
import torch


def hr_restore_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(56, interpolation=Image.BICUBIC),
        transforms.ToTensor()
    ])


class CameraPoseDataset(Dataset):
    """
        A class representing a dataset of images and their poses
    """

    def __init__(self, dataset_path, rssi_file, labels_file, baseline_transform=None, wifi_transform=None, equalize_scenes=False, mode=''):
        """
        :param dataset_path: (str) the path to the dataset
        :param labels_file: (str) a file with images and their path labels
        :param data_transform: (Transform object) a torchvision transform object
        :return: an instance of the class
        """
        super(CameraPoseDataset, self).__init__()
        self.img_paths, self.poses, self.scenes, self.scenes_ids = read_labels_file(labels_file, dataset_path)
        self.rssi_values = read_hr_rssi_file(rssi_file)
        self.dataset_size = self.poses.shape[0]
        self.num_scenes = np.max(self.scenes_ids) + 1
        self.scenes_sample_indices = [np.where(np.array(self.scenes_ids) == i)[0] for i in range(self.num_scenes)]
        self.scene_prob_selection = [len(self.scenes_sample_indices[i])/len(self.scenes_ids)
                                     for i in range(self.num_scenes)]
        if self.num_scenes > 1 and equalize_scenes:
            max_samples_in_scene = np.max([len(indices) for indices in self.scenes_sample_indices])
            unbalanced_dataset_size = self.dataset_size
            self.dataset_size = max_samples_in_scene*self.num_scenes
            num_added_positions = self.dataset_size - unbalanced_dataset_size
            # gap of each scene to maximum / # of added fake positions
            self.scene_prob_selection = [ (max_samples_in_scene-len(self.scenes_sample_indices[i]))/num_added_positions for i in range(self.num_scenes) ]
        self.transform_baseline = baseline_transform
        self.transform_wifi = wifi_transform

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):

        if idx >= len(self.poses): # sample from an under-repsented scene
            sampled_scene_idx = np.random.choice(range(self.num_scenes), p=self.scene_prob_selection)
            idx = np.random.choice(self.scenes_sample_indices[sampled_scene_idx])

        img = imread(self.img_paths[idx])
        rssi_value = self.rssi_values[idx]
        pose = self.poses[idx]
        scene = self.scenes_ids[idx]
        if self.transform_baseline:
            img = self.transform_baseline(img)
        if self.transform_wifi:
            rssi_value = self.transform_wifi(rssi_value)
        else:
            rssi_value = torch.from_numpy(rssi_value).to(dtype=torch.float32)

        sample = {'img': img, 'rssi': rssi_value, 'pose': pose, 'scene': scene}
        return sample


def read_labels_file(labels_file, dataset_path):
    df = pd.read_csv(labels_file)
    imgs_paths = [join(dataset_path, path) for path in df['img_path'].values]
    scenes = df['scene'].values
    scene_unique_names = np.unique(scenes)
    scene_name_to_id = dict(zip(scene_unique_names, list(range(len(scene_unique_names)))))
    scenes_ids = [scene_name_to_id[s] for s in scenes]
    n = df.shape[0]
    poses = np.zeros((n, 7))
    poses[:, 0] = df['t1'].values
    poses[:, 1] = df['t2'].values
    poses[:, 2] = df['t3'].values
    poses[:, 3] = df['q1'].values
    poses[:, 4] = df['q2'].values
    poses[:, 5] = df['q3'].values
    poses[:, 6] = df['q4'].values
    return imgs_paths, poses, scenes, scenes_ids


def read_lr_rssi_file(rssi_file, use_nor=False, use_int=True):
    df = pd.read_csv(rssi_file)
    df_numpy = df.values[:, :14]
    n = df.shape[0] // 14
    if use_int:
        rssi_values = np.zeros((n, 14, 14, 1), dtype=np.uint8)
    else:
        rssi_values = np.zeros((n, 14, 14, 1))
    for i in range(n):
        rssi_value = df_numpy[14*i:14*(i+1)]
        rssi_value = rssi_value.reshape(rssi_value.shape[0],rssi_value.shape[1],1)
        rssi_values[i] = rssi_value
    if use_nor:
        max_rssi_value = np.max(rssi_values)
        min_rssi_value = np.min(rssi_values)
        rssi_values = (rssi_values - min_rssi_value) / (max_rssi_value - min_rssi_value) * 255
        rssi_values = rssi_values.astype(np.uint8)
    return rssi_values


def read_hr_rssi_file(rssi_file, use_nor=False, use_int=True):
    df = pd.read_csv(rssi_file)
    df_numpy = df.values[:, :56]
    n = df.shape[0] // 56
    if use_int:
        rssi_values = np.zeros((n, 56, 56, 1), dtype=np.uint8)
        for i in range(n):
            rssi_value = df_numpy[56*i:56*(i+1)]
            rssi_value = rssi_value.reshape(rssi_value.shape[0],rssi_value.shape[1],1)
            rssi_values[i] = rssi_value
    else:
        rssi_values = np.zeros((n, 1, 56, 56))
        for i in range(n):
            rssi_value = df_numpy[56*i:56*(i+1)]
            rssi_values[i][0] = rssi_value
    if use_nor:
        max_rssi_value = np.max(rssi_values)
        min_rssi_value = np.min(rssi_values)
        rssi_values = (rssi_values - min_rssi_value) / (max_rssi_value - min_rssi_value) * 255
        rssi_values = rssi_values.astype(np.uint8)
    if use_int:
        rssi_values = np.repeat(np.repeat(rssi_values,4,axis=1),4,axis=2)
    else:
        rssi_values = np.repeat(np.repeat(rssi_values,4,axis=2),4,axis=3)
    return rssi_values



if __name__ == "__main__":
    pass