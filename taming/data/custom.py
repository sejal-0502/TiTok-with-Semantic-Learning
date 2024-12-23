import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex
import torch
from PIL import Image
from torchvision import transforms
import h5py
import random
from omegaconf import DictConfig, ListConfig


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example

class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file, random_crop=False, scale=False, crop_size=None):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, crop_size=crop_size, random_crop=random_crop, scale=scale)


class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file, random_crop=False, scale=False):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=random_crop, scale=scale) 

# Resizes the image (larger than the target size) and then crops the center with the target size
class RandomResizedCenterCrop(object):
    def __init__(self, size, scale=(0.5, 1.0), interpolation=Image.BILINEAR):
        self.scale = scale
        self.interpolation = interpolation
        self.size = size

    def __call__(self, img):
        width, height = img.size
        area = height * width
        aspect_ratio = width / height
        target_area = random.uniform(*self.scale) * area

        new_width = int(round((target_area * aspect_ratio) ** 0.5))
        new_height = int(round((target_area / aspect_ratio) ** 0.5))

        img = img.resize((new_width, new_height), self.interpolation)

        if isinstance(self.size, ListConfig):
            self.size = tuple(self.size)

        if isinstance(self.size, tuple):
            h, w = self.size
        else:
            h = w = self.size

         # Ensure the crop dimensions are valid
        if new_width < w or new_height < h:
            raise ValueError(
                f"Crop size {self.size} is larger than resized image size {(new_height, new_width)}"
            )

        x1 = (new_width - w) // 2
        y1 = (new_height - h) // 2

        return img.crop((x1, y1, x1 + w, y1 + h))

class MultiHDF5Dataset(Dataset):
    def __init__(self, size, hdf5_paths_file, aug='resize_center', scale_min=0.15, scale_max=0.5):
        self.size = size
        with open(hdf5_paths_file, 'r') as f:
            self.hdf5_paths = f.read().splitlines()

        self.files = [h5py.File(path, 'r') for path in self.hdf5_paths]
        self.lengths = []
        self.file_keys = []
        for file in self.files:
            keys = list(file.keys())
            self.file_keys.append(keys)
            self.lengths.append({key: len(file[key]) for key in keys})

        self.total_length = sum(sum(lengths.values()) for lengths in self.lengths)
        print(f'Total length: {self.total_length}')
        if aug == 'resize_center':
            self.transform = transforms.Compose([transforms.Resize(self.size),
                                         transforms.CenterCrop(self.size),
                                         transforms.ToTensor(),
                                         ])
        elif aug == 'random_resize_center':
            self.transform = transforms.Compose([
                                        RandomResizedCenterCrop(size=self.size, scale=(scale_min, scale_max)),
                                        transforms.ToTensor(),
                                        ])

    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        file_index = random.randint(0, len(self.files) - 1)
        h5_file = h5py.File(self.hdf5_paths[file_index], 'r', rdcc_nbytes=1024*1024*1024*100, rdcc_nslots=1000)
        key_index = random.randint(0, len(self.file_keys[file_index]) - 1)
        img_index = random.randint(0, self.lengths[file_index][self.file_keys[file_index][key_index]] - 1)
        key = self.file_keys[file_index][key_index]
        if 'meta_data' in key:
            key = key.replace('_meta_data', '')
        data = h5_file[key][img_index]
        image = Image.fromarray(data)
        image = self.transform(image)
        image = image * 2 - 1
        return image

    def close(self):
        for file in self.files:
            file.close()

