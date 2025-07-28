import torch.nn as nn
import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize
from skimage import io, transform
import json
import h5py
import random

class ShapeImageCodesDataset(Dataset):
    def __init__(self, phase, config):
        super(ShapeImageCodesDataset, self).__init__()
        self.data_root = config.data_root
        self.pc_root = config.data_dir
        self.path = config.split_path
        self.suffixes=config.suffixes
        self.n_views=len(self.suffixes)
        self.normalization=True
        self.normalize_img = Normalize(mean= [0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        with open(self.path, "r") as fp:
            self.all_data = json.load(fp)[phase]

        with h5py.File(self.data_root, 'r') as fp:
            self.zs = fp["{}_zs".format(phase)][:]

    def __getitem__(self, index):
        code_index=index//self.n_views
        view_suffix=self.suffixes[index%self.n_views]
        data_id = self.all_data[code_index]
        img_path = os.path.join(self.pc_root, f'{data_id}_{view_suffix}.png')
        if not os.path.exists(img_path):
            return self.__getitem__(index + 1)
        
        metadata_path = os.path.join(self.pc_root, f'{data_id}_render_metadata.txt')
        metadata=''
        with open(metadata_path,'r') as f:
            metadata=f.read()
        metadata = metadata.split('\n')[index%self.n_views]
        elevation, azimuth, distance = [float(num) for num in metadata.split(' ')]
        metadata={'elevation': elevation, 'azimuth' : azimuth, 'distance' : distance}
        img = io.imread(img_path)
        img[np.where(img[:, :, 3] == 0)] = 1
        IMG_SIZE = 224
        img = transform.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img[:, :, :3].astype(np.float32)
        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        k = random.randint(0, 3)
        img = torch.rot90(img, k=k, dims=(1, 2))
        img_normalized = self.normalize_img(img) if self.normalization else img
        shape_code = torch.tensor(self.zs[code_index], dtype=torch.float32)
        return {"images": img_normalized, "images_orig":img, "code": shape_code, "id": data_id, 'suffix':view_suffix, 'metadata':metadata}

    def __len__(self):
        return len(self.zs)*self.n_views
