import torch.nn as nn
import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import TrainClock, cycle, ensure_dirs, ensure_dir
import argparse
import h5py
import shutil
import json
import random
from plyfile import PlyData, PlyElement
import sys
sys.path.append("..")
from cadlib.extrude import CADSequence
from cadlib.visualize import CADsolid2pc, create_CAD, BatchCADSolid2views
from OCC.Extend.DataExchange import write_stl_file
from base import BaseTrainer
from skimage import io, transform
from torchvision.transforms import Normalize
from cadlib.macro import EOS_IDX
from plyfile import PlyData, PlyElement
import random
from img2cad_model import TrainAgent,ViTToDeepCADLatent
from shape_image_code_dataset import ShapeImageCodesDataset

class Config(object):
    n_points = 2048
    batch_size = 128
    num_workers = 8
    nr_epochs = 50
    lr = 1e-4
    lr_step_size = 50
    suffixes=[f'{k:02d}' for k in range(6)]
    proj_dir='proj_log'
    exp_name='image2cad'

    save_frequency = 100
    val_frequency = 10

    def __init__(self, cont):
        self.data_root = os.path.join(self.proj_dir, self.exp_name, "results/all_zs_ckpt1000.h5")
        self.exp_dir = os.path.join(self.proj_dir, self.exp_name,'image2cad_main')
        self.data_dir='data/pc_cad'
        self.split_path='data/train_val_test_split.json'
        print(self.exp_dir)
        self.log_dir = os.path.join(self.exp_dir, 'log')
        self.model_dir = os.path.join(self.exp_dir, 'model')
        self.gpu_ids = 0

        if cont is not True and os.path.exists(self.exp_dir):
            response = input('Experiment log/model already exists, overwrite? (y/n) ')
            if response != 'y':
                exit()
            shutil.rmtree(self.exp_dir)
        ensure_dirs([self.log_dir, self.model_dir])
        with open('{}/config.txt'.format(self.exp_dir), 'w') as f:
            json.dump(self.__dict__, f, indent=2)




def get_dataloader(phase, config, shuffle=None):
    is_shuffle = phase == 'train' if shuffle is None else shuffle
    dataset = ShapeCodesDataset(phase, config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=is_shuffle, num_workers=config.num_workers)
    return dataloader

@torch.no_grad
def display_examples(trainer, ae_model, example_data):
    for item in example_data:
        img_to_disp=item['img_orig'].cpu().numpy()
        out_code,_=trainer.forward({k:(v.cuda().unsqueeze(0) if type(v) is torch.Tensor else v) for k,v in item.items()})
        outputs = ae_model.decode(out_code)
        out_vec = ae_model.logits2vec(outputs)[0]
        out_command = out_vec[:, 0]
        seq_len = out_command.tolist().index(EOS_IDX)
        data=out_vec[:seq_len]
        sequence=CADSequence.from_vector(data)
        sequence.normalize()
        shape = create_CAD(sequences)
        name = random.randint(100000, 999999)
        filename="tmp_out_{}.stl".format(name)
        write_stl_file(shape, filename)
        metadata=item['metadata']
        img_name=f"image_{name}.png"
        os.system(f"blender --background --python render_view.py -- {metadata['elevation']} {metadata['azimuth']} {metadata['distance']} {filename} {img_name} 1>nul")
        os.system(f'rm {filename}')
        img=Image.open(f"{img_name}.png")
        img=np.array(img)[:,::-1,:]
        os.system(f'rm {img_name}.png')
        large_img=np.stack([img_to_disp,img])
        plt.figure()
        plt.title(f"{item['id']} {item['suffix']}")
        plt.imshow(large_img)
        plt.show()

cont=False
ckpt='latest'

cfg = Config(cont)
print("data path:", cfg.data_root)
agent = TrainAgent(cfg)

np.random.seed(42)
val_example_set = ShapeCodesDataset('validation', cfg)
random_indices=np.random.randint(0,len(val_example_set),size=16)
example_data=[]
for ind in random_indices:
    example_data.append(val_example_set[ind])


if cont:
    agent.load_ckpt(ckpt)

tr_agent = TrainerAE(cfg)
tr_agent.load_ckpt(1000)
tr_agent.net.eval()

# create dataloader
train_loader = get_dataloader('train', cfg)
val_loader = get_dataloader('validation', cfg)
val_loader = cycle(val_loader)

# start training
clock = agent.clock

for e in range(clock.epoch, cfg.nr_epochs):
    # begin iteration
    pbar = tqdm(train_loader)
    for b, data in enumerate(pbar):
        # train step
        outputs, losses = agent.train_func(data)

        pbar.set_description("EPOCH[{}][{}]".format(e, b))
        pbar.set_postfix({k: v.item() for k, v in losses.items()})

        # validation step
        if clock.step % cfg.val_frequency == 0:
            data = next(val_loader)
            outputs, losses = agent.val_func(data)

        clock.tick()

    clock.tock()

    if clock.epoch % cfg.save_frequency == 0:
        agent.save_ckpt()

    agent.save_ckpt('latest')
    display_examples(agent,tr_agent,example_data)