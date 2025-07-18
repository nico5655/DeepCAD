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
from agent import BaseAgent
from skimage import io, transform
from torchvision.transforms import Normalize
from plyfile import PlyData, PlyElement


def write_ply(points, filename, text=False):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    with open(filename, mode='wb') as f:
        PlyData([el], text=text).write(f)


class Config(object):
    n_points = 2048
    batch_size = 128
    num_workers = 8
    nr_epochs = 200
    lr = 1e-4
    lr_step_size = 50
    # beta1 = 0.5
    grad_clip = None
    noise = 0.02

    save_frequency = 100
    val_frequency = 10

    def __init__(self, args):
        self.data_root = os.path.join(args.proj_dir, args.exp_name, "results/all_zs_ckpt{}.h5".format(args.ae_ckpt))
        self.exp_dir = os.path.join(args.proj_dir, args.exp_name, "pc2cad_tune_noise{}_{}_new".format(self.n_points, self.noise))
        print(self.exp_dir)
        self.log_dir = os.path.join(self.exp_dir, 'log')
        self.model_dir = os.path.join(self.exp_dir, 'model')
        self.gpu_ids = args.gpu_ids

        if (not args.test) and args.cont is not True and os.path.exists(self.exp_dir):
            response = input('Experiment log/model already exists, overwrite? (y/n) ')
            if response != 'y':
                exit()
            shutil.rmtree(self.exp_dir)
        ensure_dirs([self.log_dir, self.model_dir])
        if not args.test:
            os.system("cp pc2cad.py {}".format(self.exp_dir))
            with open('{}/config.txt'.format(self.exp_dir), 'w') as f:
                json.dump(self.__dict__, f, indent=2)


import torchvision.models as models
import torchvision.transforms as transforms
from timm import create_model

class ViTToDeepCADLatent(nn.Module):
    def __init__(self, latent_dim=256, pretrained=True):
        super().__init__()
        self.vit = create_model("vit_base_patch16_224", pretrained=pretrained)
        self.vit.reset_classifier(0)
        self.project = nn.Linear(self.vit.num_features, latent_dim)

    def forward(self, x):
        features = self.vit(x)           # [B, 768]
        latent = self.project(features)  # [B, 256]
        return latent


class TrainAgent(BaseAgent):
    def build_net(self, config):
        net = PointNet2()
        if len(config.gpu_ids) > 1:
            net = nn.DataParallel(net)
        # net = EncoderPointNet()
        return net

    def set_loss_function(self):
        self.criterion = nn.MSELoss().cuda()

    def set_optimizer(self, config):
        """set optimizer and lr scheduler used in training"""
        self.optimizer = torch.optim.Adam(self.net.parameters(), config.lr) # , betas=(config.beta1, 0.9))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, config.lr_step_size)

    def forward(self, data):
        points = data["points"].cuda()
        code = data["code"].cuda()

        pred_code = self.net(points)

        loss = self.criterion(pred_code, code)
        return pred_code, {"mse": loss}


def read_ply(path, with_normal=False):
    with open(path, 'rb') as f:
        plydata = PlyData.read(f)
        x = np.array(plydata['vertex']['x'])
        y = np.array(plydata['vertex']['y'])
        z = np.array(plydata['vertex']['z'])
        vertex = np.stack([x, y, z], axis=1)
        if with_normal:
            nx = np.array(plydata['vertex']['nx'])
            ny = np.array(plydata['vertex']['ny'])
            nz = np.array(plydata['vertex']['nz'])
            normals = np.stack([nx, ny, nz], axis=1)
    if with_normal:
        return np.concatenate([vertex, normals], axis=1)
    else:
        return vertex


class ShapeImageCodesDataset(Dataset):
    def __init__(self, phase, config):
        super(ShapeCodesDataset, self).__init__()
        self.n_points = config.n_points
        self.data_root = config.data_root
        self.pc_root = config.pc_root
        self.path = config.split_path
        self.suffixes=config.suffixes
        self.n_views=len(self.suffixes)
        with open(self.path, "r") as fp:
            self.all_data = json.load(fp)[phase]

        with h5py.File(self.data_root, 'r') as fp:
            self.zs = fp["{}_zs".format(phase)][:]

    def __getitem__(self, index):
        code_index=index//self.n_views
        view_suffix=self.suffixes[index%self.n_views]
        data_id = self.all_data[code_index]
        img_path = os.path.join(self.pc_root, f'{data_id}_{suffix}')
        if not os.path.exists(img_path):
            return self.__getitem__(index + 1)

        img = io.imread(img_path)
        img[np.where(img[:, :, 3] == 0)] = 255
        IMG_SIZE=224
        img = transform.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img[:, :, :3].astype(np.float32)

        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        img_normalized = self.normalize_img(img) if self.normalization else img

        pc = torch.tensor(pc, dtype=torch.float32)
        shape_code = torch.tensor(self.zs[index], dtype=torch.float32)
        return {"image": pc, "code": shape_code, "id": data_id, 'suffix':view_suffix, 'metadata':metadata}

    def __len__(self):
        return len(self.zs)*self.n_views

def get_dataloader(phase, config, shuffle=None):
    is_shuffle = phase == 'train' if shuffle is None else shuffle

    dataset = ShapeCodesDataset(phase, config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=is_shuffle, num_workers=config.num_workers)
    return dataloader


parser = argparse.ArgumentParser()
# parser.add_argument('--proj_dir', type=str, default="/mnt/disk6/wurundi/cad_gen",
#                    help="path to project folder where models and logs will be saved")
parser.add_argument('--proj_dir', type=str, default="/home/rundi/project_log/cad_gen",
                   help="path to project folder where models and logs will be saved")
parser.add_argument('--exp_name', type=str, required=True, help="name of this experiment")
parser.add_argument('--ae_ckpt', type=str, required=True, help="desired checkpoint to restore")
parser.add_argument('--continue', dest='cont', action='store_true', help="continue training from checkpoint")
parser.add_argument('--ckpt', type=str, default='latest', required=False, help="desired checkpoint to restore")
parser.add_argument('--test',action='store_true', help="test mode")
parser.add_argument('--n_samples', type=int, default=100, help="number of samples to generate when testing")
parser.add_argument('-g', '--gpu_ids', type=str, default="0",
                   help="gpu to use, e.g. 0  0,1,2. CPU not supported.")
args = parser.parse_args()

if args.gpu_ids is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)

cfg = Config(args)
print("data path:", cfg.data_root)
agent = TrainAgent(cfg)

if not args.test:
    # load from checkpoint if provided
    if args.cont:
        agent.load_ckpt(args.ckpt)
        # for g in agent.optimizer.param_groups:
        #     g['lr'] = 1e-5

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

        # if clock.epoch % 10 == 0:
        agent.save_ckpt('latest')
else:
    # load trained weights
    agent.load_ckpt(args.ckpt)

    test_loader = get_dataloader('test', cfg)

    # save_dir = os.path.join(cfg.exp_dir, "results/fake_z_ckpt{}_num{}_pc".format(args.ckpt, args.n_samples))
    save_dir = os.path.join(cfg.exp_dir, "results/pc2cad_ckpt{}_num{}".format(args.ckpt, args.n_samples))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    all_zs = []
    all_ids = []
    pbar = tqdm(test_loader)
    cnt = 0
    for i, data in enumerate(pbar):
        with torch.no_grad():
            pred_z, _ = agent.forward(data)
            pred_z = pred_z.detach().cpu().numpy()
            # print(pred_z.shape)
            all_zs.append(pred_z)

        all_ids.extend(data['id'])
        pts = data['points'].detach().cpu().numpy()
        # for j in range(pred_z.shape[0]):
        #     save_path = os.path.join(save_dir, "{}.ply".format(data['id'][j]))
        #     write_ply(pts[j], save_path)
        # for j in range(pred_z.shape[0]):
        #     save_path = os.path.join(save_dir, "{}.h5".format(data['id'][j]))
        #     with h5py.File(save_path, 'w') as fp:
        #         fp.create_dataset("zs", data=pred_z[j])

        cnt += pred_z.shape[0]
        if cnt > args.n_samples:
            break

    all_zs = np.concatenate(all_zs, axis=0)
    # save generated z
    save_path = os.path.join(cfg.exp_dir, "results/pc2cad_z_ckpt{}_num{}.h5".format(args.ckpt, args.n_samples))
    ensure_dir(os.path.dirname(save_path))
    with h5py.File(save_path, 'w') as fp:
        fp.create_dataset("zs", shape=all_zs.shape, data=all_zs)

    save_path = os.path.join(cfg.exp_dir, "results/pc2cad_z_ckpt{}_num{}_ids.json".format(args.ckpt, args.n_samples))
    with open(save_path, 'w') as fp:
        json.dump(all_ids, fp)
