import torch.nn as nn
import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
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


class TrainAgent(BaseTrainer):
    def build_net(self, config):
        net = ViTToDeepCADLatent()
        return net

    def set_loss_function(self):
        self.criterion = nn.MSELoss().cuda()

    def set_optimizer(self, config):
        """set optimizer and lr scheduler used in training"""
        self.optimizer = torch.optim.Adam(self.net.parameters(), config.lr) # , betas=(config.beta1, 0.9))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, config.lr_step_size)

    def forward(self, data):
        images = data["images"].cuda()
        code = data["code"].cuda()
        pred_code = self.net(points)
        loss = self.criterion(pred_code, code)
        return pred_code, {"mse": loss}
