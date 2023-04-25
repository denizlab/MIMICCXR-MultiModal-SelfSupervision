# ==============================================================================
# Copyright (C) 2020 Haoxu Huang, Samyak Rawlekar, Sumit Chopra, Cem M Deniz
#
# This file is part of MIMICCXR-Multi-SelfSupervision
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ==============================================================================

import os
import argparse
import random
import numpy as np
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.metrics import roc_auc_score
from barbar import Bar

from timm.models.vision_transformer import _create_vision_transformer
from timm.models.layers import PatchEmbed
from timm.models.vision_transformer import VisionTransformer, _cfg


def adjust_lr(optimizer):
    for param_group in optimizer.param_groups:
       param_group['lr'] /= 10
    return


def compute_AUCs(gt, pred, N_CLASSES):
    """Computes Area Under the Curve (AUC) from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        try:
            AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
        except ValueError:
            pass
            
    return AUROCs


class densenet_model(nn.Module):
    def __init__(self, size, features_dim, out_size, pretrained=True):
        super(densenet_model, self).__init__()
        
        if size == 121:
            self.backbone = torchvision.models.densenet121(pretrained=pretrained)

        self.feature_dim_in = self.backbone.classifier.weight.shape[1]
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim_in, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


# resnet model
class resnet_model(nn.Module):
    def __init__(self, size, features_dim, out_size, pretrained=True):
        super(resnet_model, self).__init__()
        
        if size==18:
            self.backbone = torchvision.models.resnet18(pretrained=pretrained)
        elif size==50:
            self.backbone = torchvision.models.resnet50(pretrained=pretrained)
        elif size==101:
            self.backbone = torchvision.models.resnet101(pretrained=pretrained)
        else:
            raise NotImplementedError(f"ResNet with size {size} is not implemented!")

        self.feature_dim_in = self.backbone.fc.weight.shape[1]
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim_in, out_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


# vit model
class vit_model(nn.Module):
    def __init__(self, size, features_dim, out_size, pretrained=True, freeze_pos_embed=False, **kwargs):
        super(vit_model, self).__init__()

        if freeze_pos_embed:
            pass
        else:
            if size=="base":
                model_kwargs = dict(
                    patch_size=16, embed_dim=768, depth=12, num_heads=12, num_classes=0, **kwargs
                )
                self.backbone = _create_vision_transformer("vit_base_patch16_224", pretrained=pretrained, **model_kwargs)
            else:
                pass
        
        self.classifier = nn.Sequential(
                nn.Linear(features_dim, out_size),
                nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

    
def train(args, model, optimizer, criterion, train_loader, val_loader, N_CLASSES, N_EPOCHS, CLASS_NAMES):
    pred = torch.FloatTensor()
    pred = pred.cuda()
    
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
    criterion = torch.nn.BCELoss()
    best_loss = 1e5
    for epoch in range(N_EPOCHS):
        print(f"Training Epoch {epoch} ...")
        train_losses = 0
        model.train()
        for i, (inp, target) in enumerate(Bar(train_loader)):
            optimizer.zero_grad()
            inp, target = inp.cuda(), target.cuda()
            output = model(inp)
            target = target.reshape(output.shape)
            train_loss = criterion(output, target)
            train_loss.backward()
            optimizer.step()
            train_losses += train_loss.item()
        
        train_losses /= len(train_loader)
        print("Training loss: {:.3f},".format(train_losses))

        print(f"Validating Epoch {epoch} ...")
        val_losses = 0
        model.eval()
        for i, (inp, target) in enumerate(Bar(val_loader)):
            inp, target = inp.cuda(), target.cuda()
            output = model(inp)
            target = target.reshape(output.shape)
            val_loss = criterion(output, target)
            val_losses += val_loss.item()
        
        val_losses /= len(val_loader)
        print("Validation loss: {:.3f},".format(val_losses))
        if best_loss > val_losses:
            best_loss = val_losses
            best_model = copy.deepcopy(model)
            torch.save({'state_dict': model.state_dict(), 
                        'best_loss': best_loss, 'optimizer' : optimizer.state_dict()}, 
                        'model_saved/' + args.save_suffix + '.pth.tar')
            print('Epoch ' + str(epoch + 1) + ' [save] loss = ' + str(best_loss))
        else:
            print('Epoch ' + str(epoch + 1) + ' [----] loss = ' + str(best_loss))
            adjust_lr(optimizer)
         
    return best_model
    
    
def evaluate(model, test_loader, N_CLASSES, CLASS_NAMES):
    pred = torch.FloatTensor()
    pred = pred.cuda()
    gt = torch.FloatTensor()
    gt = gt.cuda()
    model.eval()
    with torch.no_grad():
        for i, (inp, target) in enumerate(Bar(test_loader)):
            target = target.cuda()
            gt = torch.cat((gt, target), 0)
            bs, n_crops, c, h, w = inp.size()
            input_var = torch.autograd.Variable(inp.view(-1, c, h, w).cuda())
            output = model(input_var)
            output_mean = output.view(bs, n_crops, -1).mean(1)
            pred = torch.cat((pred, output_mean.data), 0)

    AUROCs = compute_AUCs(gt, pred, N_CLASSES)
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs[i]))
        
        