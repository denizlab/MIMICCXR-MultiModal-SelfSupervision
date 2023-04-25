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


import torch
import torch.distributed as dist
import torch.nn.functional as F
from ..model_utils.misc_utils import gather 


def invariance_loss(z1, z2):
    return F.mse_loss(z1, z2)


def variance_loss(z1, z2):
    eps = 1e-4
    std_z1 = torch.sqrt(z1.var(dim=0) + eps)
    std_z2 = torch.sqrt(z2.var(dim=0) + eps)
    std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2)) / 2
    return std_loss


def covariance_loss(z1, z2):
    N, D = z1.size()

    cov_z1 = (z1.T @ z1) / (N - 1)
    cov_z2 = (z2.T @ z2) / (N - 1)

    diag = torch.eye(D, device=cov_z1.device)
    cov_loss = cov_z1[~diag.bool()].pow_(2).sum() / D + cov_z2[~diag.bool()].pow_(2).sum() / D
    return cov_loss


def vicreg_loss(z1, z2, invariance_lamb, variance_mu, covairance_v):
    z1, z2 = gather(z1), gather(z2)

    invar_loss = invariance_loss(z1, z2)

    # zero mean
    z1, z2 = z1 - z1.mean(dim=0), z2 - z2.mean(dim=0)
 
    var_loss = variance_loss(z1, z2)
    cov_loss = covariance_loss(z1, z2)

    loss = invariance_lamb * invar_loss + variance_mu * var_loss + covairance_v * cov_loss

    return loss

