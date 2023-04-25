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
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from ..model_utils.misc_utils import gather, all_gather


class ConVIRT_Loss(nn.Module):
    def __init__(self, batch_size, alpha=0.75, temperature=0.1, world_size=1):
        super(ConVIRT_Loss, self).__init__()
        self.batch_size = batch_size
        self.alpha = alpha
        self.temperature = temperature
        self.world_size = world_size
        self.similarity_f = nn.CosineSimilarity(dim=2)
        
    def NT_Xent(self, z_i, z_j):
        N = self.batch_size * self.world_size

        similarity_matrix = self.similarity_f(z_i.unsqueeze(1), z_j.unsqueeze(0)) / self.temperature

        nominator = torch.exp(torch.diag(similarity_matrix))
        denominator = torch.sum(torch.exp(similarity_matrix), dim=1)

        loss_partial = -torch.log(nominator / denominator)
        loss = torch.sum(loss_partial) / N

        return loss

    def forward(self, z_img, z_text):
        z_img, z_text = all_gather(z_img), all_gather(z_text)
        #z_img, z_text = z_img - z_img.mean(dim=0), z_text - z_text.mean(dim=0)
        loss_a, loss_b = self.NT_Xent(z_img, z_text), self.NT_Xent(z_text, z_img)
        return self.alpha * loss_a + (1 - self.alpha) * loss_b

