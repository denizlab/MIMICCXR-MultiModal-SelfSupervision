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
import numpy as np
from ..model_utils.misc_utils import gather, all_gather, get_rank


# modified from https://github.com/facebookresearch/SLIP/blob/main/losses.py
class CLIP_Loss(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.labels = None
        self.last_local_batch_size = None

    def forward(self, image_embed, text_embed):
        local_batch_size = image_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * get_rank() + torch.arange(
                local_batch_size, device=image_embed.device
            )
            self.last_local_batch_size = local_batch_size

        # gather features from all GPUs
        image_embed_all, text_embed_all = \
            all_gather(image_embed), all_gather(text_embed)

        # normalized features
        #image_embed, text_embed = image_embed - image_embed.mean(dim=0), text_embed - text_embed.mean(dim=0)
        image_embed = F.normalize(image_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)

        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / self.temperature))
        logit_scale = logit_scale.exp()
        # cosine similarity as logits
        logits_per_image = logit_scale * image_embed @ text_embed_all.t() 
        logits_per_text = logit_scale * text_embed @ image_embed_all.t()

        loss = (F.cross_entropy(logits_per_image, self.labels) + \
            F.cross_entropy(logits_per_text, self.labels)) / 2

        return loss

