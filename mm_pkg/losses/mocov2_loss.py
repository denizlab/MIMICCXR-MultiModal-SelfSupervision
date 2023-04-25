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


def mocov2_loss(query, key, queue, temperature):
    query, key = gather(query), gather(key)
    # normalize
    query, key = F.normalize(query, dim=-1, p=2), F.normalize(key, dim=-1, p=2)
    # positive logits: Nx1
    pos_logits = torch.einsum("nc,nc->n", [query, key]).unsqueeze(-1) 
    # negative logits: NxK
    neg_logits = torch.einsum("nc,ck->nk", [query, queue])
 
    # logits: Nx(1+K)
    logits = torch.cat([pos_logits, neg_logits], dim=1)
    # apply temperature
    logits /= temperature

    # labels
    labels = torch.zeros(logits.shape[0], device=query.device, dtype=torch.long)
    # loss
    loss = F.cross_entropy(logits, labels)
    return loss

