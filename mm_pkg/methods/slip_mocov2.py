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

from ..methods.base import *
from ..methods.base import BASE_SLIP
from ..losses.mocov2_loss import mocov2_loss
from ..model_utils.misc_utils import _batch_shuffle_ddp, _batch_unshuffle_ddp
from copy import deepcopy

class SLIP_MOCOV2(BASE_SLIP):

    def __init__(self, args):
        super().__init__(args)

        # Build Models
        self._build_model()


    def _build_model(self):
        super()._build_model()
        
        # ema model
        self.img_backbone_ema = deepcopy(self.img_backbone)
        for param in self.img_backbone_ema.parameters():
            param.requires_grad = False

        # mocov2 projectors
        self.mocov2_projector = nn.Sequential(
            nn.Linear(self.hparams.img_embedding_dim, self.hparams.img_embedding_dim),
            nn.ReLU(),
            nn.Linear(self.hparams.img_embedding_dim, self.hparams.mocov2_proj_output_dim),
        )
        self.mocov2_projector_ema = deepcopy(self.mocov2_projector) 
        for param in self.mocov2_projector_ema.parameters():
            param.requires_grad = False
       
        # create_queue 
        self.register_buffer("queue", torch.randn(3, self.hparams.mocov2_proj_output_dim, self.hparams.queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Adds new samples and removes old samples from the queue in a fifo manner.
        Args:
            keys (torch.Tensor): output features of the momentum backbone.
        """

        batch_size = keys.shape[1]
        ptr = int(self.queue_ptr)  # type: ignore
        assert self.hparams.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        keys = keys.permute(0, 2, 1)
        self.queue[:, :, ptr : ptr + batch_size] = keys
        ptr = (ptr + batch_size) % self.hparams.queue_size  # move pointer
        self.queue_ptr[0] = ptr  # type: ignore


    def shared_forward(self, batch, batch_idx):
        feats, images, mm_loss = super().shared_forward(batch, batch_idx)

        feat1, feat2, feat_mm = feats
        images_ssl1, images_ssl2, images_mm = images
        # mocov2
        # ema update
        ema(self.img_backbone, self.img_backbone_ema, self.hparams.ema_decay)
        ema(self.mocov2_projector, self.mocov2_projector_ema, self.hparams.ema_decay)

        # original encoder output
        q1, q2, q_mm = self.mocov2_projector(feat1), self.mocov2_projector(feat2), self.mocov2_projector(feat_mm)

        with torch.no_grad():
            # shuffle for making use of BN
            images1_k, idx_unshuffle1 = _batch_shuffle_ddp(images_ssl1)
            images2_k, idx_unshuffle2 = _batch_shuffle_ddp(images_ssl2)
            images_mm_k, idx_unshuffle3 = _batch_shuffle_ddp(images_mm)
            # ema encoder output
            feat1_ema, feat2_ema, feat_mm_ema = self.img_backbone_ema(images1_k), self.img_backbone_ema(images2_k), self.img_backbone_ema(images_mm_k)
            k1, k2, k_mm = self.mocov2_projector_ema(feat1_ema), self.mocov2_projector_ema(feat2_ema), self.mocov2_projector_ema(feat_mm_ema)
            # normalize
            k1, k2, k_mm = nn.functional.normalize(k1, dim=1), nn.functional.normalize(k2, dim=1), nn.functional.normalize(k_mm, dim=1)
            # undo shuffle
            k1, k2, k_mm = _batch_unshuffle_ddp(k1, idx_unshuffle1), _batch_unshuffle_ddp(k2, idx_unshuffle2), _batch_unshuffle_ddp(k_mm, idx_unshuffle3)

        # loss
        queue = self.queue.clone().detach()
        ssl_loss1 = (mocov2_loss(q1, k2, queue[1], self.hparams.temperature_ssl)
                + mocov2_loss(q2, k1, queue[0], self.hparams.temperature_ssl)) / 2
        ssl_loss2 = (mocov2_loss(q1, k_mm, queue[2], self.hparams.temperature_ssl)
                + mocov2_loss(q_mm, k1, queue[0], self.hparams.temperature_ssl)) / 2
        ssl_loss3 = (mocov2_loss(q2, k_mm, queue[2], self.hparams.temperature_ssl)
                + mocov2_loss(q_mm, k2, queue[1], self.hparams.temperature_ssl)) / 2
        ssl_loss = (ssl_loss1 + ssl_loss2 + ssl_loss3) / 3
        

        # update queue
        keys = torch.stack((gather(k1), gather(k2), gather(k_mm)))
        self._dequeue_and_enqueue(keys)

        # slip final loss
        loss = mm_loss + self.hparams.ssl_scale * ssl_loss
        return {"loss": loss, "mm_loss": mm_loss, "ssl_loss": ssl_loss}


    @property
    def learnable_params(self):
        extra_learnable_params = [{"type": "projector", "params": self.mocov2_projector.parameters(), \
                                "lr": self.hparams.lr_img_backbone}]
        return super().learnable_params + extra_learnable_params


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("slip_mocov2")

        # mocov2 hyper-parameters
        parser.add_argument("--ema_decay", type=float, default=0.999)
        # queue size
        parser.add_argument("--queue_size", type=int, default=65536)

        # mocov2 projector
        parser.add_argument("--mocov2_proj_hidden_dim", type=int, default=512)
        parser.add_argument("--mocov2_proj_output_dim", type=int, default=128)

        return parent_parser


