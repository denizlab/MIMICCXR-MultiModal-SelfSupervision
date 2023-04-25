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
from ..methods.base import BASE_SSL
from ..losses.simclr_loss import SIMCLR_Loss


class SIMCLR(BASE_SSL):

    def __init__(self, args):
        super().__init__(args)

        # Build Models
        self._build_model()


    def _build_model(self):
        super()._build_model()

        # simclr objective        
        self.criterion = SIMCLR_Loss(self.hparams.batch_size, self.hparams.temperature_ssl, \
                        self.hparams.gpus * self.hparams.num_nodes)

        # simclr projector
        self.simclr_projector = nn.Sequential(
            nn.Linear(self.hparams.img_embedding_dim, self.hparams.simclr_proj_hidden_dim, bias=False),
            nn.BatchNorm1d(self.hparams.simclr_proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hparams.simclr_proj_hidden_dim, self.hparams.simclr_proj_output_dim, bias=False),
        )


    def shared_forward(self, batch, batch_idx):
        images_ssl1, images_ssl2 = batch

        # simclr 
        feat1, feat2 = self.img_backbone(images_ssl1), self.img_backbone(images_ssl2)
        z1, z2 = self.simclr_projector(feat1), self.simclr_projector(feat2)
        ssl_loss = self.criterion(z1, z2)

        return {"loss": ssl_loss}


    @property
    def learnable_params(self):
        extra_learnable_params = [{"type": "projector", "params": self.simclr_projector.parameters(), \
                                "lr": self.hparams.lr_img_backbone}]
        return super().learnable_params + extra_learnable_params


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("simclr")

        # simclr projector
        parser.add_argument("--simclr_proj_hidden_dim", type=int, default=512)
        parser.add_argument("--simclr_proj_output_dim", type=int, default=128)

        return parent_parser


