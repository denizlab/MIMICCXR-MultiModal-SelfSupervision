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

import argparse
from ..methods.base import *
from ..methods.base import BASE_SLIP
from ..losses.vicreg_loss import vicreg_loss


class SLIP_VICREG(BASE_SLIP):

    def __init__(self, args):
        super().__init__(args)

        # Build Models
        self._build_model()


    def _build_model(self):
        super()._build_model()

        # vicreg projector
        self.vicreg_projector = nn.Sequential(
            nn.Linear(self.hparams.img_embedding_dim, self.hparams.vicreg_proj_hidden_dim),
            nn.BatchNorm1d(self.hparams.vicreg_proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hparams.vicreg_proj_hidden_dim, self.hparams.vicreg_proj_hidden_dim),
            nn.BatchNorm1d(self.hparams.vicreg_proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hparams.vicreg_proj_hidden_dim, self.hparams.vicreg_proj_output_dim),
        )


    def shared_forward(self, batch, batch_idx):
        img_feat1, img_feat2, img_feat_mm, mm_loss = super().shared_forward(batch, batch_idx)

        # vicreg
        z1, z2, z_mm = self.vicreg_projector(img_feat1), self.vicreg_projector(img_feat2), self.vicreg_projector(img_feat_mm)
        ssl_loss = vicreg_loss(z1, z2, invariance_lamb=self.hparams.invariance_lamb, 
                variance_mu=self.hparams.variance_mu, covairance_v=self.hparams.covariance_v) + \
                    vicreg_loss(z1, z_mm, invariance_lamb=self.hparams.invariance_lamb, 
                variance_mu=self.hparams.variance_mu, covairance_v=self.hparams.covariance_v) + \
                    vicreg_loss(z2, z_mm, invariance_lamb=self.hparams.invariance_lamb, 
                variance_mu=self.hparams.variance_mu, covairance_v=self.hparams.covariance_v)
        ssl_loss /= 3


        # slip final loss
        loss = mm_loss + self.hparams.ssl_scale * ssl_loss
        return {"loss": loss, "mm_loss": mm_loss, "ssl_loss": ssl_loss}


    @property
    def learnable_params(self):
        extra_learnable_params = [{"type": "projector", "params": self.vicreg_projector.parameters(), \
                                "lr": self.hparams.lr_img_backbone}]
        return super().learnable_params + extra_learnable_params


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("slip_vicreg")

        # vicreg hyper-parameters
        parser.add_argument("--invariance_lamb", type=float, default=25)
        parser.add_argument("--variance_mu", type=float, default=25)
        parser.add_argument("--covariance_v", type=float, default=1.)
        parser.add_argument("--vicreg_proj_output_dim", type=int, default=8192)
        parser.add_argument("--vicreg_proj_hidden_dim", type=int, default=8192)

        return parent_parser


