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
from ..methods.base import BASE


class ConVIRT(BASE):

    def __init__(self, args):
        super().__init__(args)

        # Build models
        self._build_model()


    def _build_model(self):
        super()._build_model()
        self.text_backbone = bert_model(self.hparams.text_backbone, self.hparams.pool)        
        # freeze first six layers of text backbone accorindg to convirt paper
        freeze_layers = [i for i in range(0, 6)]
        for layer_idx in freeze_layers:
            for param in list(self.text_backbone.model.encoder.layer[layer_idx].parameters()):
                param.requires_grad = False
        self.text_projector = ProjectionHeadConVIRT(self.hparams.text_embedding_dim, \
                        self.hparams.projection_dim, self.hparams.dropout)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.text_backbone, use_fast=True)


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("convirt")
        return parent_parser


