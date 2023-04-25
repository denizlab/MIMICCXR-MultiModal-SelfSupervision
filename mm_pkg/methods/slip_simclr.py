from ..methods.base import *
from ..methods.base import BASE_SLIP
from ..losses.simclr_loss import SIMCLR_Loss


class SLIP_SIMCLR(BASE_SLIP):

    def __init__(self, args):
        super().__init__(args)

        # Build Models
        self._build_model()


    def _build_model(self):
        super()._build_model()
        
        # simclr objective
        self.ssl_criterion = SIMCLR_Loss(self.hparams.batch_size, self.hparams.temperature_ssl, \
                            self.hparams.gpus * self.hparams.num_nodes) 

        # simclr projector
        self.simclr_projector = nn.Sequential(
            nn.Linear(self.hparams.img_embedding_dim, self.hparams.simclr_proj_hidden_dim, bias=False),
            nn.BatchNorm1d(self.hparams.simclr_proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hparams.simclr_proj_hidden_dim, self.hparams.simclr_proj_output_dim, bias=True),
        )


    def shared_forward(self, batch, batch_idx):
        feats, _, mm_loss = super().shared_forward(batch, batch_idx)

        img_feat1, img_feat2, img_feat_mm = feats
        #images_ssl1, images_ssl2, images_mm = images

        # simclr 
        z1, z2, z_mm = self.simclr_projector(img_feat1), self.simclr_projector(img_feat2), self.simclr_projector(img_feat_mm)
        ssl_loss = (self.ssl_criterion(z1, z2) + self.ssl_criterion(z1, z_mm) + self.ssl_criterion(z2, z_mm)) / 3

        # slip final loss
        loss = mm_loss + self.hparams.ssl_scale * ssl_loss
        return {"loss": loss, "mm_loss": mm_loss, "ssl_loss": ssl_loss}

    @property
    def learnable_params(self):
        extra_learnable_params = [{"type": "projector", "params": self.simclr_projector.parameters(), \
                                "lr": self.hparams.lr_img_backbone}]
        return super().learnable_params + extra_learnable_params


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("slip_simclr")

        # simclr projector
        parser.add_argument("--simclr_proj_hidden_dim", type=int, default=512)
        parser.add_argument("--simclr_proj_output_dim", type=int, default=128)

        return parent_parser


