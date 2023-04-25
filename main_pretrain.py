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
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning import loggers as pl_loggers
from mm_pkg.methods import METHODS
from mm_pkg.model_utils.misc_utils import make_contiguous
import torch

IMG_BACKBONES = {
    "resnet2d_18",
    "resnet2d_50",
    "resnet2d_101",
    "densenet2d_121",
    "vit2d_b16",
}

TEXT_BACKBONES = {
    "microsoft/BiomedVLP-CXR-BERT-general",
    "microsoft/BiomedVLP-CXR-BERT-specialized",
}

def parse_args_pretrain():
    """Parses dataset, augmentation, pytorch lightning, model specific and additional args.
    First adds shared args such as dataset, augmentation and pytorch lightning args, then pulls the
    model name from the command and proceeds to add model specific args from the desired class. If
    wandb is enabled, it adds checkpointer args. Finally, adds additional non-user given parameters.
    Returns:
        argparse.Namespace: a namespace containing all args needed for pretraining.
    """
    parser = argparse.ArgumentParser()

    # method args
    parser.add_argument("--method", type=str)

    # model specific args
    temp_args, _ = parser.parse_known_args()
    parser = METHODS[temp_args.method].add_model_specific_args(parser)

    # pytorchlightning specific args
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--save_dir", type=str, default="./")
    parser.add_argument("--val_interval", type=float, default=1.)
    parser.add_argument("--use_ddp", action='store_true')
    parser.add_argument('--pin_mem', default=False, action='store_true')
    parser.add_argument("--mode", choices=["train", "resume"], type=str, default="train")
    parser.add_argument("--checkpoint_path", type=str, default="None")

    # general model args
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--clip_grad", type=float, default=0.)
    parser.add_argument('--pretrained', default=False, action='store_true')

    # multi-modal method for SLIP. used when SLIP is called
    parser.add_argument("--multi_modal", choices=["CLIP", "ConVIRT", "None"], type=str, default="None")

    # image model
    parser.add_argument("--img_backbone", choices=IMG_BACKBONES, type=str, default="resnet2d_50")
    parser.add_argument("--ssl_transform", default=False, action='store_true')
    parser.add_argument("--temperature_ssl", type=float, default=0.07)
    # whether freeze position embedding layer proposed by mocov3 in VIT
    parser.add_argument("--freeze_patch_embed", default=False, action='store_true')

    # text model
    parser.add_argument("--text_backbone", choices=TEXT_BACKBONES, type=str, default="microsoft/BiomedVLP-CXR-BERT-general")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--pool", type=str, default="cls")
    parser.add_argument('--full_report', default=False, action='store_true')
    parser.add_argument('--exclude_label', default=False, action='store_true')

    # encoder/projector dimension
    parser.add_argument("--img_embedding_dim", type=int, default=2048)
    parser.add_argument("--text_embedding_dim", type=int, default=768)
    parser.add_argument("--projection_dim", type=int, default=512)
    parser.add_argument("--dropout", type=int, default=0.1)

    # multimodal loss temperture and hyper-parameters
    parser.add_argument("--temperature_mm", type=float, default=0.07)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--ssl_scale", type=float, default=1.0)

    # learning rate setup
    parser.add_argument("--lr_img_backbone", type=float, default=1e-4)
    parser.add_argument("--lr_text_backbone", type=float, default=1e-4)
    parser.add_argument("--min_lr_backbone", type=float, default=1e-5)
    parser.add_argument("--min_lr_projector", type=float, default=1e-5)

    # weight decay setup
    parser.add_argument("--weight_decay", type=float, default=5e-4) # 1e-4
    parser.add_argument("--weight_decay_end", type=float, default=1e-5)

    # momentum setup
    parser.add_argument("--start_momentum", type=float, default=0.99)
    parser.add_argument("--end_momentum", type=float, default=1.)
    # momentum with sgd is used
    parser.add_argument("--momentum", type=float, default=0.90)

    # misc
    parser.add_argument("--per_warmup_steps", type=float, default=0.03)
    parser.add_argument("--optimizer",  choices=['adamw', 'sgd', 'lars'], type=str, default="adamw")
    parser.add_argument("--scheduler",  choices=['cosine', 'step'], type=str, default="cosine")

    # dataset args
    parser.add_argument("--train_df_path", type=str, default= './datasets/mimic-cxr-jpg_full_train.csv')
    parser.add_argument("--val_df_path", type=str, default= './datasets/mimic-cxr-jpg_full_val.csv')

    args = parser.parse_args()
    return args


def main():
    args = parse_args_pretrain()

    #torch.set_float32_matmul_precision('high')

    print(f"args Report:\n{args}")
    seed_everything(args.seed)

    model = METHODS[args.method](args)
    make_contiguous(model)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        save_weights_only=False,
        dirpath=args.save_dir,
    )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.save_dir)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5, verbose=False, mode="min")
    bar = TQDMProgressBar(refresh_rate=5, process_position=0)

    callbacks = [checkpoint_callback, early_stop_callback, bar]

    trainer = Trainer.from_argparse_args(
        args,
        enable_checkpointing=True,
        devices=args.gpus,
        accelerator="gpu",
        precision=args.precision,
        num_nodes=args.num_nodes,
        max_epochs=args.max_epochs,
        val_check_interval=args.val_interval,
        logger=tb_logger,
        callbacks=callbacks,
        strategy="ddp",
        sync_batchnorm=True
    )

    if args.mode == "train":
        print("Train model")
        trainer.fit(model)
    elif args.mode == "resume":
        print("Resume model")
        trainer.fit(model, ckpt_path=args.checkpoint_path)
    else:
        raise NotImplementedError("hparams.mode not implemented")


if __name__ == "__main__":
    main()


