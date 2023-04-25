# Introduction
This repo contains pytorch-lightningimplementation of self-supervised and multi-modal learning benchmark on MIMIC-CXR as described in our paper: [Radiology Reports Improve Visual Representations Learned from Radiographs](https://openreview.net/pdf?id=S9EfOVFJIxQh). This repo is inspired by [solo-learn](https://github.com/vturrisi/solo-learn).

Pipeline:

<img src="https://github.com/denizlab/MIMICCXR-MutliModal-SelfSupervision/blob/main/imgs/final.png" width="500" />

# **Dependency:**
Check out our dependency at ./requirements.txt and install with
```
pip install -r ./requirements.txt
```
# **Example Pretraining command:**
You can get our pre-processed dataset at this [GoogleDrive](https://drive.google.com/drive/folders/1TqgyafDydOd7knSGhZgSWIiB2ET7mzbM?usp=share_link)

To Run Pre-training, use commands as
```
python ./main_pretrain.py --batch_size 32 --gpus 4 --num_nodes 1 --max_epochs 25 \
--lr_backbone 1e-4 --lr_projector 1e-4 --img_backbone "resnet2d_50" --max_length 128 \
--features_dim 768 --img_embedding_dim 768 --weight_decay 0.1 --optimizer "adamw" \
--method "SLIP_SIMCLR" --save_dir "slip_saved" --two_transform --pretrained \
--seed 2022 --multi_modal "CLIP"
```

or

```
python ./main_pretrain.py --batch_size <batch_size> --gpus <num_gpu> --num_nodes <num_node> \
--max_epochs <num_epochs> --lr_backbone <backbone_learning_rate> --lr_projector <projector_learning_rate> \
--img_backbone <image_backbone_name> --max_length <text_tokenizer_length> --features_dim <feature_dimension> \ 
--img_embedding_dim <image_ebmedding_dimension> --optimizer <optimizer_name> --weight_decay <weight_decay> \
--method <train_method> --save_dir <save_directory> --two_transform --pretrained --seed <seed> \
--multi_modal <multi-modal method>
```

You can find all arguments in ./main\_pretrain.py
# **Example finetuning command:**

## [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/)

```
python ./downstream_eval/chexpert/train.py --model_load_path <path_to_weights> --batch_size <batch_size> \
--max_epoch <num_epoch> --save_suffix <suffix> --seed <seed> --train_percent <train_percentage>
```


## [NIH-ChestX-ray 14](https://nihcc.app.box.com/v/ChestXray-NIHCC)

```
python ./downstream_eval/chestxray14/train.py --model_load_path <path_to_weights> --model_name "resnet50" \
--batch_size <batch_size> --max_epoch <num_epoch> --save_suffix <suffix> --seed <seed> --train_percent <train_percentage> \
--method <train_method> --num_class 14
```
## License
This repository is licensed under the terms of the GNU AGPLv3 license.

## Reference
If you found this code useful, please cite our paper:

*Radiology Reports Improve Visual Representations Learned
from Radiographs*, Medical Imaging with Deep Learning (MIDL) 2023.

```
@inproceedings{huang2023radiology,
    title = {Radiology Reports Improve Visual Representations Learned from Radiographs},
    author = {Haoxu Huang, Samyak Rawlekar, Sumit Chopra, Cem M Deniz}, 
    booktitle = {Medical Imaging with Deep Learning (MIDL)},
    year = {2023},
}
```
