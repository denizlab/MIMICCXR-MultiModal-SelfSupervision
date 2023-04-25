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


# util libraries
import math
import re

# preprocessing libraries
import numpy as np
import pandas as pd

# torch libraries
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

# user defined files
from ..data_utils.augmentation_utils import *
from ..data_utils.augmentation_utils import TrainTransform


LABELS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", \
        "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", \
        "Emphysema", "Fibrosis", "Pleural Thickening", "Hernia", "Enlarged Cardiomediastinum", \
        "Lung Lesion", "Lung Opacity", "Pleural Effusion", "Fracture"]

class MIMIC_CXR_Unsupervised(Dataset):
    def __init__(self, args, dict_image_mapping, data_df_path, full_report=True, ssl_transform=True, train=True):
        self.args = args
        self.full_report = full_report
        self.ssl_transform = ssl_transform
        self.data_df = pd.read_csv(data_df_path, sep='\t')
        self.train = train
        self.dict_image_mapping = dict_image_mapping

    def __getitem__(self, index):
        args = self.args
        # load images
        #image_path = self.data_df.iloc[index]['dicom_path']
        image_path = self.data_df.iloc[index]['jpg_path']
        image = self.dict_image_mapping[image_path]
        # to PIL images
        PIL_image = Image.fromarray(image).convert("RGB")

        # texts
        impression = self.data_df.iloc[index]['impression']
        findings = self.data_df.iloc[index]['findings']

        # if not using full report, only impression is used
        if self.full_report and isinstance(findings, str):
            text = impression + findings
        else:
            text = impression

        # if exclude pathologies
        if self.args.exclude_label:
            # case insensitive string replacement
            repl = "[MASK]"
            for label in LABELS:
                compiled = re.compile(re.escape(label), re.IGNORECASE)
                text = compiled.sub(repl, text)

        # transform images
        transform = TrainTransform(self.ssl_transform)
        images = transform(PIL_image)

        if self.ssl_transform:
            return images[0], images[1], images[2], text
        else:
            return images, text

    def __len__(self):
        return len(self.data_df)

