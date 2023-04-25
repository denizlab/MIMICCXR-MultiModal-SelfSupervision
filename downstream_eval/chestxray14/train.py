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

import sys
sys.path.append('..')
from read_data import ChestXrayDataSet
from eval_utils import *
from eval_utils import densenet_model, resnet_model, vit_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_load_path", type=str, default="None")
    parser.add_argument("--model_name", type=str, choices=["resnet50", "densenet121", "vitb16"] , default="resnet50")
    parser.add_argument("--pathology", type=str , choices=["Atelectasis", "Consolidation", "Effusion", \
                                                           "Cardiomegaly", "Pneumonia", "Pneumothorax"] , default="Atelectasis")
    parser.add_argument("--save_suffix", type=str, default="")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--method", type=str , choices=["LP", "FT", "LPFT"] ,default="LPFT")
    parser.add_argument("--pretrained", default=False, action='store_true')
    parser.add_argument("--num_class", type=int, default=14)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epoch", type=int, default=5)
    parser.add_argument("--train_percent", type=float, default=1.0)
    args = parser.parse_args()
    return args


def main(args):
    BATCH_SIZE = args.batch_size
    N_EPOCHS = args.max_epoch
    N_CLASSES = args.num_class
    DATA_DIR = '/gpfs/data/denizlab/Datasets/Public/NIH_Chest_X-ray/images'
    
    if N_CLASSES == 14:
        CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                    'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
        TRAIN_LIST = 'train_list.txt'
        VAL_LIST = 'val_list.txt'
        TEST_LIST = 'test_list.txt'
    else:
        CLASS_NAMES = [args.pathology]
        print(CLASS_NAMES)
        TRAIN_LIST = f'{args.pathology}_train_list.txt'
        VAL_LIST = f'{args.pathology}_val_list.txt'
        TEST_LIST = f'{args.pathology}_test_list.txt'

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    
    train_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                     image_list_file=TRAIN_LIST,
                                     transform=transforms.Compose([
                                         transforms.RandomHorizontalFlip(),
                                         transforms.Resize(224),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         normalize
                                     ]))
   
    print("Train data length:", len(train_dataset))
    train_size = (int)(len(train_dataset) * args.train_percent)
    train_idx = torch.randperm(len(train_dataset))[:train_size]
    train_sampler = SubsetRandomSampler(train_idx) 

    val_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                   image_list_file=VAL_LIST,
                                   transform=transforms.Compose([
                                       transforms.Resize(224),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       normalize
                                   ]))
    print("Valid data length:", len(val_dataset))
 
    test_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                    image_list_file=TEST_LIST,
                                    transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.TenCrop(224),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                             num_workers=8, pin_memory=True, sampler=train_sampler)

    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=8, pin_memory=True)
    
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=8, pin_memory=True)

    if args.model_name == "densenet121":
        size = 121 
        features_dim = 2048
        out_size = N_CLASSES
        model = densenet_model(size, features_dim, out_size, pretrained=args.pretrained).cuda() # Step 0: Initialize global model and load the model
    elif args.model_name == "resnet50":
        size = 50
        features_dim = 2048
        out_size = N_CLASSES
        model = resnet_model(size, features_dim, out_size, pretrained=args.pretrained).cuda() # Step 0: Initialize global model and load the model
    elif args.model_name == "vitb16":
        size = "base" 
        features_dim = 768
        out_size = N_CLASSES
        model = vit_model(size, features_dim, out_size, pretrained=args.pretrained).cuda() # Step 0: Initialize global model and load the model
    else:
        raise NotImplementedError("Model Not Implemented!")

    model = torch.nn.DataParallel(model).cuda()
    criterion = torch.nn.BCELoss()
    
    if args.model_load_path != "None":
        checkpoint = torch.load(args.model_load_path)
        state_dict = {k.replace("img_backbone.", "module."): v for k, v in checkpoint['state_dict'].items()}
        load_status = model.load_state_dict(state_dict, strict=False)
        print(f"Load Status: {load_status}")
    
    # freeze for linear probing
    if "LP" in args.method:

        for param in model.module.backbone.parameters():
            param.requires_grad = False

        # linear-probing
        print("---------------------Linear Probing---------------------")
        # initialize the ground truth and output tensor
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
        best_model = train(args, model, optimizer, criterion, train_loader, val_loader, N_CLASSES, N_EPOCHS, CLASS_NAMES)

        # switch to evaluate mode
        print("Testing LP ...")
        evaluate(best_model, test_loader, N_CLASSES, CLASS_NAMES)

    # fine-tuning
    if "FT" in args.method:
        print("---------------------Fine-Tuning---------------------")
        if args.method == "LPFT":
            model = best_model

        # unfreeze for finetuning
        for param in model.module.backbone.parameters():
            param.requires_grad = True

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
        best_model = train(args, model, optimizer, criterion, train_loader, val_loader, N_CLASSES, N_EPOCHS, CLASS_NAMES)

        # switch to evaluate mode
        print(f"Testing {args.method} ...")
        evaluate(best_model, test_loader, N_CLASSES, CLASS_NAMES)


if __name__ == '__main__':
    args = get_args()

    if args.seed != None:
        random_seed = args.seed
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        random.seed(random_seed)

    main(args)

