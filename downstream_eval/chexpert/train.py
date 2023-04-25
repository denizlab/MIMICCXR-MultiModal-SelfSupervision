

import pandas as pd
import time
import csv
from PIL import Image

import torch.optim as optim
from torch.utils.data import Dataset

import sys
sys.path.append('..')
from eval_utils import *
from eval_utils import densenet_model, resnet_model, vit_model

use_gpu = torch.cuda.is_available()
print(use_gpu)

train_path = '/gpfs/data/denizlab/Datasets/Public/CheXpert-v1.0/train.csv'
valid_path = '/gpfs/data/denizlab/Datasets/Public/CheXpert-v1.0/valid.csv'

Traindata = pd.read_csv(train_path)
Traindata = Traindata[Traindata['Path'].str.contains("frontal")] # use only frontal images
# separate out part of train 
Traindata.to_csv('/gpfs/data/denizlab/Users/skr2369/Chexpert/CheXpert-v1/U1-V1/train_mod.csv', index = False)
print("Train data length:", len(Traindata))

Validdata = pd.read_csv(valid_path)
Validdata = Validdata[Validdata['Path'].str.contains("frontal")] # use only frontal images
Validdata.to_csv('/gpfs/data/denizlab/Users/skr2369/Chexpert/CheXpert-v1/U1-V1/valid_mod.csv', index = False)
print("Valid data length:", len(Validdata))

pathFileTrain = '/gpfs/data/denizlab/Users/skr2369/Chexpert/CheXpert-v1/U1-V1/train_mod.csv'
pathFileValid = '/gpfs/data/denizlab/Users/skr2369/Chexpert/CheXpert-v1/U1-V1/valid_mod.csv'

# Neural network parameters:
nnIsTrained = False     # pre-trained using ImageNet
nnClassCount = 14       # dimension of the output

# Training settings: batch size, maximum number of epochs
trBatchSize = 64 
trMaxEpoch = 5

# Parameters related to image transforms: size of the down-scaled image, cropped image
imgtransResize = (320, 320)
imgtransCrop = 224

# Class names
class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
               'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 
               'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_load_path", type=str, default="None")
    parser.add_argument("--model_name", type=str, choices=["resnet50", "densenet121", "vitb16"] , default="resnet50")
    parser.add_argument("--save_suffix", type=str, default="")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--method", type=str , choices=["LP", "FT", "LPFT"] ,default="LPFT")
    parser.add_argument("--pretrained", default=False, action='store_true')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epoch", type=int, default=5)
    parser.add_argument("--train_percent", type=float, default=1.0)
    args = parser.parse_args()
    return args


class CheXpertDataSet(Dataset):
    def __init__(self, data_PATH, transform = None, policy = "ones"):
        """
        data_PATH: path to the file containing images with corresponding labels.
        transform: optional transform to be applied on a sample.
        Upolicy: name the policy with regard to the uncertain labels.
        """
        image_names = []
        labels = []

        with open(data_PATH, "r") as f:
            csvReader = csv.reader(f)
            next(csvReader, None) # skip the header
            for line in csvReader:
                image_name = line[0]
                label = line[5:]
                
                for i in range(14):
                    if label[i]:
                        a = float(label[i])
                        if a == 1:
                            label[i] = 1
                        elif a == -1:
                            if policy == "ones":
                                label[i] = 1
                            elif policy == "zeroes":
                                label[i] = 0
                            else:
                                label[i] = 0
                        else:
                            label[i] = 0
                    else:
                        label[i] = 0
                
                image_names.append('/gpfs/data/denizlab/Datasets/Public/' + image_name)
        
#                 image_names.append('./' + image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)
    

class CheXpertTrainer():

    def train(model, dataLoaderTrain, dataLoaderVal, nnClassCount, trMaxEpoch, checkpoint, save_suffix):
        best_model = copy.deepcopy(model)
        optimizer = optim.Adam(model.parameters(), lr = 0.0001, # setting optimizer & scheduler
                               betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0) 
        loss = torch.nn.BCELoss() # setting loss function
        
        if checkpoint != None and use_gpu: # loading checkpoint
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])
            optimizer.load_state_dict(modelCheckpoint['optimizer'])
            
        # Train the network
        aurocMAX = -1
        train_start = []
        train_end = []
        for epochID in range(0, trMaxEpoch):
            train_start.append(time.time()) # training starts
            losst = CheXpertTrainer.epochTrain(model, dataLoaderTrain, optimizer, trMaxEpoch, nnClassCount, loss)
            train_end.append(time.time()) # training ends
            
            aurocMean, lossv = CheXpertTrainer.epochVal(model, dataLoaderVal, loss)
            print("Training loss: {:.3f},".format(losst))#, "Valid loss: {:.3f}".format(lossv))
            
            if aurocMean > aurocMAX:
                aurocMAX = aurocMean
                best_model = copy.deepcopy(model)
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 
                            'best_auroc': aurocMAX, 'optimizer' : optimizer.state_dict()}, 
                            'model_saved/' + 'm-epoch_FL_' + save_suffix + str(epochID + 1) + '.pth.tar')
                print('Epoch ' + str(epochID + 1) + ' [save] validation auroc = ' + str(aurocMean) + \
                      ' validation loss = ' + str(lossv))
            else:
                print('Epoch ' + str(epochID + 1) + ' [----] validation auroc = ' + str(aurocMean) + \
                      ' validation loss = ' + str(lossv))

        train_time = np.array(train_end) - np.array(train_start)
        print("Training time for each epoch: {} seconds".format(train_time.round(0)))
        del model
        torch.cuda.empty_cache()
        
        return best_model
       
        
    def epochTrain(model, dataLoaderTrain, optimizer, epochMax, classCount, loss):
        losstrain = 0
        model.train()

        for batchID, (varInput, target) in enumerate(Bar(dataLoaderTrain)):
            
            varTarget = target.cuda(non_blocking = True)
            varOutput = model(varInput)
            lossvalue = loss(varOutput, varTarget)
            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()
            
            losstrain += lossvalue.item()
         
        return losstrain / len(dataLoaderTrain)
    
    
    def epochVal(model, dataLoaderVal, loss):
        lossval = 0
        model.eval()
        
        outGT = torch.FloatTensor()
        outPRED = torch.FloatTensor()
        with torch.no_grad():
            for batchID, (varInput, target) in enumerate(Bar(dataLoaderVal)):
                target = target.cuda()
                outGT = torch.cat((outGT, target.cpu()), 0)

                varTarget = target.cuda(non_blocking = True)
                varOutput = model(varInput)
                outPRED = torch.cat((outPRED, varOutput.cpu()), 0)

                lossvalue = loss(varOutput, varTarget)
                lossval += lossvalue.item()
            
        aurocIndividual = compute_AUCs(outGT, outPRED.detach(), nnClassCount)
        aurocMean = np.array(aurocIndividual).mean()
         
        return aurocMean, lossval / len(dataLoaderVal)
    
    
    def test(model, dataLoaderTest, nnClassCount, checkpoint, class_names):
        model.eval()
        
        if checkpoint != None and use_gpu:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])

        outGT = torch.FloatTensor()
        outPRED = torch.FloatTensor()
        
        with torch.no_grad():
            for i, (varInput, target) in enumerate(dataLoaderTest):

                target = target.cuda()
                outGT = torch.cat((outGT, target.cpu()), 0)
                varOutput = model(varInput)
                outPRED = torch.cat((outPRED, varOutput.cpu()), 0)
                
        aurocIndividual = compute_AUCs(outGT, outPRED, nnClassCount)
        aurocMean = np.array(aurocIndividual).mean()
        print('AUROC mean ', aurocMean)
        
        for i in range (0, len(aurocIndividual)):
            print(class_names[i], ' ', aurocIndividual[i])


def main(args):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]  # mean of ImageNet dataset(for normalization)
    IMAGENET_STD = [0.229, 0.224, 0.225]   # std of ImageNet dataset(for normalization)

    # Training settings: batch size, maximum number of epochs
    trBatchSize = args.batch_size 
    trMaxEpoch = args.max_epoch

    # Tranform data
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    transformList = []

    transformList.append(transforms.Resize((imgtransCrop, imgtransCrop))) # 224
    transformList.append(transforms.ToTensor())
    transformSequence = transforms.Compose(transformList)

    # Load dataset
    datasetTrain = CheXpertDataSet(pathFileTrain, transformSequence, policy = "ones")
    print("Train data length:", len(datasetTrain))
    train_size = (int)(len(datasetTrain) * args.train_percent)
    train_idx = torch.randperm(len(datasetTrain))[:train_size]
    train_sampler = SubsetRandomSampler(train_idx)
    dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, 
                            shuffle=False,  num_workers=10, pin_memory=True,
                            sampler=train_sampler)

    datasetValid = CheXpertDataSet(pathFileValid, transformSequence)
    print("Valid data length:", len(datasetValid))
    dataLoaderVal = DataLoader(dataset=datasetValid, batch_size=trBatchSize, 
                            shuffle=False, num_workers=10, pin_memory=True)


    if args.model_name == "densenet121":
        size = 121
        features_dim = 2048
        out_size = nnClassCount
        model = densenet_model(size, features_dim, out_size, pretrained=args.pretrained).cuda() # Step 0: Initialize global model and load the model
    elif args.model_name == "resnet50":
        size = 50
        features_dim = 2048
        out_size = nnClassCount
        model = resnet_model(size, features_dim, out_size, pretrained=args.pretrained).cuda() # Step 0: Initialize global model and load the model
    elif args.model_name == "vitb16":
        size = "base" 
        features_dim = 768
        out_size = nnClassCount
        model = vit_model(size, features_dim, out_size, pretrained=args.pretrained).cuda() # Step 0: Initialize global model and load the model
    else:
        raise NotImplementedError("Model Not Implemented!")

    model = torch.nn.DataParallel(model).cuda()
    
    if args.model_load_path != "None":
        checkpoint = torch.load(args.model_load_path)
        state_dict = {k.replace("img_backbone.", "module."): v for k, v in checkpoint['state_dict'].items()}
        #print(state_dict.keys())
        load_status = model.load_state_dict(state_dict, strict=False)
        print(f"Load Status: {load_status}")
    
    # freeze for linear probing
    if "LP" in args.method:
        
        for param in model.module.backbone.parameters():
            param.requires_grad = False
            
        # linear-probing
        print("---------------------Linear Probing---------------------")
        # initialize the ground truth and output tensor
        best_model = CheXpertTrainer.train(model, dataLoaderTrain, dataLoaderVal, nnClassCount, trMaxEpoch, checkpoint = None, save_suffix = args.save_suffix)
        print("Model trained")
        
        # switch to evaluate mode
        print("Testing LP ...")
        CheXpertTrainer.test(best_model, dataLoaderVal, nnClassCount, None, class_names)
        
    # fine-tuning
    if "FT" in args.method:
        print("---------------------Fine-Tuning---------------------")
        if args.method == "LPFT":
            model = best_model
        
        # unfreeze for finetuning
        for param in model.module.backbone.parameters():
            param.requires_grad = True

        best_model = CheXpertTrainer.train(model, dataLoaderTrain, dataLoaderVal, nnClassCount, trMaxEpoch, checkpoint = None, save_suffix = args.save_suffix)

        # switch to evaluate mode
        print(f"Testing {args.method} ...")
        CheXpertTrainer.test(best_model, dataLoaderVal, nnClassCount, None, class_names)


if __name__ == "__main__":
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

