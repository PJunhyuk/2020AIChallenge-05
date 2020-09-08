import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils import data
import torchvision.transforms as transforms

from tqdm import tqdm


class CustomDataset(data.Dataset):
    def __init__(self, root, phase='validate', transform=None):
        self.root = root
        self.phase = phase
        self.labels = {}
        self.transform = transform
        self.just_load = False
        self.label_path = os.path.join(root, self.phase, self.phase + '_label.csv')
        self.dir = os.path.join(root, self.phase)
        # used to prepare the labels and images path
        self.direc_df = pd.read_csv(self.label_path)
        self.direc_df.columns = ["image1", "image2", "label"]

    def __getitem__(self, index):
        # getting the image path
        image1_path = os.path.join(self.dir, self.direc_df.iat[index, 0])
        image2_path = os.path.join(self.dir, self.direc_df.iat[index, 1])
        # Loading the image
        img0 = Image.open(image1_path)
        img1 = Image.open(image2_path)

        # Apply image transformations
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        if self.phase == "test":
            dummy = ""
            return (self.direc_df.iat[index, 0], img0, self.direc_df.iat[index, 1], img1, dummy)
        else:
            return (self.direc_df.iat[index, 0], img0, self.direc_df.iat[index, 1], img1,
                    torch.from_numpy(np.array([int(self.direc_df.iat[index, 2])], dtype=np.float32)))

    def __len__(self):
        return len(self.direc_df)

    def get_label_file(self):
        return self.label_path

def data_loader(root, batch_size=64, phase='validate', use_efficientnet=False, img_size=112):
    if use_efficientnet == True:
        dataset = CustomDataset(root, phase, transform=transforms.Compose([
            transforms.Resize((img_size,img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # EfficientNet
        ]))
    else:
        dataset = CustomDataset(root, phase, transform=transforms.Compose([
            # transforms.Resize((128,128)), # resnet_face18
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]))
    dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    return dataloader, dataset.get_label_file()
