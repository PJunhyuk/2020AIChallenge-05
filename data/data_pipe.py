from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import transforms as trans
from torchvision.datasets import ImageFolder
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import cv2
import bcolz
import pickle
import torch
# import mxnet as mx
from tqdm import tqdm

def de_preprocess(tensor):
    return tensor*0.5 + 0.5

def get_eff_train_dataset(imgs_folder, img_size):
    train_transform = trans.Compose([
        trans.RandomHorizontalFlip(),
        trans.Resize((img_size, img_size)),
        trans.ToTensor(),
        trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # EfficientNet
    ])
    ds = ImageFolder(imgs_folder, train_transform)
    class_num = ds[-1][1] + 1
    return ds, class_num

def get_train_dataset(imgs_folder):
    train_transform = trans.Compose([
        trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    ds = ImageFolder(imgs_folder, train_transform)
    class_num = ds[-1][1] + 1
    return ds, class_num

def get_train_loader(conf):
    if conf.net_mode == 'efficientnet':
        ds, class_num = get_eff_train_dataset(conf.data_path/'train_id', conf.input_size)
    else:
        ds, class_num = get_train_dataset(conf.data_path/'train_id')
    loader = DataLoader(ds, batch_size=conf.batch_size, shuffle=True, pin_memory=conf.pin_memory, num_workers=conf.num_workers)
    return loader, class_num
