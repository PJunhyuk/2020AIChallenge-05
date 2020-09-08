## data 폴더 세팅
import pandas as pd
import os
import shutil
from tqdm import tqdm

train_meta = pd.read_csv("/datasets/objstrgzip/05_face_verification_Accessories/train/train_meta.csv")
id_list = list(set(train_meta['face_id']))

data_dir = './data/train_id/'
if os.path.isdir(data_dir):
    shutil.rmtree(data_dir)
os.mkdir(data_dir)

for id in tqdm(id_list):
    os.mkdir(data_dir + str(id))
    candidate = train_meta[train_meta['face_id'] == int(id)]
    for i in range(len(candidate)):
        shutil.copyfile('/datasets/objstrgzip/05_face_verification_Accessories/train/' + candidate.iloc[i]['file_name'], data_dir + str(id) + '/' + candidate.iloc[i]['file_name'])
