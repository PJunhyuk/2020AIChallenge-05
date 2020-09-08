import os
import torch
import numpy as np

from config import get_config
from src.Learner import face_learner
from src.models.efficientnet import EfficientNet


# 재현을 위해 seed 고정하기
import random
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':

    conf = get_config()
    ## conf.work_path 가 없으면 생성
    if not os.path.isdir(conf.model_path):
        os.mkdir(conf.model_path)

    # 재현을 위해 seed 고정하기
    seed_everything(conf.seed)

    learner = face_learner(conf)

    learner.train(conf)
