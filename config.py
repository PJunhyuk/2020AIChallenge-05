from easydict import EasyDict as edict
from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms as trans

from src.models.efficientnet import EfficientNet


def get_config():
    conf = edict()

    ## Fixed
    conf.data_path = Path('data')
    conf.model_path = Path('weights')
    conf.save_path = Path('.')
    conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf.ce_loss = CrossEntropyLoss()
    conf.pin_memory = True

    ## network 세팅
    conf.net_mode = 'mobilefacenet' # [ir, ir_se, mobilefacenet, efficientnet, mobilenetv3, DW_seesawFaceNetv2, seesaw_shuffleFaceNet, seesaw_shareFaceNet]
    if conf.net_mode in ['ir', 'ir_se', 'efficientnet']:
        conf.net_depth = 0 # ir_se, ir 에서는 [50, 100, 152]// efficientnet 에서는 [0, 1, 2, 3, ...]
    if conf.net_mode in ['ir', 'ir_se']:
        conf.drop_ratio = 0.6
    if conf.net_mode == 'efficientnet':
        conf.input_size = int(EfficientNet.get_image_size('efficientnet-b' + str(conf.net_depth)))
    else:
        conf.input_size = 112

    conf.batch_size = 50
    if conf.net_mode == 'mobilefacenet':
        conf.batch_size_val = 400 # validation 을 위한 batch_size 크기
    else:
        conf.batch_size_val = conf.batch_size

    ## scheduler 세팅
    conf.scheduler_mode = 'multistep' # [auto, multistep]
    if conf.scheduler_mode == 'multistep':
        conf.milestones = [7]
    if conf.scheduler_mode == 'auto':
        conf.patience = 1

    ## optimizer 세팅
    conf.optimizer_mode = 'SGD' # [Adam, RMSprop, SGD]
    ## mobilefacenet -> [SGD], efficientnet -> [Adam, RMSprop], mobilenetv3 -> [Adam], ir/ir_se -> [SGD]

    conf.embedding_size = 512
    conf.epochs = 9
    conf.lr = 1e-2
    conf.lr_gamma = 0.1 # lr 을 떨구는 비율
    conf.num_workers = 1
    conf.seed = 0
    conf.momentum = 0.9
    conf.use_best_th = True

    ## pretrained model 로딩 관련
    conf.load_model = ""
    conf.start_step = 0
    conf.start_epoch = 0

    return conf
