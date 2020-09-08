import torch
from torch import optim
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from PIL import Image
from torchvision import transforms as trans
import math
import torch.nn.functional as F
import os
import datetime
import sys
import sklearn.metrics

from data.data_pipe import de_preprocess, get_train_loader
from .dataloader import data_loader
from .metrics import ArcMarginProduct
from .models.model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
from .models.efficientnet import EfficientNet
from .models.mobilenetv3 import mobilenetv3_large, mobilenetv3_small


## utils
def separate_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])
    return paras_only_bn, paras_wo_bn

def get_time():
    return (str(datetime.datetime.now())[:-10]).replace(' ','-').replace(':','-')


## evaluation
def evaluation_metrics(prediction_file, testset_path):
    prediction_labels = read_prediction_pt(prediction_file)
    gt_labels = read_prediction_gt(testset_path)
    return evaluate(prediction_labels, gt_labels)

def read_prediction_pt(file_name):
    with open(file_name) as f:
        lines = f.readlines()
    pt = [list(l.replace("\n","").split(' '))[2] for l in lines]
    return pt

def read_prediction_gt(file_name):
    with open(file_name) as f:
        lines = f.readlines()
    gt = [list(l.replace("\n", "").split(','))[2] for l in lines]
    return gt[1:]

def evaluate(prediction_labels, gt_labels):
    f1 = sklearn.metrics.f1_score(gt_labels,prediction_labels, pos_label='0')
    return f1


def _infer(model, cuda, data_loader, best_th):
    res_id = []
    res_fc = []
    sims = []
    image_names = []
    for i, data in tqdm(enumerate(data_loader), desc="Validate ", total=len(data_loader)):
        with torch.no_grad():
            iter1_, x0, iter2_, x1, label = data
            if cuda:
                x0 = x0.cuda()
                x1 = x1.cuda()
            output1 = model(x0)
            output2 = model(x1)
            sim = F.cosine_similarity(output1, output2)
            # sim = cosin_metric(output1, output2)
            for i in range(len(iter1_)):
                image_name = iter1_[i] + ' ' + iter2_[i]
                sims.append(sim[i])
                image_names.append(image_name)
    if best_th != 0:
        print("use best_th - ", best_th)
        temp = best_th
    else:
        # 그냥 반으로 나눠서 0, 1 부여
        temp = sorted(sims)[int(len(sims) / 2)]
    for i in range(len(sims)):
        if sims[i] < temp:
            result = 1 # 다른 사람
        else:
            result = 0 # 동일인
        res_fc.append(result)
        res_id.append(image_names[i])
    return [res_id, res_fc]


def feed_infer(output_file, infer_func):
    prediction_name, prediction_class = infer_func()

    # print('write output')
    predictions_str = []
    for index, name in enumerate(prediction_name):
        test_str = name + ' ' + str(prediction_class[index])
        predictions_str.append(test_str)
    with open(output_file, 'w') as file_writer:
        file_writer.write("\n".join(predictions_str))

    if os.stat(output_file).st_size == 0:
        raise AssertionError('output result of inference is nothing')


def validate(prediction_file, model, validate_dataloader, validate_label_file, cuda, epoch_):
    feed_infer(prediction_file, lambda : _infer(model, cuda, data_loader=validate_dataloader, best_th=0))

    metric_result = evaluation_metrics(prediction_file, validate_label_file)

    return metric_result


def test_def(prediction_file, model, test_dataloader, cuda, best_th):
    feed_infer(prediction_file, lambda : _infer(model, cuda, data_loader=test_dataloader, best_th=best_th))


def val_def(model, val_dataloader, val_label_file, cuda):
    val_label = read_prediction_gt(val_label_file)
    res_fc = []
    sims = []

    # load 한 model 에 validation images 들을 넣어서 그 cosine_similarity 를 sims 로 반환
    for i, data in tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc="Validate "):
        with torch.no_grad():
            iter1_, x0, iter2_, x1, label = data
            if cuda:
                x0 = x0.cuda()
                x1 = x1.cuda()
            output1 = model(x0)
            output2 = model(x1)
            sim = F.cosine_similarity(output1, output2)
            # sim = cosin_metric(output1, output2)
            for i in range(len(iter1_)):
                image_name = iter1_[i] + ' ' + iter2_[i]
                sims.append(sim[i])

    # 그냥 반으로 나눠서 0, 1 부여
    temp = sorted(sims)[int(len(sims) / 2)]
    for i in range(len(sims)):
        if sims[i] < temp:
            result = '1' # 다른 사람
        else:
            result = '0' # 동일인
        res_fc.append(result)
    score = evaluate(val_label, res_fc)
    print("score - half: ", score)

    # 최적의 threshold Search
    best_th = 0
    thresholds =np.arange(0.0, 0.5, 0.01)
    best_score =  0
    for th in tqdm(thresholds, desc="Search   "):
        res_fc_tmp = []
        for i in range(len(sims)):
            if sims[i] < th:
                result = 1
            else:
                result = 0
            res_fc_tmp.append(str(result))
        score = evaluate(val_label, res_fc_tmp)
        if score < best_score:
            best_th = th
            break
        best_score = score

    print("best_score - raw: ", best_score)
    print("best_th: - raw: ", best_th)

    return best_th


class face_learner(object):
    def __init__(self, conf, inference=False):
        # for time logging
        self.time_start = datetime.datetime.now()
        print("time_start: ", self.time_start+datetime.timedelta(hours=9))
        print(conf)
        if conf.net_mode == 'mobilefacenet':
            self.model = MobileFaceNet(conf.embedding_size).to(conf.device)
            print('MobileFaceNet model generated')
        elif conf.net_mode == 'efficientnet':
            self.model = EfficientNet.from_name('efficientnet-b' + str(conf.net_depth))
            # fc1 = torch.nn.Linear(1280, conf.embedding_size, bias=True)
            bn1 = torch.nn.BatchNorm1d(conf.embedding_size)
            # self.model = torch.nn.Sequential(self.model, fc1, bn1)
            fc_in = self.model._fc.in_features
            self.model._fc = torch.nn.Linear(fc_in, conf.embedding_size)
            self.model = torch.nn.Sequential(self.model, bn1)
            self.model = self.model.to(conf.device)
        elif conf.net_mode == 'mobilenetv3':
            self.model = mobilenetv3_large()
            fc_in = self.model.classifier[3].in_features
            self.model.classifier[3] = torch.nn.Linear(fc_in, conf.embedding_size)
            self.model = self.model.to(conf.device)
        elif conf.net_mode in ['ir', 'ir_se']:
            self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
            print('{}_{} model generated'.format(conf.net_mode, conf.net_depth))
        else:
            print('net_mode error!')
            sys.exit(-1)

        # check parameter of model
        print("------------------------------------------------------------")
        total_params = sum(p.numel() for p in self.model.parameters())
        print("num of parameter :", total_params/1000000, "M")
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("num of trainable_ parameter :", trainable_params/1000000, "M")
        print("------------------------------------------------------------")

        if not inference:
            self.loader, self.class_num = get_train_loader(conf)

            self.writer = SummaryWriter()
            self.step = conf.start_step
            if conf.net_mode in ['efficientnet', 'mobilenetv3']:
                self.head = ArcMarginProduct(conf.embedding_size, self.class_num, s=30, m=0.5, easy_margin=False).to(conf.device)
            else:
                self.head = Arcface(embedding_size=conf.embedding_size, classnum=self.class_num, s=30).to(conf.device)
            self.lr_gamma = conf.lr_gamma

            paras_only_bn, paras_wo_bn = separate_bn_paras(self.model)

            print("Initialize optimizer")
            if conf.net_mode == 'mobilefacenet':
                if conf.optimizer_mode == 'SGD':
                    self.optimizer = optim.SGD([
                                        {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                                        {'params': [paras_wo_bn[-1]] + [self.head.kernel], 'weight_decay': 4e-4},
                                        {'params': paras_only_bn}
                                    ], lr = conf.lr, momentum = conf.momentum)
            elif conf.net_mode == 'efficientnet':
                if conf.optimizer_mode == 'Adam':
                    self.optimizer = optim.Adam([
                                    {'params': self.model.parameters()}
                                ], lr = conf.lr)
                elif conf.optimizer_mode == 'RMSprop':
                    self.optimizer = optim.RMSprop([
                                    {'params': self.model.parameters()}
                                ], lr = conf.lr, momentum = conf.momentum)
            elif conf.net_mode == 'mobilenetv3':
                self.optimizer = optim.Adam([
                                    {'params': self.model.parameters()}
                                ], lr = conf.lr)
            else: ## ir, ir_se
                self.optimizer = optim.SGD([
                                    {'params': paras_wo_bn + [self.head.kernel], 'weight_decay': 5e-4},
                                    {'params': paras_only_bn}
                                ], lr = conf.lr, momentum = conf.momentum)
            print(self.optimizer)

            # for i in range(conf.start_epoch): ## start_epoch 만큼 scheduler 작동시켜놓아주기
            #     self.scheduler.step()
            if conf.scheduler_mode == 'auto':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=self.lr_gamma, patience=conf.patience, verbose=True)
            elif conf.scheduler_mode == 'multistep':
                self.scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer, milestones=conf.milestones, gamma= self.lr_gamma)

            self.board_loss_every = len(self.loader)//100
            self.evaluate_every = len(self.loader)
            self.save_every = len(self.loader)
        else:
            self.threshold = conf.threshold


    def save_state(self, conf, accuracy, e, to_save_folder=False, extra=None, model_only=False):
        if to_save_folder:
            save_path = conf.save_path
            torch.save(self.model.state_dict(), save_path/('model_last.pth'))
        else:
            save_path = conf.model_path
            torch.save(
                self.model.state_dict(), save_path /
                ('model_{}_accuracy:{}_epoch:{}_{}.pth'.format(get_time(), accuracy, e+1, extra)))
        if not model_only:
            torch.save(
                self.head.state_dict(), save_path /
                ('head_{}_accuracy:{}_epoch:{}_{}.pth'.format(get_time(), accuracy, e+1, extra)))
            torch.save(
                self.optimizer.state_dict(), save_path /
                ('optimizer_{}_accuracy:{}_epoch:{}_{}.pth'.format(get_time(), accuracy, e+1, extra)))
            # torch.save(
            #     self.scheduler.state_dict(), save_path /
            #     ('scheduler_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra)))


    def load_state(self, conf, fixed_str, from_save_folder=False, model_only=False):
        if from_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        self.model.load_state_dict(torch.load(save_path/'model_{}'.format(fixed_str)))
        if not model_only:
            self.head.load_state_dict(torch.load(save_path/'head_{}'.format(fixed_str)))
            self.optimizer.load_state_dict(torch.load(save_path/'optimizer_{}'.format(fixed_str)))
            # self.scheduler.load_state_dict(torch.load(save_path/'scheduler_{}'.format(fixed_str)))


    def train(self, conf):

        # load pretrained model
        if conf.load_model != "":
            print("loading.. ", conf.load_model)
            self.load_state(conf, conf.load_model, from_save_folder=False, model_only=False)
            ## update lr
            for params in self.optimizer.param_groups:
                params['lr'] = conf.lr

        self.model.train()
        running_loss = 0.
        if conf.net_mode == 'efficientnet':
            validate_dataloader, validate_label_file = data_loader(root='/datasets/objstrgzip/05_face_verification_Accessories/', phase='validate', batch_size=conf.batch_size_val, use_efficientnet=True, img_size=conf.input_size)
        else:
            validate_dataloader, validate_label_file = data_loader(root='/datasets/objstrgzip/05_face_verification_Accessories/', phase='validate', batch_size=conf.batch_size_val, use_efficientnet=False, img_size=conf.input_size)

        for e in range(conf.start_epoch, conf.epochs):
            ## train
            print('epoch {:4d} started -- lr: {:.3e}'.format(e+1, self.optimizer.param_groups[0]['lr']))
            epoch_loss = []
            for imgs, labels in tqdm(iter(self.loader), desc="Train    "):
                imgs = imgs.to(conf.device)
                labels = labels.to(conf.device)
                self.optimizer.zero_grad()
                embeddings = self.model(imgs)
                thetas = self.head(embeddings, labels)
                loss = conf.ce_loss(thetas, labels)
                loss.backward()
                running_loss += loss.item()
                epoch_loss.append(loss.item())
                self.optimizer.step()

                if self.step % self.board_loss_every == 0 and self.step != conf.start_step:
                    loss_board = running_loss / self.board_loss_every
                    self.writer.add_scalar('step/train_loss', loss_board, self.step)
                    running_loss = 0.
                self.step += 1
            epoch_loss_avg = sum(epoch_loss)/len(epoch_loss)
            ## evaluation
            self.model.eval()
            accuracy = validate('prediction.txt', self.model, validate_dataloader, validate_label_file, True, e)
            print("epoch {:4d} -- Accuracy: {:.4f} / Loss: {:2.4f} -- total elapsed: {}".format(e+1, accuracy, epoch_loss_avg, datetime.datetime.now()-self.time_start))
            self.writer.add_scalar('epoch/accuracy', accuracy, e+1)
            self.writer.add_scalar('epoch/loss', epoch_loss_avg, e+1)
            self.model.train()
            if conf.scheduler_mode == 'auto':
                self.scheduler.step(accuracy)
            elif conf.scheduler_mode == 'multistep':
                self.scheduler.step()
            self.save_state(conf, accuracy, e)

        self.save_state(conf, accuracy, e=0, to_save_folder=True, extra='final')
        self.writer.close()


    def test(self, conf):
        # load pretrained model
        if conf.load_model != "":
            print("loading.. ", conf.load_model)
            self.model.load_state_dict(torch.load(conf.load_model))

        self.model.eval()

        print("Validation")
        validate_dataloader, validate_label_file = data_loader(root='/datasets/objstrgzip/05_face_verification_Accessories/', phase='validate', batch_size=conf.batch_size_val)
        best_th = val_def(self.model, validate_dataloader, validate_label_file, True)
        # best_th = 0

        print("Test")
        test_dataloader, _ = data_loader(root='/datasets/objstrgzip/05_face_verification_Accessories/', phase='test', batch_size=conf.batch_size_val)
        test_def('prediction.txt', self.model, test_dataloader, True, best_th)
