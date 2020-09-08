from src.dataloader import data_loader
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config import get_config
from src.models.model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm

from tqdm import tqdm
import torch
import torch.nn.functional as F
import os
import numpy as np


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
    return [res_id, res_fc, sims]


def feed_infer(i, infer_func):
    prediction_name, prediction_class, prediction_sims = infer_func()

    # print('write output')
    predictions_str = []
    for index, name in enumerate(prediction_name):
        test_str = name + ' ' + str(prediction_class[index])
        predictions_str.append(test_str)
    output_file = 'prediction_' + str(i) + '.txt'
    with open(output_file, 'w') as file_writer:
        file_writer.write("\n".join(predictions_str))
    if os.stat(output_file).st_size == 0:
        raise AssertionError('output result of inference is nothing')

    predictions_str = []
    for index, name in enumerate(prediction_name):
        test_str = name + ' ' + str(prediction_sims[index].item())
        predictions_str.append(test_str)
    output_file = 'sim_' + str(i) + '.txt'
    with open(output_file, 'w') as file_writer:
        file_writer.write("\n".join(predictions_str))
    if os.stat(output_file).st_size == 0:
        raise AssertionError('output result of inference is nothing')


def test_def(i, model, test_dataloader, cuda, best_th):
    feed_infer(i, lambda : _infer(model, cuda, data_loader=test_dataloader, best_th=best_th))


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

    # 최적의 threshold Search
    best_th = 0

    thresholds = np.arange(0.35, 0.45, 0.001)
    best_score = 0
    dict_score = {}
    for th in tqdm(thresholds, desc="Search   "):
        res_fc_tmp = []
        for i in range(len(sims)):
            if sims[i] < th:
                result = 1
            else:
                result = 0
            res_fc_tmp.append(str(result))
        score = evaluate(val_label, res_fc_tmp)
        dict_score[th] = score
    best_th = max(dict_score, key=lambda k: dict_score[k])
    best_score = dict_score[best_th]
    print("best_score - raw: ", best_score)
    print("best_th: - raw: ", best_th)

    return best_th


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


if __name__ == '__main__':

    conf = get_config()

    validate_dataloader, validate_label_file = data_loader(root='/datasets/objstrgzip/05_face_verification_Accessories/', phase='validate', batch_size=conf.batch_size_val)
    test_dataloader, _ = data_loader(root='/datasets/objstrgzip/05_face_verification_Accessories/', phase='test', batch_size=conf.batch_size_val)

    model_list = []
    model_list.append('weights/model_2020-06-29-22-13_accuracy:0.9936_epoch:10_None.pth')
    model_list.append('weights/model_2020-06-29-22-16_accuracy:0.9932_epoch:11_None.pth')
    model_list.append('weights/model_2020-06-29-22-19_accuracy:0.9936_epoch:12_None.pth')
    model_list.append('weights/model_2020-06-29-22-22_accuracy:0.9928_epoch:13_None.pth')
    model_list.append('weights/model_2020-06-29-22-26_accuracy:0.9932_epoch:14_None.pth')
    model_list.append('weights/model_2020-06-29-22-29_accuracy:0.9928_epoch:15_None.pth')
    model_list.append('weights/model_2020-06-29-22-31_accuracy:0.9936_epoch:16_None.pth')

    model = MobileFaceNet(conf.embedding_size).to(conf.device)

    for i, model_name in enumerate(model_list):

        # load pretrained model
        print("loading.. ", model_name)
        model.load_state_dict(torch.load(model_name))

        best_th = val_def(model, validate_dataloader, validate_label_file, True)

        test_def(i, model, test_dataloader, True, best_th)
