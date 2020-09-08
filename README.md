# 2020AIChallenge-05

본 repo 는 [mAy-I](https://may-i.io) 팀으로 참가한 [2020 인공지능 온라인 경진대회](http://aichallenge.or.kr/main/main.do) 중 `액세서리 착용자에 대한 인식 및 조회 모델 개발` 태스크 수행을 위한 레포지토리입니다.  

[mAy-I](https://may-i.io) 는 [과학기술정보통신부](https://www.msit.go.kr/)가 주최하고 [정보통신산업진흥원](https://www.nipa.kr/)이 주관하는 [2020 인공지능 온라인 경진대회](http://aichallenge.or.kr/main/main.do)에 참가하여, **종합 5등** 을 달성하였습니다.  

본 repo 는 그 중 [[과제 05] 액세서리 착용자에 대한 인식 및 조회 모델 개발](http://aichallenge.or.kr/task/detail.do?taskId=T000005) 과 [[과제 06] (경량화 5M) 액세서리 착용자에 대한 인식 및 조회 모델 개발](http://aichallenge.or.kr/task/detail.do?taskId=T000006) 태스크를 다루고 있으며, [mAy-I](https://may-i.io) 는 두 태스크에서 모두 **2등** 을 달성하였습니다.  

대회 중 작성하였었던 코드를 아카이빙하는 것이 목적이라, *별도의 문서화나 리팩토링을 거치지 않은 점*, 양해 부탁드립니다:)

## Usage

모든 코드는 *current work directory(작업 폴더)* 하에서의 실행을 전제합니다.  

#### dependencies 설치

```bash
$ pip install -r requirements.txt
```

** 개발 과정에서 몇몇 dependencies 가 빠져있을 수 있습니다.  

#### 데이터 준비

```bash
## 기본 제공 데이터를 data 폴더에 압축 해제
$ unzip "/datasets/objstrgzip/05_face_verification_Accessories.zip" -d "./data/"
## 학습을 위한 자체 형식(id 별)의 데이터를 data/train_id에 준비
$ python src/data_prepare.py
```

#### 학습

`config.py` 파일의 변수들을 잘 수정한 후, 학습을 진행합니다.  

```bash
$ python train.py
```

#### 테스트

```bash
## 최종 결과로 테스트
$ python test.py
## 특정 weight 을 불러와서 테스트
$ python test.py -m {weight경로}
> 예시. `$ python test.py -m weights/model_2020-06-29-11-00_accuracy:0.9938_epoch:9_None.pth`
```

#### 제출

```bash
## 제출을 위한 코드 압축
$ tar cvzf mAy-I.tgz ./data/*.py ./src/*.py ./src/models/*.py ./src/models/efficientnet/*.py ./*.py ./*.md ./*.txt
$ python
> from aifactory.modules import activate, submit
> activate('', '')
> task = 6
> code = './mAy-I.tgz'
> weight = './weights/model_2020-06-29-11-00_accuracy:0.9938_epoch:9_None.pth'
> result = './prediction.txt'
> submit(task, code, weight, result)
```

## 코드 설명

#### 코드 구조

아래는 제출한 압축 파일 `mAy-I.tgz` 를 압축 해제하면 나오는 코드 구조입니다.

```
- data/
    - __init__.py
    - data_pipe.py
- src/
    - models/
        - efficientnet/
            - __init__.py
            - model.py
            - utils.py
        - mobilenetv3.py
        - model.py
    - data_prepare.py
    - dataloader.py
    - Learner.py
    - metrics.py
- config.py
- README.md
- requirements.txt
- test.py
- train.py
```

아래는 학습 및 실행 과정에서 추가로 생성되는 파일들의 코드 구조입니다.

```
- data/
    - test/
        - ...
    - train/
        - ...
    - validate/
        - ...
    - train_id/
        - ...
- runs/
- weights/
    - ...
- mAy-I.tgz
- prediction.txt
```

#### 코드 설명

기본 코드 파일들에 대한 설명입니다.

- `data/__init__.py`: data 폴더를 다른 곳에서 참조하기 위한 생성자입니다.
- `data/data_pipe.py`: 모델들을 학습하기 위한 형태로 data를 불러오는 코드입니다.
- `src/models/efficientnet/`: efficientnet 코드입니다. 세부 customizing 을 위해 [EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch/tree/master/efficientnet_pytorch) 를 그대로 가져왔습니다.
- `src/data_prepare.py`: 학습에 적합하도록 dataset 을 재가공한 `data/train_id/` 폴더를 생성하기 위한 코드입니다.
- `src/dataloader.py`: 모델들을 추론하기 위한 형태로 data를 불러오는 코드입니다.
- `src/Learner.py`: 학습 및 추론에 실질적으로 사용되는 코드입니다. model을 불러오고, optimizer를 정의하고, 학습을 진행하고, validation 및 test 를 진행합니다.
- `src/metric.py`: 학습에 사용되는 `ArcMarginProduct` 함수가 정의되어 있는 파일입니다.
- `config.py`: 학습에 필요한 환경이 설정되어 있는 파일입니다. 학습 환경을 조절 할 때는 본 파일만 수정하면 충분합니다.
- `README.md`: 본 문서입니다. 코드에 대한 전반적인 설명이 담겨있습니다.
- `requirements.txt`: 본 코드를 실행하기 위한 dependencies 들이 담겨있습니다.
- `test.py`: test를 하기 위한 코드입니다. 사전 학습된 model 을 받아 test dataset 을 기반으로 evaluation 하여 `requirements.txt` 파일을 생성합니다.
- `train.py`: train을 하기 위한 코드입니다. `config.py` 의 환경대로 모델을 학습합니다.

추가 생성되는 파일들/폴더들에 대한 설명입니다.

- `data/test/`, `data/train/`, `data/validate/`: 기본적으로 제공된 데이터 압축 파일을 그대로 압축 해제하면 나오는 파일들입니다.
- `data/train_id/`: `train/train_meta.csv` 을 기반으로, `data/train/` 의 이미지들을 identity 를 이름으로 하는 폴더 안에 저장한 폴더입니다. 학습에 직접적으로 사용됩니다.
- `runs/`: 학습 과정에서 나오는 tensorboard 로그 파일들이 저장되는 폴더입니다.
- `weights/`: 학습 과정에서 나오는 `.pth` 파일들이 저장되는 폴더입니다. 추가 학습을 위해, model 뿐 아니라 head 와 optimizer 도 저장합니다.
- `mAy-I.tgz`: 최종 제출을 위해 압축한 코드 파일입니다.
- `prediction.txt`: 최종 제출을 위해 추론한 파일입니다.

## Reference

#### Codes
- [TreB1eN/InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch/)
- [d-li14/mobilenetv3.pytorch](https://github.com/d-li14/mobilenetv3.pytorch)
- [lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)

#### Papers
- [MobileFaceNets: Efficient CNNs for Accurate Real-Time Face Verification on Mobile Devices](https://arxiv.org/abs/1804.07573)

#### mAy-I

- [2020AIChallenge-04](https://github.com/jessekim-ck/2020AIChallenge-04) : 함께 [mAy-I](https://may-i.io) 팀으로 참여한 [jessekim](https://github.com/jessekim-ck) 님의 [[과제 04] (경량화 5M) 얼굴 다각도 인식 및 조회 모델 개발
](http://aichallenge.or.kr/task/detail.do?taskId=T000004) 코드 repo 입니다. [mAy-I](https://may-i.io) 는 해당 태스크에서 **1등** 을 달성하였습니다.  
- [메이아이 홈페이지](https://may-i.io)
- [메이아이 깃허브](https://github.com/mAy-I)
