import os
import torch
import numpy as np

from config import get_config
from src.Learner import face_learner
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-m", "--load_model", default="", type=str)
    args = parser.parse_args()

    conf = get_config()

    conf.load_model = args.load_model

    learner = face_learner(conf)

    learner.test(conf)
