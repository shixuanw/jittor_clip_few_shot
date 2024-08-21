import os
import math
import random
from collections import defaultdict
from copy import copy

# import torch
# import torchvision
# import torchvision.transforms as transforms
import jittor
from jittor.dataset import ImageFolder
import jittor.transform as transforms
from utils import tfm_train_base, tfm_test_base

imagenet_classes = []
imagenet_templates = [
    "itap of a {}",
    "a bad photo of the {}",
    "a origami {}",
    "a photo of the large {}",
    "a {} in a video game",
    "art of the {}",
    "a photo of the small {}"
]


class TestA():

    def __init__(self, num_shots=4):  # num_shots代表着训练集合的样本数量

        train_preprocess = tfm_train_base
        test_preprocess = tfm_test_base

        #train_preprocess = []
        #test_preprocess = []
        # print(num_shots)
        import os

        print("Current working directory:", os.getcwd())
        self.train = ImageFolder('DataSet/Train', transform=train_preprocess)
        self.val = ImageFolder('DataSet/Test', transform=test_preprocess)
        self.test = ImageFolder('DataSet/Test', transform=test_preprocess)
        classes = open('feature.txt').read().splitlines()
        cls = open('classes_b.txt').read().splitlines()
        for c in classes:
            imagenet_classes.append(c)
        tmp = []
        for c in cls:
            c = c.split(' ')[0]
            tmp.append(c)
        self.tmp = copy(tmp)
        self.template = copy(imagenet_templates)
        self.classnames = copy(imagenet_classes)
        split_by_label_dict = defaultdict(list)
        mp = {}
        for i in range(len(self.train.imgs)):
            # val = self.train.targets[i]
            val = self.train.imgs[i][1]
            # print(self.train.imgs[i][1])
            #print(self.train.imgs[i][0].split('/')[5])
            name = self.train.imgs[i][0].split('/')[5]
            mp[name] = val
            # print(name)
            v = 0
            ok = 0
            for j in range(len(tmp)):
                if tmp[j] == name:
                    ok = 1
                    v = j
                    break

            self.tmp[val] = tmp[v]
            self.classnames[val] = imagenet_classes[v]
            split_by_label_dict[self.train.imgs[i][1]].append(self.train.imgs[i])
        # super().__init__(train_x=self.train, val=self.val, test=self.test)
