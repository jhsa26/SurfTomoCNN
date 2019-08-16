#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 @Author: HUJING
 @Time:5/14/18 10:11 AM 2018
 @Email: jhsa26@mail.ustc.edu.cn
 @Site:jhsa26.github.io
 """
import os
import numpy as np
from sklearn.model_selection import train_test_split
from config import Config
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
class Reader(object):
    def __init__(self):
        self.config = Config()
        self.filepath_disp_training = self.config.filepath_disp_training
        self.filepath_vs_training = self.config.filepath_vs_training
        self.filepath_disp_pred = self.config.filepath_disp_pred
        self.filepath_vs_pred = self.config.filepath_vs_pred
        self.batch_size = self.config.batch_size
        self.disp_filenames, self.vs_filenames, self.keys = self.get_train_filename()
        self.disp_filenames_pred, self.keys_pred = self.get_realdata_filename()
    def get_batch_file(self):
        disp_filenames = self.disp_filenames
        vs_filenames = self.vs_filenames
        keys = self.keys

        train_pos, test_pos = train_test_split(range(len(keys)), test_size=Config().testsize, random_state=42)
        random.shuffle(train_pos)
        random.shuffle(test_pos)

        train_keys_num = len(train_pos)

        batch_size = self.batch_size
        batch_num = int(float(train_keys_num) / float(batch_size)) + 1
        batch_array = []
        for i in range(1, batch_num + 1):
            index1 = i * batch_size
            batch_array.append(index1 - 1)
        if batch_array[-1] >= len(train_pos) - 1:
            batch_array[-1] = len(train_pos) - 1
        return batch_array, train_pos, test_pos

    def get_train_filename(self):
        filename_dispersion_total = []
        filename_vs_total = []
        key_total = []
        if os.path.exists(self.filepath_disp_training) and os.path.exists(self.filepath_vs_training):
            files_disp = os.listdir(self.filepath_disp_training)
            # read inputs
            for file in files_disp:

                key = file[2:-4]  # add group disp and phase disp

                filename_disp = self.filepath_disp_training + file

                filename_vs = self.filepath_vs_training + file
                if os.path.exists(filename_vs) and os.path.exists(filename_disp):
                    filename_dispersion_total.append(filename_disp)

                    filename_vs_total.append(filename_vs)

                    key_total.append(key)

            return np.array(filename_dispersion_total), np.array(filename_vs_total), np.array(key_total)
        else:
            print('Input train file path is not exist, check the input path!')
            return None, None, None

    def get_predsyn_filename(self):
        filename_dispersion_total = []
        filename_vs_total = []
        key_total = []
        if os.path.exists(self.filepath_disp_pred):
            files_disp = os.listdir(self.filepath_disp_pred)
            # read inputs
            for file in files_disp:
                key = file[2:-4]
                filename_disp = self.filepath_disp_pred + file
                filename_vs = self.filepath_vs_pred + file
                if os.path.exists(filename_disp) and os.path.exists(filename_vs):
                    filename_dispersion_total.append(filename_disp)
                    filename_vs_total.append(filename_vs)
                    key_total.append(key)
            return np.array(filename_dispersion_total), np.array(filename_vs_total), np.array(key_total)
        else:
            print('Input test file path is not exist, check the input path!')
            return None, None, None

    def get_realdata_filename(self):
        filename_dispersion_total = []
        key_total = []
        if os.path.exists(self.filepath_disp_pred):

            files_disp = os.listdir(self.filepath_disp_pred)
            # read inputs
            for file in files_disp:
                key = file[2:-4]
                filename_disp = self.filepath_disp_pred + file
                if os.path.exists(filename_disp):
                    filename_dispersion_total.append(filename_disp)
                    key_total.append(key)
            return np.array(filename_dispersion_total), np.array(key_total)
        else:

            print('Input test file path is not exist, check the input path!')
            return None, None

    def get_batch_data(self, data_type, index1, index2, train_pos, test_pos):
        disp_filenames = self.disp_filenames
        vs_filenames = self.vs_filenames
        keys = self.keys

        if data_type == 'train':
            filenames = disp_filenames[train_pos[index1:index2 + 1]]
            batch_x = self.readdata(filenames, 'disp')
            filenames = vs_filenames[train_pos[index1:index2 + 1]]
            batch_y = self.readdata(filenames, 'vs')
            batch_keys = keys[train_pos[index1:index2 + 1]]
            return batch_x, batch_y, batch_keys

        elif data_type == 'test':
            sample_indexs = random.sample(test_pos, len(test_pos))
            filenames = disp_filenames[sample_indexs];
            test_x = self.readdata(filenames, 'disp')
            filenames = vs_filenames[sample_indexs];
            test_y = self.readdata(filenames, 'vs')
            test_keys = keys[test_pos]
            return test_x, test_y, test_keys
        else:
            print('check the data_type, which must be "train" or "test" ')
            return None, None, None

    def readdata(self, filenames, data_type):
        data = []
        if data_type == 'disp':
            for file in filenames:
                temp_data = []
                with open(file, 'r') as f:
                    lines = f.read().splitlines()
                    for line in lines:
                        temp = list(map(float, line.split()))
                        temp_data.append(temp)

                temp_data_temp = np.array(temp_data)
                # according to your activation function range
                #
                # remean_temp = temp_data_temp[:, 1] - np.mean(temp_data_temp[:, 1])
                # temp_data_temp[:, 1] = (remean_temp[:]-np.min(remean_temp[:]))/(np.max(remean_temp)-np.min(remean_temp)+0.01)

                data.append(temp_data_temp[:, 1:3])
            return np.array(data)
        elif data_type == "vs":
            for file in filenames:
                temp_data = []
                with open(file, 'r') as f:
                    lines = f.read().splitlines()
                    for line in lines:
                        temp = list(map(float, line.split()))
                        temp_data.append(temp)
                temp_data_temp = np.array(temp_data)
                data.append(temp_data_temp[0:300, 0:2])
            return np.array(data)
        else:
            print('check the data_type, which must be "vs" or "disp" ')
            return None

    def get_batch_gaussian_map(self, data_type, index1, index2, train_pos, test_pos):
        disp_filenames = self.disp_filenames
        vs_filenames = self.vs_filenames
        keys = self.keys

        if data_type == 'train':
            filenames = disp_filenames[train_pos[index1:index2 + 1]]
            batch_x = self.read_gaussian_map(filenames, 'disp')
            filenames = vs_filenames[train_pos[index1:index2 + 1]]
            batch_y = self.read_gaussian_map(filenames, 'vs')
            batch_keys = keys[train_pos[index1:index2 + 1]]
            return batch_x, batch_y, batch_keys

        elif data_type == 'test':
            sample_indexs = random.sample(test_pos, len(test_pos))
            filenames = disp_filenames[sample_indexs];
            test_x = self.read_gaussian_map(filenames, 'disp')
            filenames = vs_filenames[sample_indexs]
            test_y = self.read_gaussian_map(filenames, 'vs')
            test_keys = keys[test_pos]
            return test_x, test_y, test_keys
        else:
            print('check the data_type, which must be "train" or "test" ')
            return None, None, None

    def read_gaussian_map(self, filenames, data_type):
        data = []
        if data_type == 'disp':
            for file in filenames:
                temp_data_temp = np.load(file)
                data.append(temp_data_temp)
            return np.array(data)
        elif data_type == "vs":
            for file in filenames:
                temp_data_temp = np.load(file)
                data.append(temp_data_temp)
            return np.array(data)
        else:
            print('check the data_type, which must be "vs" or "disp" ')
            return None

    def get_gaussian_map_predsyn(self):
        disp_filenames = self.disp_filenames_pred
        vs_filenames = self.vs_filenames_pred
        keys = self.keys_pred
        # sample_indexs = random.sample(range(len(keys)), len(keys))
        filenames = disp_filenames  # [sample_indexs];
        test_x = self.read_gaussian_map(filenames, 'disp')
        filenames = vs_filenames  # [sample_indexs]
        test_y = self.read_gaussian_map(filenames, 'vs')
        test_keys = keys
        return test_x, test_y, test_keys

    def get_real_gaussian_map(self):
        disp_filenames = self.disp_filenames_pred
        keys = self.keys_pred
        filenames = disp_filenames  # [sample_indexs];
        # print(filenames)
        test_x = self.read_gaussian_map(filenames, 'disp')
        test_keys = keys
        return test_x, test_keys

    def get_batch_disp_gaussian_map_vs_curve(self, data_type, index1, index2, train_pos, test_pos):
        disp_filenames = self.disp_filenames
        vs_filenames = self.vs_filenames
        keys = self.keys

        if data_type == 'train':
            filenames = disp_filenames[train_pos[index1:index2 + 1]]
            batch_x = self.read_gaussian_map(filenames, 'disp')
            filenames = vs_filenames[train_pos[index1:index2 + 1]]
            batch_y = self.read_vs_curves(filenames, 'vs')
            batch_keys = keys[train_pos[index1:index2 + 1]]
            return batch_x, batch_y, batch_keys

        elif data_type == 'test':
            sample_indexs = random.sample(test_pos, len(test_pos))
            filenames = disp_filenames[sample_indexs];
            test_x = self.read_gaussian_map(filenames, 'disp')
            filenames = vs_filenames[sample_indexs]
            test_y = self.read_vs_curves(filenames, 'vs')
            test_keys = keys[test_pos]
            return test_x, test_y, test_keys
        else:
            print('check the data_type, which must be "train" or "test" ')
            return None, None, None

    def read_vs_curves(self, filenames, data_type):
        data = []
        for file in filenames:
            temp_data_temp = np.loadtxt(file)
            data.append(temp_data_temp[:, 1])
        return np.array(data)


def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.fill_(0)

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
    def forward(self, pred, truth):
        c = (pred - truth)
        d = torch.norm(c) / np.sqrt(torch.numel(c)) 
        return d
# obtain weights
def extract_weights(model):
    # for k,v in model.state_dict().iteritems():
    # print("Layer {} ".format(k))
    # print(v)
    norm_weight = []
    a = 0
    for layer in model.modules():
        a = a + 1
        if isinstance(layer, nn.Linear):
            norm_weight.append(torch.norm(layer.weight, 2))
            # print("Layer {} linear  ".format(a))
        if isinstance(layer, nn.Conv2d):
            norm_weight.append(torch.norm(layer.weight, 2))
            # print("Layer {} conv2  ".format(a))
    return norm_weight
