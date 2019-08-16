#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
@Time   15/05/2018 11:41 AM 2018
@Author HJ@USTC
@Email  jhsa26@mail.ustc.edu.cn
@Blog   jhsa26.github.io
"""

class Config(object):
    def __init__(self):
        self.filepath_disp_training = '../Dataset/TrainingDataset/USA_Tibet/disp_combine_gaussian_map/'
        self.filepath_vs_training   = '../Dataset/TrainingDataset/USA_Tibet/vs_curve/'
        self.filepath_vs_pred       = '../DataSet/TestData/real-8-50s/vs_syn_gaussian/'  # None
        self.filepath_disp_pred     = '../Dataset/TestDataset/China/8-50s/disp_combine_gaussian_map_test_4512/'
        self.batch_size = 64     # training batch size
        self.nEpochs = 600          # umber of epochs to train for
        self.lr = 0.000005
        self.seed = 123             # random seed to use. Default=123
        self.plot = True
        self.alpha=0.0000             # damping
        self.testsize=0.2
        self.pretrained =True        # False: train; True: test/predict
        self.start=600                # load the index number model to train or test/predict 
        self.pretrain_net = "./model_para/model_epoch_"+str(self.start)+".pth"
if __name__ == '__main__':
    pass
