#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
@Time   01/07/2018 10:46 AM 2018
@Author HJ@USTC
@Email  jhsa26@mail.ustc.edu.cn
@Blog   jhsa26.github.io
"""
def  writerlogfile(writer,norm_weight,epoch, tloss, vloss,vrms, vloss_hist):
    writer.add_scalar('net/conv1', norm_weight[0], epoch)
    writer.add_scalar('net/conv2', norm_weight[1], epoch)
    writer.add_scalar('net/conv3', norm_weight[2], epoch)
    writer.add_scalar('net/conv4', norm_weight[3], epoch)
    writer.add_scalar('net/Linear1', norm_weight[4], epoch)
    writer.add_scalar('data/TrainingLoss', tloss, epoch)
    writer.add_scalar('data/ValidationLoss', vloss, epoch)
    writer.add_scalar('data/ValidationTrueRMS', vrms, epoch)
    writer.add_histogram('data/ValidationLossHistorgram', vloss_hist, global_step=epoch, bins='tensorflow')
    return writer
if __name__ == '__main__':
    pass
