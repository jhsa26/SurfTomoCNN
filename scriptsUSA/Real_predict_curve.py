#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
 @Author: HUJING
 @Time:5/21/18 3:36 PM 2018
 @Email: jhsa26@mail.ustc.edu.cn
 @Site:jhsa26.github.io
 """
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.util import *
from src.NetModel_curve import Net as Net
from config import Config
import pickle
from scipy.signal import savgol_filter as smooth
def predict(epoch,model,alpha,option):
    model.eval()
    test_x,vel_loc= Reader(). get_real_gaussian_map()
    vel_pred_total=[]
    for i in range(len(test_x)):
        # dispersion axis
        input = torch.Tensor(test_x[i])
        input = input.view([1,input.size(0),input.size(1),input.size(2)])
        # compute output
        output = model(input)  # output[batchsize,H,W]

        output = output.view([output.size(1)]).data.numpy()
        vel_pred = smooth(output,5,3)
        vel_pred_total.append(vel_pred)
    return vel_pred_total,vel_loc




option = Config()
alpha = option.alpha
# fixed seed, because pytoch initilize weight randomly
torch.manual_seed(option.seed)
# model = Net(image_width=17,
#              image_height=60,
#              image_outwidth=13,
#              image_outheight=60,
#              inchannel=2,outchannel=4)
# model = Unet(in_ch=2, out_ch=1, image_len=17, image_len_out=13)
model = Net(image_width=17,
             image_height=60,
             image_outwidth=301,
             image_outheight=1,
             inchannel=2,outchannel=4)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
print(model)
if option.pretrained:
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(option.pretrain_net))
    else:
        model.load_state_dict(torch.load(option.pretrain_net,map_location={'cuda:0':'cpu'}))
else:
    model.apply(weights_init)
print('===> Predicting')
model.cpu()
vel_total,vel_loc = predict(option.start,model,alpha,option)
os.system("rm -rf vs_cnn && mkdir vs_cnn" )
for i in range(len(vel_total)):
    vel = vel_total[i]
    prefixname = vel_loc[i]
    np.savetxt("./vs_cnn/"+prefixname+'.txt',np.array([np.arange(0,150.5,0.5),vel]).T,fmt='%10.5f')


