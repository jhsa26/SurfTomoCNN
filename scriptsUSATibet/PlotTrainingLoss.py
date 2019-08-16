
#!/usr/bin/env python3  
# -*- coding: utf-8 -*-  
"""  
 @Author: HUJING
 @Time:5/29/18 10:44 PM 2018
 @Email: jhsa26@mail.ustc.edu.cn
 @Site:jhsa26.github.io
 """
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import sys
from mpl_toolkits.axes_grid.inset_locator import inset_axes
os.system('test -d  Figs_show || mkdir Figs_show')
# time_loc='2018_12_25_21'
#time_loc='2019_8_8_16' # ./output/epochInfo2019_8_8_16.txt'
epoch=600
fontsize = 18
linewidth = 1.5
temp = np.loadtxt('./output/epochInfo.txt')
fig = plt.figure(num=1, figsize=(8, 6), dpi=80, clear=True)
epoch_train_collection    = temp[0:epoch,0]
ave_train_loss_collection = temp[0:epoch,1]
ave_test_loss_collection  = temp[0:epoch,2]
ave_test_rms_collection   = temp[0:epoch,3]
# plot epoch-training loss and epoch-validation loss
plt.plot(epoch_train_collection,ave_train_loss_collection,'-bo',linewidth=2,markersize=5,
         label='Training loss')
plt.plot(epoch_train_collection,ave_test_loss_collection,'-ro',linewidth=2,markersize=5,
         label= 'Validation loss')
# plt.plot(epoch_train_collection, ave_test_rms_collection, '-go', linewidth=2, markersize=5,
#          label='validation true loss')
plt.legend(loc=0,fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xlabel('Epoch',fontsize=fontsize)
plt.ylabel('Loss(km/s)',fontsize=fontsize)
plt.axis([-10,620,-0.2,4.5])
# plt.title('CNN trained with Tibet',fontsize=fontsize)
ax = plt.gca()
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(linewidth)
ax.tick_params(axis='y', direction='in', length=6, labelrotation=0, width=2, bottom=True, top=False, left=True,
               right=False)
ax.tick_params(axis='x', direction='in', length=6, labelrotation=0, width=2, bottom=True, top=False, left=True,
               right=False)
inset_axes = inset_axes(ax, 
                    width="50%", # width = 30% of parent_bbox
                    height="40%", # height : 1 inch
                    loc='center' )
start=15
epoch_train_collection    = temp[start:epoch,0]
ave_train_loss_collection = temp[start:epoch,1]
ave_test_loss_collection  = temp[start:epoch,2]
ave_test_rms_collection   = temp[start:epoch,3]
plt.plot(epoch_train_collection,ave_train_loss_collection,'-bo',linewidth=2,markersize=5,
         label='training loss')
plt.plot(epoch_train_collection,ave_test_loss_collection,'-ro',linewidth=2,markersize=5,
         label= 'validation loss')
ax = plt.gca() 
ax.tick_params(axis='both', direction='in', length=6, labelrotation=0, width=2, bottom=True, top=False, left=True,
               right=False,labelsize=fontsize-6) 
name = './Figs_show/EpochLossCurve_withTibet'+".png"
plt.savefig(name, dpi=300,bbox_inches="tight")
plt.pause(0.5)
fig.clear()
