# -*- coding: utf-8 -*-
"""
Created on Thu May  9 09:44:04 2019

@author: xiaoke
"""
from torch.utils.data import Dataset
import torch.utils.data as Data
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
import matplotlib.pyplot as plt
from torch import nn


#net0 = torch.load('net0.pkl')
data0=pd.read_csv('E:/NN/Metro_train/pre_datas1.csv',index_col=0).values
means=np.random.rand(144,81)
for i in range(144):
    means[i]=data0[i::144].mean(0)
    
we_means=(d1.values+means)/2

part1=torch.from_numpy(d1.values[-20:]).unsqueeze(0).unsqueeze(2).float()
y=net(part1.cuda()/3334.,10)*3334.
y[y<0]=0

part1=torch.from_numpy(d1.values[-10:]).unsqueeze(0).unsqueeze(2).float().cuda()
part2=(torch.from_numpy(we_means[36:46]).unsqueeze(0).unsqueeze(2).float().cuda()+y)/2
x=torch.cat((part1,part2),dim=1)
y=net(x/3334.,10)*3334.
y[y<0]=0

result=torch.zeros(144,81)
for i in range(0,9):
    part1= part2
    
    part2= (torch.from_numpy(we_means[46+10*i:56+10*i]).unsqueeze(0).unsqueeze(2).float().cuda() +y)/2
    
    x=torch.cat((part1,part2),dim=1)
    result[36+10*i:56+10*i]=x[0,:,0,:]
    y=net(x/3334,10)*3334.
    y[y<0]=0


result[0:36]=torch.from_numpy(we_means[0:36])

a=result[116:136].unsqueeze(0).unsqueeze(2).cuda()
y=net(a/3334.,8)*3334.
y[y<0]=0
result[136:]=y[0,:,0,:]

d=pd.DataFrame(result.cpu().data.numpy().T.reshape(-1,1))