# -*- coding: utf-8 -*-
"""
Created on Wed May  1 18:29:18 2019

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

import convlstm1
from convlstm1 import ConvLSTM,GCLayer
from graph import laplacian
class PopDataset(Dataset):
    """ 数据集演示 """
    def __init__(self,t1,t2):
        """实现初始化方法，在初始化的时候将数据读载入"""
        self.data0=pd.read_csv('E:/NN/Metro_train/del_datas0.csv',index_col=0).values
        self.data0=self.data0/self.data0.max()
        self.data1=pd.read_csv('E:/NN/Metro_train/del_datas1.csv',index_col=0).values
        self.data1=self.data1/self.data1.max()
        self.t1=t1
        self.t2=t2
    def __len__(self):
        '''
        返回df的长度
        '''
        return len(self.data0)-self.t1-self.t2
    def __getitem__(self, idx):
        '''
        根据IDX返回一列数据
        '''
        
        #x0=torch.from_numpy(self.data0[idx:idx+self.t1])
        x1=torch.from_numpy(self.data1[idx:idx+self.t1])
        #x=torch.stack([x0,x1], dim=1)
        x=x1.unsqueeze(1)
        #y0=torch.from_numpy(self.data0[idx+self.t1:idx+self.t1+self.t2])
        y1=torch.from_numpy(self.data1[idx+self.t1:idx+self.t1+self.t2])
        #y=torch.stack([y0,y1], dim=1)
        y=y1.unsqueeze(1)
        return (x,y)

def MAELoss(out,truth):
    err=out.data.numpy()-truth.data.numpy()
    MAE = np.average(np.abs(err))
    return MAE

train_data=PopDataset(t1=20,t2=10)
train_loader = Data.DataLoader(dataset=train_data, batch_size=15, shuffle=True)

mex=pd.read_csv('E:/NN/Metro_train/Metro_roadMap.csv',index_col=0)
a=mex.as_matrix()
w=sp.csc_matrix(a).astype(float)
L=laplacian(w)

class Net(nn.Module):
    def __init__(self, L, input_dim, hidden_dim, k, num_layers,
                 batch_first=True, bias=True, return_all_layers=True):
        super(Net, self).__init__()
        self.L=L
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.k=k
        self.num_layers=num_layers
        self.clstm=ConvLSTM(self.L,self.input_dim,self.hidden_dim,self.k,self.num_layers,
                 batch_first=True, bias=True, return_all_layers=True).cuda()
        self.conv0= GCLayer(L=self.L,Fin=sum(self.hidden_dim),
                              Fout=self.input_dim,
                              K=1
                              )
    def forward(self,input_tensor,future=1):
        out=[]
        inner_states=self.clstm(input_tensor)
        x=torch.cat(inner_states[0],dim=2)[:,-1,:,:]
        x=self.conv0(x)
        x=nn.functional.tanh(x)
        out.append(x)
        for layer_idx in range(self.num_layers):

            h, c = inner_states[1][layer_idx]
            
            for t in range(future-1):

                h, c = self.clstm.cell_list[layer_idx](input_tensor=x,
                                                 cur_state=[h, c])
                x=self.conv0(h)
                x=nn.functional.tanh(x)
                out.append(x)
        return torch.stack(out,dim=1)
net=Net(L,1,[64],8,1).cuda()
loss_fn = torch.nn.MSELoss()
optimizer     = torch.optim.RMSprop(net.parameters(), lr=0.001)
testy=train_data[2000][1].float().cuda()
testx=train_data[2000][0].float().unsqueeze(0).cuda()

total_loss=[]
for t in range(0,500):
    loss1=[]
    for step,(x,y) in enumerate(train_loader):
        x=x.float().cuda()
        y=y.float().cuda()
        out=net(x,future=10)
        loss = loss_fn(out, y)     # 计算两者的误差
        optimizer.zero_grad()   # 清空上一步的残余更新参数值
        loss.backward()         # 误差反向传播, 计算参数更新值
        optimizer.step()   
        mae=MAELoss(out.cpu(),y.cpu())
        loss1.append(mae)

    
    print('第{}个周期，MAE：{}'.format(t,sum(loss1)/len(loss1)))
    total_loss.append(sum(loss1)/len(loss1))
    #testout=net(testx,future=10)
    #print(testy.cpu()[:,0,0:5].data.numpy())
    #print(testout[0].cpu()[:,0,0:5].data.numpy())