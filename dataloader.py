# -*- coding: utf-8 -*-
"""
Created on Wed May  1 18:29:18 2019

@author: xiaoke
"""
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch

class PopDataset(Dataset):
    """ 数据集演示 """
    def __init__(self,t1,t2):
        """实现初始化方法，在初始化的时候将数据读载入"""
        self.data0=pd.read_csv('pre_datas0.csv',index_col=0)
        self.data1=pd.read_csv('pre_datas1.csv',index_col=0)
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
        x0=torch.from_numpy(self.data0.iloc[idx:idx+self.t1].values)
        x1=torch.from_numpy(self.data1.iloc[idx:idx+self.t1].values)
        x=torch.stack([x0,x1], dim=1)
        y0=torch.from_numpy(self.data0.iloc[idx+1:idx+self.t1+1].values)
        y1=torch.from_numpy(self.data1.iloc[idx+1:idx+self.t1+1].values)
        y=torch.stack([y0,y1], dim=1)
        return (x,y)
