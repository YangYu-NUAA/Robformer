import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import random
from time import time
from typing import Union

"""
this is a polynomial module for trend prediction
"""



def linear_space(backcast_length, forecast_length, is_forecast=True):
    horizon = forecast_length if is_forecast else backcast_length
    return np.arange(0, horizon) / horizon


class Block(nn.Module):

    def __init__(self, thetas_dim, seq_len, pred_len):
        super(Block, self).__init__()


        self.units = 512   #hidden_layer_units
        self.thetas_dim = thetas_dim
        self.backcast_length = seq_len
        self.forecast_length = pred_len
        self.share_thetas = False
        self.fc1 = nn.Linear(self.backcast_length, self.units)
        self.fc2 = nn.Linear(self.units, self.units)
        self.fc3 = nn.Linear(self.units, self.units)
        self.fc4 = nn.Linear(self.units, self.units)
        self.fc5 = nn.Linear(self.units, self.units)
        self.dropout = nn.Dropout(0.1)
        self.device = "cuda"
        self.backcast_linspace = linear_space(self.backcast_length, self.forecast_length, is_forecast=False)
        self.forecast_linspace = linear_space(self.backcast_length, self.forecast_length, is_forecast=True)
        if self.share_thetas:
            self.theta_f_fc = self.theta_b_fc = nn.Linear(self.units, self.thetas_dim, bias=False)
        else:
            self.theta_b_fc = nn.Linear(self.units, self.thetas_dim, bias=False)
            self.theta_f_fc = nn.Linear(self.units, self.thetas_dim, bias=False)

    def forward(self, x):

        x = F.relu(self.fc1(x.to(self.device)))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = F.relu(self.fc5(x))
        x = self.dropout(x)
        return x

    def __str__(self):
        block_type = type(self).__name__
        return f'{block_type}(units={self.units}, thetas_dim={self.thetas_dim}, ' \
               f'backcast_length={self.backcast_length}, forecast_length={self.forecast_length}, ' \
               f'share_thetas={self.share_thetas}) at @{id(self)}'




class GenericBlock(Block):

    def __init__(self, thetas_dim, seq_len, pred_len):
        super(GenericBlock, self).__init__(thetas_dim, seq_len, pred_len)

        self.backcast_fc = nn.Linear(thetas_dim, seq_len)
        self.forecast_fc = nn.Linear(thetas_dim, pred_len)

    def forward(self, x):
        # no constraint for generic arch.
        x = super(GenericBlock, self).forward(x)  #jichegn wan zhijie forward

        theta_b = self.theta_b_fc(x)
        theta_f = self.theta_f_fc(x)

        backcast = self.backcast_fc(theta_b)  # generic. 3.3.
        forecast = self.forecast_fc(theta_f)  # generic. 3.3.

        return backcast, forecast

class RobTF(nn.Module):

    def __init__(self, thetas_dim1, thetas_dim2, seq_len, pred_len):
        super(RobTF, self).__init__()
        """
        stack 1
        """
        self.genericblock1 = GenericBlock(thetas_dim=thetas_dim1, seq_len= seq_len, pred_len= pred_len)
        self.genericblock2 = GenericBlock(thetas_dim=thetas_dim1, seq_len= seq_len, pred_len= pred_len)
        self.genericblock3 = GenericBlock(thetas_dim=thetas_dim1, seq_len= seq_len, pred_len= pred_len)
        """
               stack 2
        """
        self.genericblock4 = GenericBlock(thetas_dim=thetas_dim2, seq_len= seq_len, pred_len= pred_len)
        self.genericblock5 = GenericBlock(thetas_dim=thetas_dim2, seq_len= seq_len, pred_len= pred_len)
        self.genericblock6 = GenericBlock(thetas_dim=thetas_dim2, seq_len= seq_len, pred_len= pred_len)

        """
           stack 3
        """
        self.genericblock7 = GenericBlock(thetas_dim=thetas_dim2, seq_len=seq_len, pred_len=pred_len)
        self.genericblock8 = GenericBlock(thetas_dim=thetas_dim2, seq_len=seq_len, pred_len=pred_len)
        self.genericblock9 = GenericBlock(thetas_dim=thetas_dim2, seq_len=seq_len, pred_len=pred_len)
        """
           stack 4
        """
        self.genericblock10 = GenericBlock(thetas_dim=thetas_dim1, seq_len=seq_len, pred_len=pred_len)
        self.genericblock11 = GenericBlock(thetas_dim=thetas_dim1, seq_len=seq_len, pred_len=pred_len)
        self.genericblock12 = GenericBlock(thetas_dim=thetas_dim1, seq_len=seq_len, pred_len=pred_len)
        """
           stack 5
        """
        self.genericblock13 = GenericBlock(thetas_dim=thetas_dim1, seq_len=seq_len, pred_len=pred_len)
        self.genericblock14 = GenericBlock(thetas_dim=thetas_dim1, seq_len=seq_len, pred_len=pred_len)
        self.genericblock15 = GenericBlock(thetas_dim=thetas_dim1, seq_len=seq_len, pred_len=pred_len)
        """
           stack 6
        """
        self.genericblock16 = GenericBlock(thetas_dim=thetas_dim1, seq_len=seq_len, pred_len=pred_len)
        self.genericblock17 = GenericBlock(thetas_dim=thetas_dim1, seq_len=seq_len, pred_len=pred_len)
        self.genericblock18 = GenericBlock(thetas_dim=thetas_dim1, seq_len=seq_len, pred_len=pred_len)

    def forward(self,x):
        #[8, 192, 7]
        y = x
        x_back1, x1 = self.genericblock1(x)
        x = y - x_back1
        y = x
        x_back2, x2 = self.genericblock2(x)
        x = y - x_back2
        y = x
        x_back3, x3 = self.genericblock3(x)

        forecast1 = x1 + x2 + x3

        res = y - x_back3

        y = res
        x_back4, x4 = self.genericblock4(res)
        x = y - x_back4
        y = x
        x_back5, x5 = self.genericblock5(x)
        x = y - x_back5
        y = x
        x_back6, x6 = self.genericblock6(x)

        forecast2 = x4 + x5 + x6

        res2 = y - x_back6
        y = res2
        x_back7, x7 = self.genericblock7(res2)
        x = y - x_back7
        y = x
        x_back8, x8 = self.genericblock8(x)
        x = y - x_back8
        y = x
        x_back9, x9 = self.genericblock9(x)

        forecast3 = x7 + x8 + x9

        res3 = y - x_back9
        y = res3
        x_back10, x10 = self.genericblock7(res3)
        x = y - x_back10
        y = x
        x_back11, x11 = self.genericblock8(x)
        x = y - x_back11
        y = x
        x_back12, x12 = self.genericblock9(x)

        forecast4 = x10 + x11 + x12

        res4 = y - x_back12
        y = res4
        x_back13, x13 = self.genericblock7(res4)
        x = y - x_back13
        y = x
        x_back14, x14 = self.genericblock8(x)
        x = y - x_back14
        y = x
        x_back15, x15 = self.genericblock9(x)

        forecast5 = x13 + x14 + x15

        res5 = y - x_back15
        y = res5
        x_back16, x16 = self.genericblock7(res5)
        x = y - x_back16
        y = x
        x_back17, x17 = self.genericblock8(x)
        x = y - x_back17
        y = x
        x_back18, x18 = self.genericblock9(x)

        forecast6 = x16 + x17 + x18

        forecast = forecast1 + forecast2 + forecast3 + forecast4 + forecast5 + forecast6
        forecast = forecast.transpose(1, 2)
        #print("forecast", forecast.shape)

        return forecast


