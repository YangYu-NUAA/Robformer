import matplotlib as mpl
#print(matplotlib.get_backend())


#mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn, einsum
#rearrange : zhuanzhi
#repeat : ():zheshi xiangcheng , duochu de c shi yan yige xinde weidu fuzhi jici
import numpy as np


a = np.array([1,2,-3,-4])
b = +a
print(b)

dataInit = pd.DataFrame([1,2,3,4,5,6])
plt.plot(dataInit)
plt.show()