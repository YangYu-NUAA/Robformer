import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import toeplitz

from cvxopt import blas, lapack, solvers
import math
from cvxopt import matrix, spdiag, mul, div, sparse
class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
 
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
       
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)  #1,12,1
        x = torch.cat([front, x, end], dim=1)
       
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
      
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
       
        moving_mean = self.moving_avg(x)   #trend
       
        res = x - moving_mean   #season
        return res, moving_mean


class series_decomp_2diff(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp_2diff, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)
        self.layer = torch.nn.Linear(1, 2) #1 -> 2 difference


    def forward(self, x):
        
        trend = []

        moving_mean = self.moving_avg(x)   
        moving_mean2 = self.moving_avg(moving_mean) 
        trend.append(moving_mean.unsqueeze(-1))
        trend.append(moving_mean2.unsqueeze(-1))
       
        trend = torch.cat(trend, dim=-1)
       
        trend = torch.sum(trend * nn.Softmax(-1)(self.layer(x.unsqueeze(-1))), dim=-1)

        res = x - trend   #season
        return res, trend
      


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean=[]
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean=torch.cat(moving_mean,dim=-1)
        moving_mean = torch.sum(moving_mean*nn.Softmax(-1)(self.layer(x.unsqueeze(-1))),dim=-1)
        res = x - moving_mean
        return res, moving_mean


class season_adjust(nn.Module):
    """
    adjust part
    """
    def __init__(self, season_len):
        super(season_adjust, self).__init__()
        self.season_len = season_len  


    def forward(self, x, y): 
        B,L,E = y.shape
        num_season = int(L/self.season_len)
        
        adjust = torch.mean(y[:,:self.season_len*num_season,:],dim=1, keepdim=True)
       

        trend = x + adjust
        season = y - adjust

        return season, trend


class seasonality_extraction(nn.Module):
    """
    Robust seasonal decomposition block
    """
    def __init__(self, season_len, K, H):
        super(seasonality_extraction, self).__init__()
        self.ds1 = 50.
        self.ds2 = 1.
        #self.ds1 = nn.parameter(torch.randn())
        self.season_len = 25
        self.K = 3
        self.H = 1
        self.layer = torch.nn.Linear(1, 2)
        self.layer2 = torch.nn.Linear(1, 3)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_neighbor_idx(self, total_len, target_idx, H=1):
        '''
        Let i = target_idx.
        Then, return i-H, ..., i, ..., i+H, (i+H+1)
        '''
        return [np.max([0, target_idx - H]), np.min([total_len, target_idx + H + 1])]

    def get_neighbor_range(self, total_len, target_idx, H=1):
        start_idx, end_idx = self.get_neighbor_idx(total_len, target_idx, H)
        return np.arange(start_idx, end_idx)

    def get_season_idx(self, total_len, target_idx, T=25):
       
        K = self.K
        H = self.H
        num_season = np.min([K, int(target_idx / T)])

        if target_idx < T:
            key_idxs = target_idx + np.arange(0, num_season + 1) * (-1 * T)
        else:
            key_idxs = target_idx + np.arange(1, num_season + 1) * (-1 * T)

        idxs = list(map(lambda idx: self.get_neighbor_range(total_len, idx, H), key_idxs))
        season_idxs = []
        for item in idxs:
            season_idxs += list(item)
        season_idxs = np.array(season_idxs)
     
        return season_idxs

    def bilateral_filter(self, j, t, y_j, y_t, delta1=1.0, delta2=1.0):
       
        y_j = y_j.shape[1]
        y_t = y_t.shape[1]

        idx1 = -1.0 * (math.fabs(j - t) ** 2.0) / (2.0 * delta1 ** 2)
        idx2 = -1.0 * (math.fabs(y_j - y_t) ** 2.0) / (2.0 * delta2 ** 2)
        weight = (math.exp(idx1) * math.exp(idx2))
        
        return weight

    def get_season_value(self,idx,length,x):
        
        season_len = self.season_len
        B,L,E = x.shape

        idxs = self.get_season_idx(length, idx)
       
        if idxs.size == 0:
            return x[idx]

        weight_sample = x[:,idxs,:] 
        
        weights = torch.tensor(list(map(lambda j: self.bilateral_filter(j, idx, x[:,j,:], x[:,idx,:], self.ds1, self.ds2), idxs))).to("cuda")
       
        result = (torch.einsum("l,ble->ble",weights,weight_sample)) 
        
        result = torch.sum(result,dim= 1, keepdim= True)
       
        season_value = result/torch.sum(weights)  
      
        return season_value


    def forward(self,x):
        B, length, embedding = x.shape
        
        value_mean = []
       
        for i in range(length):
            weight_id = []
            #if i > self.season_len and i < 2*self.season_len:
            if i > self.season_len :
                
                weight_id.append(x[:,i,:].unsqueeze(dim=1).unsqueeze(-1))
                weight_id.append(x[:,i-self.season_len,:].unsqueeze(dim=1).unsqueeze(-1))   #[]
              

                weight_id_tensor = torch.cat(weight_id, dim=-1)  #[8,1,512,2]
               
                id_value = torch.sum(
                    weight_id_tensor * nn.Softmax(-1)(self.layer(x.unsqueeze(-1))), dim=-1)
              
                value_mean.append(id_value)

            elif i > 2*self.season_len:
                weight_id.append(x[:, i, :].unsqueeze(dim=1).unsqueeze(-1))
                weight_id.append(x[:, i - self.season_len, :].unsqueeze(dim=1).unsqueeze(-1))
             
                weight_id.append(x[:, i - 2 * self.season_len, :].unsqueeze(dim=1).unsqueeze(-1))  

                weight_id_tensor = torch.cat(weight_id, dim=-1)

                id_value = torch.sum(weight_id_tensor * nn.Softmax(-1)(self.layer2(x.unsqueeze(-1))), dim= -1)
                value_mean.append(id_value)


            else:
                value_mean.append(x[:,i,:].unsqueeze(dim=1))

        season_init = torch.cat(value_mean,dim=1)

        return season_init



class robust_decomp_enc(nn.Module):
    """
    Robust series decomposition block
    """
    def __init__(self, reg1, reg2, max_iter = 100):
        super(robust_decomp_enc,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_len = 96  #96->25

        self.max_iter = max_iter
        # self.opt = torch.optim.Adam([self.dt], 0.001)
        self.loss = torch.nn.L1Loss(reduction='sum').to(self.device)

        self.reg1 = reg1
        self.reg2 = reg2
        #self.D = D

    def get_toeplitz(self, shape, entry):
        h, w = shape
        num_entry = len(entry)
        assert np.ndim(entry) < 2
        if num_entry < 1:
            return np.zeros(shape)
        row = np.concatenate([entry[:1], np.zeros(h - 1)])
        col = np.concatenate([np.array(entry), np.zeros(w - num_entry)])
        return toeplitz(row, col)

    def l1(self, P, q):

      
        m, n = P.size  
        c = matrix(n * [0.0] + m * [1.0]) 
        h = matrix([q, -q])  
       
        def Fi(x, y, alpha=1.0, beta=0.0, trans='N'):
            if trans == 'N':
                u = P * x[:n, :]  
                y[:m] = alpha * (u - x[n:]) + beta * y[:m]
                y[m:] = alpha * (-u - x[n:]) + beta * y[m:]
            else:
                y[:n] = alpha * P.T * (x[:m] - x[m:]) + beta * y[:n]
                y[n:] = -alpha * (x[:m] + x[m:]) + beta * y[n:]

        def Fkkt(W):
            d1, d2 = W['d'][:m], W['d'][m:]
            D = 4 * (d1 ** 2 + d2 ** 2) ** -1
            A = P.T * spdiag(D) * P
            lapack.potrf(A)

            def f(x, y, z):
                x[:n] += P.T * (mul(div(d2 ** 2 - d1 ** 2, d1 ** 2 + d2 ** 2), x[n:])
                                + mul(.5 * D, z[:m] - z[m:]))
                lapack.potrs(A, x)

                u = P * x[:n]
                x[n:] = div(x[n:] - div(z[:m], d1 ** 2) - div(z[m:], d2 ** 2) +
                            mul(d1 ** -2 - d2 ** -2, u), d1 ** -2 + d2 ** -2)

                z[:m] = div(u - x[n:] - z[:m], d1)
                z[m:] = div(-u - x[n:] - z[m:], d2)

            return f

        uls = +q
       
        lapack.gels(+P, uls)  
        rls = P * uls[:n] - q  

        x0 = matrix([uls[:n], 1.1 * abs(rls)])  #
       
        s0 = +h
     
        Fi(x0, s0, alpha=-1, beta=1)
      

        if max(abs(rls)) > 1e-10:
            w = .9 / max(abs(rls)) * rls
        else:
            w = matrix(0.0, (m, 1))
        z0 = matrix([.5 * (1 + w), .5 * (1 - w)])
     
        dims = {'l': 2 * m, 'q': [], 's': []}
       
        sol = solvers.conelp(c, Fi, h, dims, kktsolver=Fkkt)

        return sol['x'][:n]

    def get_relative_trends(self, delta_trends):
        init_value = np.array([0])
        idxs = np.arange(len(delta_trends))
        relative_trends = np.array(list(map(lambda idx: np.sum(delta_trends[:idx]), idxs)))
        relative_trends = np.concatenate([init_value, relative_trends])
        return relative_trends


    def forward(self,x):
 

        B, length, embedding = x.shape
        # split
        x_batch = []
        x_batch = torch.split(x,[1,1,1,1,1,1,1,1],dim=0)


        season_len = 25 
        M = self.get_toeplitz([length - season_len, length - 1],
                              np.ones([season_len]))
        D = self.get_toeplitz([length - 2, length - 1], np.array([1, -1]))
  
        delta_trends = []
        relative_trends = []
        detrend_samples = []
        for i in x_batch:
         
            i = torch.squeeze(i, dim=0).cpu()
          
            season_diff = i[season_len:] - i[:-season_len]
            assert len(season_diff) == (length - season_len)
            
            q = np.concatenate([season_diff, np.zeros([length * 2 - 3, embedding])])  # q (261, 512)
            q = np.reshape(q, [len(q), embedding])
            q = matrix(q)

            P = np.concatenate([M, self.reg1 * np.eye(length - 1), self.reg2 * D], axis=0)
            P = matrix(P)
           
            delta_trends = self.l1(P,q)
            relative_trend = self.get_relative_trends(delta_trends) 
           
            relative_trend = torch.from_numpy(relative_trend).float()
            relative_trend = torch.unsqueeze(relative_trend,1)
          
            detrend_sample = i - relative_trend
            relative_trend = torch.unsqueeze(relative_trend,0)
            detrend_sample = torch.unsqueeze(detrend_sample,0)


            relative_trends.append(relative_trend)
            detrend_samples.append(detrend_sample)

        relative_trends = torch.cat(relative_trends, dim= 0).to(self.device)
        detrend_sample = torch.cat(detrend_samples, dim= 0 ).to(self.device)


        return detrend_sample, relative_trends


class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
      
        self.decomp3 = series_decomp(25)
        self.decomp4 = series_decomp(25)

        self.adjust1 = season_adjust(season_len=25)
        self.adjust2 = season_adjust(season_len=25)

        self.decomp1 = series_decomp_2diff(25)
        self.decomp2 = series_decomp_2diff(47)

        self.season1 = seasonality_extraction(season_len = 25, K = 3 , H = 1)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
     
        x = x + self.dropout(new_x)
        x, trend = self.decomp3(x)


        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp4(x + y)
   
        return res, attn


class Encoder(nn.Module):
    """
    Robformer encoder
    """
    #EncoderLayer norm_layer
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    Robformer decoder layer
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)

        self.adjust1 = season_adjust(season_len=25)
        self.adjust2 = season_adjust(season_len=25)

        self.decomp4 = series_decomp(moving_avg)
        self.decomp5 = series_decomp(moving_avg)
        self.decomp6 = series_decomp(moving_avg)

        self.decomp1 = series_decomp_2diff(moving_avg)
        self.decomp2 = series_decomp_2diff(moving_avg)
        self.decomp3 = series_decomp_2diff(moving_avg)

        self.season1 = seasonality_extraction(season_len = 25, K= 2, H = 1)
        self.season2 = seasonality_extraction(season_len = 25, K= 2, H = 1)

        self.trend_attention = nn.Linear(1,3)

        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,   #d_model = 512 c_out = 7
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.c_out = c_out
        self.trend_linear = nn.Linear(384, 384)
        self.trend_linear2 = nn.Linear(384, 384)
 
   

    def forward(self, x, cross, x_mask=None, cross_mask=None):

        res = x

        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])

        x, trend1 = self.decomp4(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,   
            attn_mask=cross_mask
        )[0])
        x, trend2 = self.decomp5(x)


        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp6(x + y)

        trend_mean = []

   

        residual_trend = trend1 + trend2 + trend3 
    
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
   
        return x, residual_trend


class Decoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection
        self.trend_attention = nn.Linear(1, 2)
        self.alpha = nn.Parameter(torch.randn(1))

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend = None):  #dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                
        for layer in self.layers:
    
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
      

            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, trend
