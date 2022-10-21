import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Robformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, seasonality_extraction, series_decomp_2diff, season_adjust
import math
import numpy as np
from layers.Trend_Forecast import RobTF

class Model(nn.Module):
    """
		Robformer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        self.channels = configs.enc_in
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend1 = nn.Linear(self.seq_len, 128)
        self.Linear_Trend2= nn.Linear(128, self.pred_len)
        #self.Linear_Trend.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        #self.Linear_Trend = nn.ModuleList()

        #for i in range(self.channels):
        #    self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        self.layer = torch.nn.Linear(1, 2)

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)     
        self.decomp_2diff = series_decomp_2diff(kernel_size)
        self.seasonal_extraction = seasonality_extraction(season_len = 25, K= 2, H = 1)
        #pseu label
        self.alpha = nn.Parameter(torch.randn(1))
        #self.adjust = season_adjust(season_len= 25)
        self.dropout = nn.Dropout(0.1)
        #self.period = period_detection(factor=1)
        self.gelu = nn.GELU()

        self.robTF = RobTF(thetas_dim1 = 512, thetas_dim2 = 512, seq_len = self.seq_len, pred_len = self.pred_len)
       
        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),  #True zhide shi shifou mask
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),  #huake diandongxi
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,

                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).cuda()
        
        seasonal_init, trend_init = self.decomp_2diff(x_enc)
        #seasonal_init, trend_init = self.decomp(x_enc)

        season, trend = seasonal_init, trend_init

        # pseudo-label
        
        mean_pro = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
   
        seasonal_init = self.seasonal_extraction(seasonal_init)


        predict_init = trend_init.permute(0, 2, 1)

        trend_output = self.robTF(predict_init)
 
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        mean = torch.mean(trend_output, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)

        trend_init_our = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean_pro], dim=1)

        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)  
       
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        #seasonal_part = self.seasonal_extraction(seasonal_part)

        trend_output = torch.cat([trend_init[:, -self.label_len:, :], trend_output], dim=1)
        #dec_out = trend_part + seasonal_part
        dec_out = trend_output + seasonal_part
     
        # output the first decomposition result

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns, seasonal_part[:, -self.pred_len:, :], trend_part[:, -self.pred_len:, :]
        else:
            return dec_out[:, -self.pred_len:, :], seasonal_part[:, -self.pred_len:, :], trend_part[:, -self.pred_len:, :] # [B, L, D]
