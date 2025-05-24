import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# ChaMTeC: CHAnnel Mixing and TEmporal Convolution Network for Time-Series Anomaly Detection
# https://www.mdpi.com/2076-3417/15/10/5623


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        return x + self.pe[:x.size(1)]
    
class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.temporal_pos_encoder = TemporalPositionalEncoding(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)  # [Batch, Variate, Time] -> [Batch, Time, Variate]
        
        if x_mark is None:
            x = self.value_embedding(x)  # [Batch, Time, Variate] -> [Batch, Time, d_model]
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        
        # Add temporal positional encoding
        x = self.temporal_pos_encoder(x)  # [Batch, Time, d_model]
        
        return self.dropout(x)
    
class TemporalAttention(nn.Module):
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"
        
        # Linear layers for Q, K, V projections
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_dropout = nn.Dropout(dropout)
        self.scale = 1.0 / (self.head_dim ** 0.5)
        
    def forward(self, x):
        # x shape: (B, F, D)
        B, F, D = x.shape
        
        # Project to query, key, value - we operate on the temporal dimension (D)
        q = self.query(x).view(B, F, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, H, F, head_dim)
        k = self.key(x).view(B, F, self.n_heads, self.head_dim).permute(0, 2, 1, 3)    # (B, H, F, head_dim)
        v = self.value(x).view(B, F, self.n_heads, self.head_dim).permute(0, 2, 1, 3)   # (B, H, F, head_dim)
        
        # Compute attention scores (across time for each feature)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, F, F)
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).permute(0, 2, 1, 3).reshape(B, F, D)  # (B, F, D)
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        return x

    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1, activation="relu", n_heads=8):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        
        # Temporal attention (now properly handling BxFxD input)
        self.temporal_attn = TemporalAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Conv FFN (modified to handle the dimensionality)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        # x shape: (B, F, D)
        
        # Temporal attention
        x = x + self.temporal_attn(self.norm1(x))
        
        # Conv FFN
        y = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.permute(0, 2, 1))))  # Conv1D expects (B, D, F)
        y = self.dropout(self.conv2(y).permute(0, 2, 1))  # Back to (B, F, D)
        
        return x + y


class Encoder(nn.Module):
    def __init__(self, feature_dim, d_model, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.channel = nn.Sequential(
                nn.Linear(feature_dim, d_model),
                nn.ReLU(),
                nn.Linear(d_model, feature_dim),
                nn.Dropout(0.1)
            )
        self.conv_layers = nn.ModuleList(conv_layers)
        self.norm = norm_layer

    def forward(self, x):
        # x [B, L, D]
        x = self.channel(x.permute(0,2,1)).permute(0,2,1)
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
       
        if self.norm is not None:
            x = self.norm(x)

        return x


class MSEFeedbackRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(MSEFeedbackRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)

        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # Forward pass through RNN
        out, _ = self.rnn(x, h0)
        out = self.fc(out)

        return out


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder
        self.encoder = Encoder( configs.enc_in, configs.d_model,
            [
                EncoderLayer(
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)

            rnn_hidden_size = configs.enc_in//2
            rnn_output_size = configs.enc_in
            self.rnn = MSEFeedbackRNN(input_size= configs.enc_in, hidden_size=rnn_hidden_size, output_size=rnn_output_size, num_layers=8)  
            
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)




    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out = self.encoder(enc_out)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]


        mse_error = ((x_enc - dec_out) ** 2)#.mean(dim=1, keepdim=True)
        # combined_input = torch.cat((output, mse_error), dim=1)
        combined_input = dec_out + mse_error
        
        rnn_output = self.rnn(combined_input)
        dec_out = dec_out +  rnn_output.squeeze(1)


        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))

        return dec_out
