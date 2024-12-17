import yfinance as yf
import talib as ta
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.fft import fft, ifft, fftfreq
import math
from data import download_and_prepare_data

class StockDataStructure:
    def __init__(self, df):
        self.df = df
        self.time_index = df.index
        self.features = df.columns
        
        # 添加更强的标准化
        self.scaler = StandardScaler()
        normalized_data = self.scaler.fit_transform(df.values)
        
        # 添加异常值处理
        normalized_data = np.clip(normalized_data, -3, 3)  # 限制在3个标准差内
        
        # 转换为tensor
        self.tensor_data = torch.tensor(normalized_data, dtype=torch.float32)
        
    def get_time_window(self, t, window_size=10):
        start = max(0, t - window_size)
        end = min(len(self.time_index), t + window_size + 1)
        window_data = self.tensor_data[start:end]
        return window_data.unsqueeze(0)  # Add batch dimension [1, window_size, features]

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # 确保d_model是偶数，如果是奇数就加1
        d_model_adjusted = d_model + (d_model % 2)
        
        pe = torch.zeros(max_len, d_model_adjusted)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算除数项
        div_term = torch.exp(
            torch.arange(0, d_model_adjusted, 2).float() * 
            (-math.log(10000.0) / d_model_adjusted)
        )
        
        # 计算位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 如果原始维度是奇数，去掉多余的一列
        if d_model % 2 == 1:
            pe = pe[:, :d_model]
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0)]

class ImprovedAttentionMechanism(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim=64, n_heads=4, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        
        # Multi-head attention layers
        self.W_query = torch.nn.Parameter(torch.randn(n_heads, feature_dim, self.head_dim))
        self.W_key = torch.nn.Parameter(torch.randn(n_heads, feature_dim, self.head_dim))
        self.W_value = torch.nn.Parameter(torch.randn(n_heads, feature_dim, self.head_dim))
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(feature_dim)
        
        # Layer normalization
        self.layer_norm = torch.nn.LayerNorm(feature_dim)
        
        # Dropout
        self.dropout = torch.nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for param in [self.W_query, self.W_key, self.W_value]:
            torch.nn.init.xavier_uniform_(param)
    
    def compute_relative_position_bias(self, window_size):
        # 调整位置编码的衰减率
        gamma = 0.5  # 从1.0降低到0.5,使远程依赖保持更多权重
        positions = torch.arange(window_size).unsqueeze(1) - torch.arange(window_size).unsqueeze(0)
        positions = positions.float()
        # 添加可学习的缩放因子
        self.scale = torch.nn.Parameter(torch.ones(1))
        return self.scale * torch.exp(-gamma * positions.pow(2))
    
    def compute_temporal_attention(self, data_window, temperature=1.0):
        batch_size, window_size, features = data_window.shape
        
        # Layer normalization
        data_window = self.layer_norm(data_window)
        
        # Add positional encoding
        pos_encoded = self.pos_encoding(data_window)
        
        attention_heads = []
        attention_weights_heads = []
        
        for head in range(self.n_heads):
            # Compute Q, K, V
            Q = torch.matmul(pos_encoded, self.W_query[head])
            K = torch.matmul(pos_encoded, self.W_key[head])
            V = torch.matmul(pos_encoded, self.W_value[head])
            
            # Compute attention scores
            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # Add relative position bias
            relative_bias = self.compute_relative_position_bias(window_size).unsqueeze(0)
            attention_scores = attention_scores + relative_bias
            
            # Apply temperature
            attention_scores = attention_scores / temperature
            
            # Dropout
            attention_scores = self.dropout(attention_scores)
            
            # Softmax
            attention_weights = F.softmax(attention_scores, dim=-1)
            
            # Compute weighted sum
            head_output = torch.matmul(attention_weights, V)
            attention_heads.append(head_output)
            attention_weights_heads.append(attention_weights)
        
        # Combine multiple heads
        multi_head_output = torch.cat(attention_heads, dim=-1)
        multi_head_weights = torch.stack(attention_weights_heads, dim=1)
        
        return multi_head_output, multi_head_weights.mean(dim=1)

def visualize_attention(attention_weights, temperature, ax=None):
    """Visualize attention weights"""
    if ax is None:
        ax = plt.gca()
    
    im = ax.imshow(attention_weights, cmap='viridis')
    ax.set_title(f'Attention Weights (T={temperature})')
    ax.set_xlabel('Time steps')
    ax.set_ylabel('Time steps')
    plt.colorbar(im, ax=ax)

class WeightedLaplacian(torch.nn.Module):
    def __init__(self, attention_mechanism):
        super().__init__()
        self.attention = attention_mechanism
        
    def compute_temporal_gradient(self, data_window):
        """计算时间维度的梯度"""
        # 使用中心差分
        gradient = torch.zeros_like(data_window)
        gradient[:, 1:-1] = (data_window[:, 2:] - data_window[:, :-2]) / 2
        # 边界使用前向/后向差分
        gradient[:, 0] = data_window[:, 1] - data_window[:, 0]
        gradient[:, -1] = data_window[:, -1] - data_window[:, -2]
        return gradient
    
    def compute_feature_gradient(self, data_window):
        """计算特征维度的梯度"""
        # 转置数据以计算特征维度的梯度
        data_t = data_window.transpose(1, 2)
        gradient = torch.zeros_like(data_t)
        gradient[:, 1:-1] = (data_t[:, 2:] - data_t[:, :-2]) / 2
        gradient[:, 0] = data_t[:, 1] - data_t[:, 0]
        gradient[:, -1] = data_t[:, -1] - data_t[:, -2]
        return gradient.transpose(1, 2)
    
    def compute_weighted_gradient(self, data_window, temperature=1.0):
        """计算带权重的梯度"""
        # 获取注意力权重
        _, attention_weights = self.attention.compute_temporal_attention(
            data_window, 
            temperature=temperature
        )
        
        # 计算时间和特征梯度
        temporal_grad = self.compute_temporal_gradient(data_window)
        feature_grad = self.compute_feature_gradient(data_window)
        
        # 使用注意力权重加权时间梯度
        weighted_temporal_grad = torch.matmul(attention_weights, temporal_grad)
        
        # 特征梯度使用自适应权重
        feature_importance = torch.sigmoid(feature_grad.abs().mean(dim=1, keepdim=True))
        weighted_feature_grad = feature_grad * feature_importance
        
        return weighted_temporal_grad, weighted_feature_grad
    
    def compute_divergence(self, temporal_grad, feature_grad):
        """计算散度"""
        # 添加数值稳定性处理
        epsilon = 1e-6
        
        # 添加梯度裁剪
        temporal_grad = torch.clamp(temporal_grad, -10, 10)
        feature_grad = torch.clamp(feature_grad, -10, 10)
        
        # 计算时间维度散度
        temporal_div = torch.zeros_like(temporal_grad)
        temporal_div[:, 1:-1] = (temporal_grad[:, 2:] - temporal_grad[:, :-2]) / (2 + epsilon)
        temporal_div[:, 0] = temporal_grad[:, 1] - temporal_grad[:, 0]
        temporal_div[:, -1] = temporal_grad[:, -1] - temporal_grad[:, -2]
        
        # 添加平滑处理
        kernel_size = 3
        smoothing = torch.nn.AvgPool1d(kernel_size, stride=1, padding=kernel_size//2)
        temporal_div = smoothing(temporal_div)
        
        # 特征维度的散度
        feature_div = torch.zeros_like(feature_grad)
        feature_t = feature_grad.transpose(1, 2)
        feature_div_t = torch.zeros_like(feature_t)
        feature_div_t[:, 1:-1] = (feature_t[:, 2:] - feature_t[:, :-2]) / 2
        feature_div_t[:, 0] = feature_t[:, 1] - feature_t[:, 0]
        feature_div_t[:, -1] = feature_t[:, -1] - feature_t[:, -2]
        feature_div = feature_div_t.transpose(1, 2)
        
        return temporal_div + feature_div
    
    def forward(self, data_window, temperature=1.0):
        """计算加权拉普拉斯算子"""
        # 计算加权梯度
        weighted_temporal_grad, weighted_feature_grad = self.compute_weighted_gradient(
            data_window, 
            temperature
        )
        
        # 计算散度
        laplacian = self.compute_divergence(weighted_temporal_grad, weighted_feature_grad)
        
        return laplacian

def visualize_laplacian(laplacian_output, feature_names, time_steps, ax=None):
    """可视化拉普拉斯算子的输出"""
    if ax is None:
        ax = plt.gca()
    
    im = ax.imshow(laplacian_output[0].detach().numpy(), cmap='RdBu', aspect='auto')
    plt.colorbar(im, ax=ax)
    
    # 添加特征标签
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=8)
    
    # 添加时间标签
    ax.set_xlabel('Time steps')
    ax.set_title('Weighted Laplacian Output')
    
    return im

def main():
    # Download and prepare data
    data = download_and_prepare_data('AAPL', '2019-01-01', '2024-01-01')
    print("Data shape:", data.shape)
    
    # 创建模型
    stock_data = StockDataStructure(data)
    attention_mechanism = ImprovedAttentionMechanism(len(data.columns))
    laplacian_operator = WeightedLaplacian(attention_mechanism)
    
    # 获取数据窗口
    window_data = stock_data.get_time_window(100, window_size=40)
    
    # 计算拉普拉斯算子
    laplacian_output = laplacian_operator(window_data)
    
    # 创建可视化
    plt.figure(figsize=(15, 10))
    
    # 显示注意力权重
    plt.subplot(2, 1, 1)
    _, attention_weights = attention_mechanism.compute_temporal_attention(window_data)
    plt.imshow(attention_weights[0].detach().numpy(), cmap='viridis')
    plt.colorbar()
    plt.title('Attention Weights')
    plt.xlabel('Time steps')
    plt.ylabel('Time steps')
    
    # 显示拉普拉斯算子输出
    plt.subplot(2, 1, 2)
    visualize_laplacian(laplacian_output, data.columns, 40)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()