# fusion_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import sys
import os
import numpy as np
import pandas as pd
import pickle
from CNN import ResidualFinancialBlock, EnhancedStockDataset
from model import StockLSTMCell, EventProcessor, EnhancedTMDO, prepare_feature_groups
from train import EnhancedCombinedLoss, train_enhanced_model, generate_event_data#, combine_stock_data
from data import load_data_from_csv, download_and_prepare_data
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def combine_stock_data(symbols, start_date, end_date):
    """
    下载多只股票的数据并拼接
    
    Args:
        symbols (list): 股票代码列表
        start_date (str): 开始日期
        end_date (str): 结束日期
    
    Returns:
        pd.DataFrame: 拼接后的数据
    """
    all_data = []
    
    for symbol in tqdm(symbols):
        # 获取单个股票数据
        data = download_and_prepare_data(symbol, start_date, end_date)

        if not data.empty:
            # 将数据添加到列表中
            all_data.append(data)
    
    # 直接拼接所有数据
    combined_data = pd.concat(all_data, axis=0, ignore_index=True)
    
    return combined_data

class FusionStockDataset(Dataset):
    """
    融合数据集类,同时支持CNN和LSTM特性
    """
    def __init__(self, data, events, sequence_length=250, prediction_horizon=5):
        """
        Args:
            data: DataFrame containing stock data
            events: Event data matrix
            sequence_length: Number of time steps to look back
            prediction_horizon: Number of days to predict ahead
        """
        # 定义特征顺序
        self.feature_order = [
            'Close', 'Open', 'High', 'Low',  # 价格指标
            'MA5', 'MA20',                   # 均线指标
            'MACD', 'MACD_Signal', 'MACD_Hist',  # MACD族
            'RSI', 'Upper', 'Middle', 'Lower',    # RSI和布林带
            'CRSI', 'Kalman_Price', 'Kalman_Trend',  # 高级指标
            'FFT_21', 'FFT_63',              # FFT指标
            'Volume', 'Volume_MA5'           # 成交量指标
        ]
        
        self.data = data
        self.events = events
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # 组织2D特征数据
        self.feature_data = self._organize_features()
        
        # 计算时间距离矩阵
        self.time_distances = self._compute_time_distances()
        
    def _organize_features(self):
        """组织特征为2D张量格式"""
        # 检查特征完整性
        for feature in self.feature_order:
            if feature not in self.data.columns:
                raise ValueError(f"Missing feature: {feature}")
        
        # 提取并堆叠特征
        feature_data = []
        for feature in self.feature_order:
            values = self.data[feature].values
            # 标准化数据
            mean = np.mean(values)
            std = np.std(values)
            normalized = (values - mean) / (std + 1e-8)  # 避免除零
            feature_data.append(normalized)
            
        return torch.FloatTensor(np.stack(feature_data, axis=0))
    
    def _compute_time_distances(self):
        """计算事件时间距离"""
        distances = np.zeros((len(self.data), 1))
        last_event_idx = -1
        
        for i in range(len(self.data)):
            if self.events[i].any():
                last_event_idx = i
            distances[i] = i - last_event_idx if last_event_idx != -1 else 100
            
        # 标准化时间距离
        distances = distances / 100.0  # 归一化到[0,1]范围
        return distances
    
    def __len__(self):
        return len(self.data) - self.sequence_length - self.prediction_horizon + 1
    
    def __getitem__(self, idx):
        # 获取特征序列
        feature_seq = self.feature_data[:, idx:idx + self.sequence_length]
        
        # 获取DataFrame格式的序列(用于LSTM)
        df_seq = self.data.iloc[idx:idx + self.sequence_length]
        sequence = torch.FloatTensor(df_seq.values)
        
        # 获取事件数据
        events = torch.LongTensor(
            self.events[idx:idx + self.sequence_length]
        )
        
        # 获取时间距离
        time_distances = torch.FloatTensor(
            self.time_distances[idx:idx + self.sequence_length]
        )
        
        # 计算目标值
        future_idx = idx + self.sequence_length + self.prediction_horizon - 1
        future_price = self.feature_data[0, future_idx]  # Close price
        current_price = self.feature_data[0, idx + self.sequence_length - 1]
        
        # 计算收益率
        returns = (future_price - current_price) / current_price
        
        # 生成分类标签
        if returns < -0.02:
            label = 0  # 下跌
        elif returns > 0.02:
            label = 2  # 上涨
        else:
            label = 1  # 横盘
            
        # 回归目标
        target = torch.FloatTensor([self.data.iloc[future_idx]['Close']])
        
        return {
            'feature_seq': feature_seq,      # CNN使用
            'sequence': sequence,            # LSTM使用
            'events': events,                # 事件数据
            'time_distances': time_distances,  # 时间距离
            'label': label,                  # 分类标签
            'target': target,                # 回归目标
            'current_price': current_price   # 当前价格
        }

class ATLASCNNFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, event_dim, num_event_types, feature_groups):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 添加输入投影层
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 添加输入标准化层
        self.input_norm = nn.BatchNorm2d(input_dim)
        
        # 修改CNN分支的初始化和结构
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(
                input_dim, hidden_dim, 
                kernel_size=3, 
                padding=1,
                bias=False  # 使用BN时不需要偏置
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(
                hidden_dim, hidden_dim,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )
        
        # 初始化权重
        for m in self.cnn_branch.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # 其他组件
        self.atlas = StockLSTMCell(input_dim, hidden_dim)
        self.event_processor = EventProcessor(
            event_dim=event_dim,
            hidden_dim=hidden_dim,
            num_event_types=num_event_types
        )
        self.tmdo = EnhancedTMDO(input_dim)
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.market_state = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 初始化所有Linear层
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _apply_cnn(self, x):
        """CNN特征提取"""
        batch_size, seq_len, features = x.shape
        
        # [batch, seq, features] -> [batch, features, seq, 1]
        x = x.permute(0, 2, 1).unsqueeze(-1)
        
        # 应用输入标准化
        x = self.input_norm(x)
        
        # # 打印中间状态
        # print(f"Input shape: {x.shape}")
        # print(f"Input range: [{x.min().item()}, {x.max().item()}]")
        # print(f"Input mean: {x.mean().item()}")
        # print(f"Input std: {x.std().item()}")
        
        # CNN特征提取
        cnn_features = self.cnn_branch(x)
        
        # # 打印CNN输出状态
        # print(f"CNN output shape: {cnn_features.shape}")
        # print(f"CNN output range: [{cnn_features.min().item()}, {cnn_features.max().item()}]")
        # print(f"CNN output mean: {cnn_features.mean().item()}")
        # print(f"CNN output std: {cnn_features.std().item()}")
        
        # [batch, hidden, seq, 1] -> [batch, seq, hidden]
        return cnn_features.squeeze(-1).permute(0, 2, 1)

    
    def forward(self, x, events, time_distances):
        batch_size, seq_len = x.size(0), x.size(1)
        
        # CNN特征提取
        cnn_features = self._apply_cnn(x)
        if torch.isnan(cnn_features).any():
            print("NaN detected in CNN features")
            return None, None, None
            
        # TMDO特征提取
        tmdo_features, lap_features = self.tmdo(x)
        if torch.isnan(tmdo_features).any() or torch.isnan(lap_features).any():
            print("NaN detected in TMDO/LAP features")
            return None, None, None
            
        # 初始化ATLAS状态
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        c = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        outputs = []
        combined_features = []
        
        for t in range(seq_len):
            current_x = x[:, t]  # [batch, input_dim]
            current_cnn = cnn_features[:, t]  # [batch, hidden_dim]
            current_events = events[:, t]
            current_distances = time_distances[:, t]
            
            # 将输入投影到hidden_dim维度
            projected_x = self.input_proj(current_x)  # [batch, hidden_dim]
            if torch.isnan(projected_x).any():
                print(f"NaN detected in projected_x at step {t}")
                return None, None, None
                
            # 处理事件影响
            event_impact = self.event_processor(
                current_events,
                h,
                current_distances
            )
            if torch.isnan(event_impact).any():
                print(f"NaN detected in event_impact at step {t}")
                return None, None, None
                
            # ATLAS步进 - 使用投影后的x
            h, c = self.atlas(
                projected_x,  # 现在是hidden_dim维度
                h, c,
                current_x,  # 原始input_dim维度作为indicators
                event_impact
            )
            if torch.isnan(h).any() or torch.isnan(c).any():
                print(f"NaN detected in LSTM state at step {t}")
                return None, None, None
                
            # 交叉注意力处理
            h_query = h.unsqueeze(1)
            cnn_kv = current_cnn.unsqueeze(1)
            
            h_enhanced, _ = self.cross_attention(
                h_query,
                cnn_kv,
                cnn_kv
            )
            h_enhanced = h_enhanced.squeeze(1)
            if torch.isnan(h_enhanced).any():
                print(f"NaN detected in attention output at step {t}")
                return None, None, None
                
            # 特征融合
            fusion_weight = self.fusion_gate(
                torch.cat([h_enhanced, current_cnn], dim=-1)
            )
            if torch.isnan(fusion_weight).any():
                print(f"NaN detected in fusion_weight at step {t}")
                return None, None, None
                
            market_impact = self.market_state(
                torch.cat([h_enhanced, current_cnn], dim=-1)
            )
            if torch.isnan(market_impact).any():
                print(f"NaN detected in market_impact at step {t}")
                return None, None, None
                
            # 最终特征组合
            combined = fusion_weight * h_enhanced + (1 - fusion_weight) * current_cnn
            combined = combined * market_impact
            
            if torch.isnan(combined).any():
                print(f"NaN detected in combined features at step {t}")
                return None, None, None
                
            pred = self.predictor(combined)
            if torch.isnan(pred).any():
                print(f"NaN detected in predictions at step {t}")
                return None, None, None
                
            outputs.append(pred)
            combined_features.append(combined)
        
        predictions = torch.stack(outputs, dim=1)
        features = torch.stack(combined_features, dim=1)
        
        return predictions, tmdo_features, features

# 训练函数
def train_fusion_model(model, train_loader, val_loader, 
                      n_epochs=50, device='cuda', learning_rate=0.0001):
    """
    训练融合模型的函数
    
    Args:
        model: ATLASCNNFusion模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        n_epochs: 训练轮数
        device: 训练设备
        learning_rate: 学习率
    """
    criterion = EnhancedCombinedLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    # 使用Cosine退火学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )
    
    # 初始化早停
    best_val_loss = float('inf')
    patience = 30
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(n_epochs):
        model.train()
        total_metrics = {
            'mse': 0, 'direction': 0, 'smoothness': 0,
            'tmdo_reg': 0, 'group_consistency': 0
        }
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}')
        for batch in pbar:
            # 解包数据
            sequence = batch['sequence'].to(device)
            events = batch['events'].to(device)
            time_distances = batch['time_distances'].to(device)
            target = batch['target'].to(device)
            current_price = batch['current_price'].to(device)
            
            optimizer.zero_grad()
            
            try:
                # 前向传播
                predictions, tmdo_features, group_features = model(
                    sequence, events, time_distances
                )

                if predictions is None or tmdo_features is None or group_features is None:
                    print("Skip this batch due to NaN")
                    continue
                
                # 计算损失
                loss, metrics = criterion(
                    predictions,
                    target,
                    current_price,
                    tmdo_features,
                    group_features
                )
                
                # 检查loss是否为NaN
                if torch.isnan(loss):
                    print("NaN loss detected!")
                    continue
                
                # 反向传播
                loss.backward()
                
                # 更严格的梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
                # 监控梯度
                grad_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item()
                if grad_norm > 10:
                    print(f"Large gradient norm: {grad_norm}")
                
                optimizer.step()
                scheduler.step()
                
                # 更新指标
                for k, v in metrics.items():
                    if not torch.isnan(torch.tensor(v)):
                        total_metrics[k] += v
                
            except RuntimeError as e:
                print(f"Error during training: {e}")
                continue
            
            # 更新进度条,过滤NaN值
            valid_metrics = {
                k: v/len(pbar) 
                for k, v in total_metrics.items() 
                if not np.isnan(v/len(pbar))
            }
            pbar.set_postfix({'loss': loss.item(), **valid_metrics})
        
        # 打印当前学习率
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
        
        # 验证
        model.eval()
        val_loss = 0
        val_metrics = {k: 0 for k in total_metrics.keys()}
        
        with torch.no_grad():
            for batch in val_loader:
                # 解包数据
                sequence = batch['sequence'].to(device)
                events = batch['events'].to(device)
                time_distances = batch['time_distances'].to(device)
                target = batch['target'].to(device)
                current_price = batch['current_price'].to(device)
                
                # 前向传播
                predictions, tmdo_features, group_features = model(
                    sequence, events, time_distances
                )
                
                # 计算损失
                loss, metrics = criterion(
                    predictions,
                    target,
                    current_price,
                    tmdo_features,
                    group_features
                )
                
                # 只累加非NaN的loss
                if not torch.isnan(loss):
                    val_loss += loss.item()
                    for k, v in metrics.items():
                        if not torch.isnan(torch.tensor(v)):
                            val_metrics[k] += v
        
        val_loss /= len(val_loader)
        
        # 打印验证指标
        print(f"\nEpoch {epoch+1} Validation Metrics:")
        print(f"Loss: {val_loss:.4f}")
        for k, v in val_metrics.items():
            print(f"{k}: {v/len(val_loader):.4f}")
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("\nEarly stopping triggered")
                break
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

# 主函数示例
def main():
    # 参数设置
    input_dim = 21  # 输入特征维度
    hidden_dim = 128  # 隐藏层维度
    event_dim = 32  # 事件嵌入维度
    num_event_types = 10  # 事件类型数量
    
    # 数据准备和训练过程
    symbols = ['AAPL', 'MSFT']
    data = combine_stock_data(symbols, '1980-01-01', '2024-01-01')
    events = generate_event_data(data)  # 需要实现这个函数
    
    # 划分训练集和验证集
    train_data, val_data = train_test_split(data, test_size=0.2, shuffle=False)
    train_events, val_events = train_test_split(events, test_size=0.2, shuffle=False)
    
    # 创建数据集
    train_dataset = FusionStockDataset(train_data, train_events)
    val_dataset = FusionStockDataset(val_data, val_events)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=40,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=40,
        shuffle=False,
        num_workers=4
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device: ", device)

    # 创建模型实例
    model = ATLASCNNFusion(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        event_dim=event_dim,
        num_event_types=num_event_types,
        feature_groups=prepare_feature_groups()
    ).to(device=device)

    train_fusion_model(model, train_loader, val_loader)
    
if __name__ == "__main__":
    main()