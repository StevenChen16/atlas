import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureGroupLayer(nn.Module):
    def __init__(self, group_features, hidden_dim=64):  # 添加hidden_dim参数
        super().__init__()
        self.group_features = group_features
        
        # 添加投影层,将输入特征投影到hidden_dim维度
        self.projection = nn.Linear(len(group_features), hidden_dim)
        
        # 现在使用固定的hidden_dim作为embed_dim
        self.group_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,  # 使用hidden_dim
            num_heads=8,  # 8头注意力,因为64可以被8整除
            batch_first=True
        )
        
        # 添加输出投影,将结果投影回原始维度
        self.output_projection = nn.Linear(hidden_dim, len(group_features))
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        group_data = x[:, :, self.group_features]
        
        # 投影到高维空间
        projected = self.projection(group_data)
        
        # 组内注意力
        attn_out, _ = self.group_attention(
            projected, projected, projected
        )
        
        # 投影回原始维度
        output = self.output_projection(attn_out)
        
        return output

class LaplacianStockLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        
        # 创建2D拉普拉斯核
        kernel_2d = torch.Tensor([
            [1, -2, 1],
            [-2, 4, -2],
            [1, -2, 1]
        ])
        
        # 扩展为完整的卷积核
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            padding=1
        )
        
        # 初始化卷积权重
        with torch.no_grad():
            self.conv.weight.data = kernel_2d.unsqueeze(0).unsqueeze(0)
            self.conv.bias.data.zero_()
        
        # 冻结参数
        for param in self.conv.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        batch_size, seq_len, n_features = x.shape
        # print(f"Input shape: {x.shape}")
        
        # 将输入转换为图像格式
        x = x.unsqueeze(1)  # (batch, 1, seq_len, features)
        # print(f"After unsqueeze shape: {x.sha/pe}")
        
        # 应用2D拉普拉斯
        laplacian = self.conv(x)
        # print(f"After conv shape: {laplacian.shape}")
        
        # 确保输出维度正确
        output = laplacian.squeeze(1)  # 移除channel维度
        # print(f"Output shape: {output.shape}")
        
        assert output.shape == (batch_size, seq_len, n_features), \
            f"Output shape {output.shape} doesn't match expected shape {(batch_size, seq_len, n_features)}"
        
        return output

class EventProcessor(nn.Module):
    def __init__(self, event_dim, hidden_dim, num_event_types):
        super().__init__()
        self.event_embed = nn.Embedding(num_event_types, event_dim)
        
        # 事件影响编码器
        self.impact_encoder = nn.Sequential(
            nn.Linear(event_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 时间衰减注意力
        self.time_decay_attention = nn.MultiheadAttention(
            hidden_dim, 
            num_heads=4,
            batch_first=True
        )
        
    def forward(self, events, market_state, time_distances):
        # 事件编码
        event_embeds = self.event_embed(events)
        
        # 计算事件影响
        impact = self.impact_encoder(event_embeds)
        
        # 考虑时间衰减的注意力
        decay_weights = torch.exp(-0.1 * time_distances).unsqueeze(-1)
        impact = impact * decay_weights
        
        # 与市场状态的交互
        attn_out, _ = self.time_decay_attention(
            market_state.unsqueeze(1),
            impact,
            impact
        )
        
        return attn_out.squeeze(1)

class StockLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        # 标准LSTM门
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        
        # 将indicators投影到hidden_size维度
        self.indicator_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        
        # 技术指标门
        self.indicator_gate = nn.Linear(hidden_size * 2, hidden_size)
        
        # 事件门
        self.event_gate = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self, x, h_prev, c_prev, indicators, event_impact):
        # 基础LSTM步骤
        h_next, c_next = self.lstm(x, (h_prev, c_prev))
        
        # 首先将indicators投影到hidden_size维度
        indicators_hidden = self.indicator_projection(indicators)
        
        # 技术指标整合
        indicator_combined = torch.cat([h_next, indicators_hidden], dim=-1)
        indicator_gate = torch.sigmoid(self.indicator_gate(indicator_combined))
        h_next = h_next * indicator_gate + indicators_hidden * (1 - indicator_gate)
        
        # 事件影响整合
        event_combined = torch.cat([h_next, event_impact], dim=-1)
        event_gate = torch.sigmoid(self.event_gate(event_combined))
        h_next = h_next * event_gate + event_impact * (1 - event_gate)
        
        return h_next, c_next

class StockPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, event_dim,
                 num_event_types, feature_groups):
        super().__init__()
        # 保存重要参数为类属性
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.event_dim = event_dim
        self.num_event_types = num_event_types
        self.feature_groups = feature_groups
        
        # 特征组处理层
        self.group_layers = nn.ModuleDict({
            name: FeatureGroupLayer(indices)
            for name, indices in feature_groups.items()
        })
        
        # 拉普拉斯层
        self.laplacian = LaplacianStockLayer(input_dim)
        
        # 事件处理器
        self.event_processor = EventProcessor(
            event_dim, hidden_dim, num_event_types
        )
        
        # 改进的LSTM
        self.lstm = StockLSTMCell(input_dim, hidden_dim)
        
        # 预测层
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x, events, time_distances):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # 初始化状态
        h = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        c = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        
        # 计算特征组
        group_outputs = []
        for group_layer in self.group_layers.values():
            group_out = group_layer(x)
            group_outputs.append(group_out)
        
        # 拉普拉斯特征
        lap_features = self.laplacian(x)
        
        outputs = []
        for t in range(seq_len):
            # 获取当前时间步的特征
            current_x = x[:, t, :]  # (batch_size, input_dim)
            current_indicators = lap_features[:, t, :]  # (batch_size, input_dim)
            
            # 处理当前时间步的事件
            current_events = events[:, t]
            current_distances = time_distances[:, t]
            event_impact = self.event_processor(
                current_events, h, current_distances
            )  # 确保输出维度是(batch_size, hidden_dim)
            
            # LSTM步进
            h, c = self.lstm(
                current_x, h, c,
                current_indicators, event_impact
            )
            
            # 生成预测
            pred = self.predictor(h)
            outputs.append(pred)
        
        return torch.stack(outputs, dim=1)

# # 使用示例
def prepare_feature_groups():
    return {
        'price': [0, 1, 2, 3, 4],  # OHLC + Adj Close
        'volume': [5, 15],         # Volume, Volume_MA5
        'ma': [6, 7],             # MA5, MA20
        'macd': [8, 9, 10],       # MACD, Signal, Hist
        'momentum': [11, 16],      # RSI, CRSI
        'bollinger': [12, 13, 14], # Upper, Middle, Lower
        'kalman': [17, 18],        # Kalman_Price, Kalman_Trend
        'fft': [19, 20]           # FFT_21, FFT_63
    }

# # 在创建模型时,为所有组指定相同的hidden_dim
# model = StockPredictor(
#     input_dim=17,         # 总特征数
#     hidden_dim=128,      # 隐藏层维度
#     event_dim=32,        # 事件嵌入维度
#     num_event_types=10,  # 事件类型数量
#     feature_groups=prepare_feature_groups()
# )

# print(model.eval())