# Stock Market Prediction with TMDO and 2D-CNN

这个项目结合了两种创新的方法来预测股票市场：时序多维差异算子（TMDO）和专门设计的二维CNN架构。

## 核心理念

### TMDO (时序多维差异算子)

TMDO是一个创新的数学算子，它同时捕捉了两个关键维度的信息：
1. 时间维度：通过二阶时间差分捕捉指标的"加速度"
2. 指标维度：通过加权差分捕捉不同指标间的关系

数学表达式：
$$D(f)_{t,i} = \alpha(\frac{\partial^2 f_i}{\partial t^2}) + \beta(\sum_{j \neq i} w_{ij}(f_{t,i} - f_{t,j}))$$

其中：
- $\frac{\partial^2 f_i}{\partial t^2}$ 捕捉单个指标的时间变化趋势
- $\sum_{j \neq i} w_{ij}(f_{t,i} - f_{t,j})$ 捕捉指标间的相互关系
- $w_{ij}$ 是可学习的权重矩阵，反映指标间的相关性
- $\alpha,\beta$ 是平衡系数，用于调节两个分量的相对重要性

这个算子特别适合金融时间序列，因为它：
1. 能够识别加速上涨/下跌趋势（通过二阶差分）
2. 考虑指标间的相互影响（通过加权差分）
3. 适应性强（通过可学习权重）

### 2D-CNN架构

这个方法的创新之处在于将金融时间序列重新组织为"图像"格式：
- x轴：时间
- y轴：不同指标，按与价格的相关性排序

指标排列示意（y轴）：
1. 价格相关指标（Close, Open, High, Low等）
2. 技术指标（MA, MACD等）
3. 成交量相关指标
4. 其他市场指标

为什么这样组织？
- 相关指标在y轴上相邻，便于卷积核捕捉它们之间的关系
- 不同尺寸的卷积核可以捕捉不同类型的模式

特殊设计的卷积核：
1. 长期趋势识别核 (3,50)
   - 纵向小：只关注少数相关指标
   - 横向大：捕捉长期趋势
   ```
   [--------- 50 ---------]
   [--------- 50 ---------]
   [--------- 50 ---------]
   ```

2. 形态识别核 (5,25)
   - 中等大小：用于识别头肩顶等经典形态
   ```
   [------ 25 ------]
   [------ 25 ------]
   [------ 25 ------]
   [------ 25 ------]
   [------ 25 ------]
   ```

3. 价格-成交量关系核 (7,15)
   - 更高的纵向维度：可以同时考虑更多指标
   ```
   [----- 15 -----]
   [----- 15 -----]
   [----- 15 -----]
   [----- 15 -----]
   [----- 15 -----]
   [----- 15 -----]
   [----- 15 -----]
   ```

4. 短期模式核 (10,10)
   - 正方形：捕捉局部平衡关系
   ```
   [---- 10 ----]
   [---- 10 ----]
   [---- 10 ----]
   ...（10个）
   ```

5. 指标关联核 (15,3)
   - 纵向大：强调指标间关系
   - 横向小：关注短期联动
   ```
   [- 3 -]
   [- 3 -]
   [- 3 -]
   ...（15个）
   ```

## 代码结构

- `model.py`: TMDO和基础模型实现
- `CNN.py`: CNN架构和专用卷积层实现
- `train.py`: 训练逻辑和数据处理

## 实现细节

1. 可变形卷积的使用：
   - 允许卷积核形状动态调整
   - 更好地适应不规则的市场模式

2. 多尺度特征融合：
   ```python
   def forward(self, x):
       trend = self.trend_conv(x)      # 长期趋势
       pattern = self.pattern_conv(x)   # 形态特征
       pv_relation = self.price_volume_conv(x)  # 价格-成交量关系
       short_term = self.short_term_conv(x)     # 短期特征
       indicator = self.indicator_conv(x)        # 指标关系
       
       return trend + pattern + pv_relation + short_term + indicator
   ```

3. 损失函数设计：
   - MSE损失：基础预测准确性
   - 方向预测损失：涨跌方向准确性
   - 连续性损失：预测平滑性
   - TMDO正则化：特征提取质量
   - 特征组一致性：不同层面特征的协调性

## 使用方法

1. 数据准备：
```python
data = combine_stock_data(symbols, start_date, end_date)
```

2. 创建模型：
```python
model = EnhancedStockPredictor(
    input_dim=21,
    hidden_dim=128,
    event_dim=32,
    num_event_types=10,
    feature_groups=feature_groups
)
```

3. 训练：
```python
trained_model = train_enhanced_model(
    model,
    train_loader,
    val_loader,
    n_epochs=50,
    device=device
)
```

这个项目尝试将传统的技术分析理念（如趋势、形态识别）与现代深度学习方法相结合，通过创新的数据组织方式和专门的模型架构来提升预测效果。

# Citation

如果我们的工作对您的研究有帮助，请考虑引用：

```bibtex
@article{chen2023atlas,
  title={Atlas-Cnn: A Novel Hybrid Deep Learning Architecture for Stock Market Prediction},
  author={Chen, Yucheng},
  year={2023},
  journal={SSRN Electronic Journal},
  doi={10.2139/ssrn.5099419},
  url={https://ssrn.com/abstract=5099419}
}
```

> 论文已投稿至 Elsevier Expert Systems with Applications，正在审稿中。