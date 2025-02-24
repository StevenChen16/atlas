# Stock Market Prediction with TMDO and 2D-CNN

This project combines two innovative approaches to predict the stock market: Temporal Multi-Dimensional Operator (TMDO) and a specially designed 2D-CNN architecture.

## Core Concepts

### TMDO (Temporal Multi-Dimensional Operator)

TMDO is an innovative mathematical operator that simultaneously captures information in two key dimensions:
1. Temporal dimension: Captures indicator "acceleration" through second-order time differentiation
2. Indicator dimension: Captures relationships between different indicators through weighted differentiation

Mathematical expression:
$$D(f)_{t,i} = \alpha(\frac{\partial^2 f_i}{\partial t^2}) + \beta(\sum_{j \neq i} w_{ij}(f_{t,i} - f_{t,j}))$$

Where:
- $\frac{\partial^2 f_i}{\partial t^2}$ captures temporal trend changes of individual indicators
- $\sum_{j \neq i} w_{ij}(f_{t,i} - f_{t,j})$ captures inter-indicator relationships
- $w_{ij}$ is a learnable weight matrix reflecting indicator correlations
- $\alpha,\beta$ are balance coefficients that adjust the relative importance of both components

This operator is particularly suitable for financial time series because it:
1. Can identify accelerating upward/downward trends (through second-order differentiation)
2. Considers inter-indicator influences (through weighted differentiation)
3. Highly adaptive (through learnable weights)

### 2D-CNN Architecture

The innovation of this method lies in reorganizing financial time series into an "image" format:
- x-axis: Time
- y-axis: Different indicators, sorted by their correlation with price

Indicator arrangement illustration (y-axis):
1. Price-related indicators (Close, Open, High, Low, etc.)
2. Technical indicators (MA, MACD, etc.)
3. Volume-related indicators
4. Other market indicators

Why organize this way?
- Related indicators are adjacent on the y-axis, facilitating relationship capture by convolution kernels
- Different kernel sizes can capture different types of patterns

Specially designed convolution kernels:
1. Long-term trend recognition kernel (3,50)
   - Vertically small: focuses on few related indicators
   - Horizontally large: captures long-term trends
   ```
   [--------- 50 ---------]
   [--------- 50 ---------]
   [--------- 50 ---------]
   ```

2. Pattern recognition kernel (5,25)
   - Medium size: for identifying classic patterns like head and shoulders
   ```
   [------ 25 ------]
   [------ 25 ------]
   [------ 25 ------]
   [------ 25 ------]
   [------ 25 ------]
   ```

3. Price-volume relationship kernel (7,15)
   - Higher vertical dimension: can consider more indicators simultaneously
   ```
   [----- 15 -----]
   [----- 15 -----]
   [----- 15 -----]
   [----- 15 -----]
   [----- 15 -----]
   [----- 15 -----]
   [----- 15 -----]
   ```

4. Short-term pattern kernel (10,10)
   - Square: captures local balance relationships
   ```
   [---- 10 ----]
   [---- 10 ----]
   [---- 10 ----]
   ...（10 rows）
   ```

5. Indicator correlation kernel (15,3)
   - Vertically large: emphasizes inter-indicator relationships
   - Horizontally small: focuses on short-term linkages
   ```
   [- 3 -]
   [- 3 -]
   [- 3 -]
   ...（15 rows）
   ```

## Code Structure

- `model.py`: TMDO and base model implementation
- `CNN.py`: CNN architecture and specialized convolution layer implementation
- `train.py`: Training logic and data processing

## Implementation Details

1. Use of deformable convolution:
   - Allows dynamic adjustment of kernel shapes
   - Better adapts to irregular market patterns

2. Multi-scale feature fusion:
   ```python
   def forward(self, x):
       trend = self.trend_conv(x)      # Long-term trend
       pattern = self.pattern_conv(x)   # Pattern features
       pv_relation = self.price_volume_conv(x)  # Price-volume relationship
       short_term = self.short_term_conv(x)     # Short-term features
       indicator = self.indicator_conv(x)        # Indicator relationships
       
       return trend + pattern + pv_relation + short_term + indicator
   ```

3. Loss function design:
   - MSE loss: Basic prediction accuracy
   - Direction prediction loss: Up/down direction accuracy
   - Continuity loss: Prediction smoothness
   - TMDO regularization: Feature extraction quality
   - Feature group consistency: Coordination of different aspect features

## Usage

1. Data preparation:
```python
data = combine_stock_data(symbols, start_date, end_date)
```

2. Create model:
```python
model = EnhancedStockPredictor(
    input_dim=21,
    hidden_dim=128,
    event_dim=32,
    num_event_types=10,
    feature_groups=feature_groups
)
```

3. Training:
```python
trained_model = train_enhanced_model(
    model,
    train_loader,
    val_loader,
    n_epochs=50,
    device=device
)
```

This project attempts to combine traditional technical analysis concepts (such as trends and pattern recognition) with modern deep learning methods through innovative data organization and specialized model architecture to enhance prediction performance.

# Citation

If our work is helpful for your research, please consider citing:

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

> The paper has been submitted to Elsevier Expert Systems with Applications and is under review.