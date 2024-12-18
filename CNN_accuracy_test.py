import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from CNN import *
import os

def evaluate_model(model, data_loader, device='cuda'):
    """
    评估模型性能
    
    Args:
        model: PyTorch模型
        data_loader: 数据加载器
        device: 计算设备
    
    Returns:
        dict: 包含各种评估指标的字典
    """
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    # 用于记录每个类别的统计信息
    class_correct = [0] * 3
    class_total = [0] * 3
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # 统计总体准确度
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 统计每个类别的准确度
            for i in range(len(labels)):
                label = labels[i]
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
            
            # 保存预测结果用于详细分析
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算各项指标
    overall_accuracy = 100. * correct / total
    class_accuracies = [100. * correct / total for correct, total 
                       in zip(class_correct, class_total) if total > 0]
    
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # 生成分类报告
    class_names = ['下跌', '平稳', '上涨']
    classification_rep = classification_report(all_labels, all_preds, 
                                            target_names=class_names)
    
    # 构建结果字典
    results = {
        'overall_accuracy': overall_accuracy,
        'class_accuracies': {class_names[i]: acc 
                            for i, acc in enumerate(class_accuracies)},
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': classification_rep,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return results

def plot_results(results, save_path=None):
    """
    可视化评估结果
    
    Args:
        results: evaluate_model返回的结果字典
        save_path: 保存图表的路径(可选)
    """
    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    
    # 1. 绘制准确率柱状图
    accuracies = results['class_accuracies']
    axes[0].bar(accuracies.keys(), accuracies.values())
    axes[0].set_title('各类别预测准确率')
    axes[0].set_ylabel('准确率 (%)')
    axes[0].grid(True)
    
    # 2. 绘制混淆矩阵热力图
    conf_matrix = np.array(results['confusion_matrix'])
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['下跌', '平稳', '上涨'],
                yticklabels=['下跌', '平稳', '上涨'],
                ax=axes[1])
    axes[1].set_title('预测混淆矩阵')
    axes[1].set_xlabel('预测类别')
    axes[1].set_ylabel('实际类别')
    
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        plt.savefig(save_path)
    plt.show()

def test_model(model, train_loader, test_loader, save_dir='./results'):
    """
    完整的模型测试流程
    
    Args:
        model: PyTorch模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        save_dir: 结果保存目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 评估训练集
    print("评估训练集性能...")
    train_results = evaluate_model(model, train_loader)
    
    # 评估测试集
    print("评估测试集性能...")
    test_results = evaluate_model(model, test_loader)
    
    # 打印结果
    print("\n训练集结果:")
    print(f"整体准确率: {train_results['overall_accuracy']:.2f}%")
    print("\n分类报告:\n", train_results['classification_report'])
    
    print("\n测试集结果:")
    print(f"整体准确率: {test_results['overall_accuracy']:.2f}%")
    print("\n分类报告:\n", test_results['classification_report'])
    
    # 保存结果
    results = {
        'train': train_results,
        'test': test_results,
        'model_timestamp': timestamp
    }
    
    with open(f'{save_dir}/results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # 绘制并保存图表
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plot_results(train_results, f'{save_dir}/train_vis_{timestamp}.png')
    plt.subplot(2, 1, 2)
    plot_results(test_results, f'{save_dir}/test_vis_{timestamp}.png')

class StockDataset(Dataset):
    def __init__(self, data, sequence_length=250, prediction_horizon=5):
        """
        Args:
            data: DataFrame containing stock data
            sequence_length: Number of time steps to look back
            prediction_horizon: Number of days to predict ahead
        """
        self.data = torch.FloatTensor(data.values)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # 计算并打印标签分布
        self.calculate_label_distribution()
        
    def calculate_label_distribution(self):
        """计算并打印标签分布情况"""
        labels = []
        for i in range(len(self)):
            future_price = self.data[i + self.sequence_length + self.prediction_horizon - 1, 3]  # Close price
            current_price = self.data[i + self.sequence_length - 1, 3]
            returns = (future_price - current_price) / current_price
            
            # 打印一些样本的实际收益率
            if i < 10:
                print(f"Sample {i} return: {returns:.4f}")
                
            if returns < -0.02:
                labels.append(0)
            elif returns > 0.02:
                labels.append(2)
            else:
                labels.append(1)
        
        # 打印标签分布
        unique, counts = np.unique(labels, return_counts=True)
        distribution = dict(zip(unique, counts))
        print("\nLabel Distribution:")
        print(f"下跌 (0): {distribution.get(0, 0)}")
        print(f"平稳 (1): {distribution.get(1, 0)}")
        print(f"上涨 (2): {distribution.get(2, 0)}")
        
        # 计算平均收益率和标准差
        all_returns = []
        for i in range(len(self)):
            future_price = self.data[i + self.sequence_length + self.prediction_horizon - 1, 3]
            current_price = self.data[i + self.sequence_length - 1, 3]
            returns = (future_price - current_price) / current_price
            all_returns.append(returns)
            
        print(f"\n收益率统计:")
        print(f"平均收益率: {np.mean(all_returns):.4f}")
        print(f"收益率标准差: {np.std(all_returns):.4f}")
        print(f"最大收益率: {np.max(all_returns):.4f}")
        print(f"最小收益率: {np.min(all_returns):.4f}")
        
    def __len__(self):
        return len(self.data) - self.sequence_length - self.prediction_horizon + 1
        
    def __getitem__(self, idx):
        X = self.data[idx:idx + self.sequence_length]
        
        future_price = self.data[idx + self.sequence_length + self.prediction_horizon - 1, 3]  # Close price
        current_price = self.data[idx + self.sequence_length - 1, 3]
        returns = (future_price - current_price) / current_price
        
        # 使用更合理的阈值
        if returns < -0.02:
            y = 0
        elif returns > 0.02:
            y = 2
        else:
            y = 1
            
        return X, y

# 使用示例
if __name__ == "__main__":
    # 加载模型
    model = StockPredictor()
    model.load_state_dict(torch.load('best_model.pth'))
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 准备数据
    symbols = ['AAPL', 'GOOGL', 'MSFT', "NVDA", "AMD", "AMZN", "TSLA"]  # 示例股票
    data = combine_stock_data(symbols, '2010-01-01', '2024-01-01')
    
    # 创建数据集
    dataset = StockDataset(data)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=200, shuffle=False)
    
    # 测试模型
    test_model(model, train_loader, test_loader)