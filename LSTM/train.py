import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1, dropout=0.2):
        super(TimeSeriesLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden=None):
        # LSTM前向传播
        lstm_out, hidden = self.lstm(x, hidden)
        
        # 取最后一个时间步
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        """初始化隐藏状态"""
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_(),
                  weight.new(self.num_layers, batch_size, self.hidden_size).zero_())
        return hidden

def create_synthetic_data(seq_length=200, train_ratio=0.8):
    """创建合成时间序列数据"""
    t = np.linspace(0, 20, seq_length)
    data = np.sin(t) + 0.1 * np.random.randn(seq_length) + 0.1 * t
    
    # 标准化
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    
    # 划分训练测试集
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    return train_data, test_data, scaler

def create_sequences(data, seq_len=10):
    """创建输入序列和标签"""
    sequences = []
    labels = []
    
    for i in range(len(data) - seq_len):
        seq = data[i:i+seq_len]
        label = data[i+seq_len]
        sequences.append(seq)
        labels.append(label)
    
    return np.array(sequences), np.array(labels)

# 主训练函数
def train_lstm_model():
    # 超参数
    seq_length = 20
    hidden_size = 64
    num_layers = 2
    learning_rate = 0.001
    epochs = 100
    batch_size = 16
    
    # 创建数据
    train_data, test_data, scaler = create_synthetic_data()
    
    # 创建序列
    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)
    
    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train).unsqueeze(-1)  # (samples, seq_len, features)
    y_train = torch.FloatTensor(y_train).unsqueeze(-1)
    X_test = torch.FloatTensor(X_test).unsqueeze(-1)
    y_test = torch.FloatTensor(y_test).unsqueeze(-1)
    
    print(f"训练数据形状: {X_train.shape}")
    print(f"测试数据形状: {X_test.shape}")
    
    # 创建模型
    model = TimeSeriesLSTM(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=1,
        dropout=0.3
    )
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)
    
    # 训练记录
    train_losses = []
    test_losses = []
    
    # 训练循环
    print("开始训练...")
    for epoch in range(epochs):
        model.train()
        epoch_train_losses = []
        
        # 迷你批次训练
        for i in range(0, len(X_train), batch_size):
            # 获取批次数据
            if i + batch_size > len(X_train):
                break
                
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            # 前向传播
            optimizer.zero_grad()
            outputs, _ = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪 - 防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 更新参数
            optimizer.step()
            
            epoch_train_losses.append(loss.item())
        
        # 计算平均训练损失
        avg_train_loss = np.mean(epoch_train_losses)
        train_losses.append(avg_train_loss)
        
        # 验证
        model.eval()
        with torch.no_grad():
            test_outputs, _ = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            test_losses.append(test_loss.item())
        
        # 更新学习率
        scheduler.step(test_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Test Loss: {test_loss.item():.6f}')
    
    return model, train_losses, test_losses, X_test, y_test, scaler

# 运行训练
model, train_losses, test_losses, X_test, y_test, scaler = train_lstm_model()