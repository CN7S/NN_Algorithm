import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

class LSTMCell:
    """
    一个简单的LSTM单元实现。
    """
    def __init__(self, input_size, hidden_size):
        # 初始化权重和偏置
        # 我们将所有门的权重合并成一个大的矩阵，便于计算
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 权重矩阵形状: [hidden_size, input_size + hidden_size]
        # 偏置向量形状: [hidden_size]
        # 顺序: 输入门, 遗忘门, 候选细胞状态, 输出门
        total_size = input_size + hidden_size
        self.W = np.random.randn(4 * hidden_size, total_size) * 0.01
        self.b = np.zeros((4 * hidden_size, 1))
        
    def forward(self, x, h_prev, c_prev):
        """
        前向传播单个时间步。
        
        参数:
        x - 当前输入，形状 [input_size, 1]
        h_prev - 上一个隐藏状态，形状 [hidden_size, 1]
        c_prev - 上一个细胞状态，形状 [hidden_size, 1]
        """
        # 拼接输入和上一个隐藏状态
        concat = np.vstack((h_prev, x)) # 形状 [input_size + hidden_size, 1]
        
        # 计算所有门的激活值
        gates = self.W @ concat + self.b # 形状 [4 * hidden_size, 1]
        
        # 分割成各个门
        i_gate = sigmoid(gates[0:self.hidden_size])                    # 输入门
        f_gate = sigmoid(gates[self.hidden_size:2*self.hidden_size])   # 遗忘门
        g_gate = tanh(gates[2*self.hidden_size:3*self.hidden_size])    # 候选细胞状态
        o_gate = sigmoid(gates[3*self.hidden_size:])                   # 输出门
        
        # 更新细胞状态
        c_next = f_gate * c_prev + i_gate * g_gate
        
        # 更新隐藏状态
        h_next = o_gate * tanh(c_next)
        
        return h_next, c_next

# 使用示例
if __name__ == "__main__":
    # 定义参数
    input_size = 3
    hidden_size = 2
    seq_length = 5
    
    # 创建LSTM单元
    lstm_cell = LSTMCell(input_size, hidden_size)
    
    # 初始化隐藏状态和细胞状态
    h_prev = np.zeros((hidden_size, 1))
    c_prev = np.zeros((hidden_size, 1))
    
    # 创建一个简单的输入序列
    X = [np.random.randn(input_size, 1) for _ in range(seq_length)]
    
    # 前向传播
    print("从零开始实现LSTM前向传播:")
    for t in range(seq_length):
        h_prev, c_prev = lstm_cell.forward(X[t], h_prev, c_prev)
        print(f"时间步 {t}: h_t = {h_prev.flatten()}, c_t = {c_prev.flatten()}")