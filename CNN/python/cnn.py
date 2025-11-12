import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # He初始化
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.bias = np.zeros((out_channels, 1))
        
        self.cache = None
        
    def forward(self, x):
        """前向传播"""
        batch_size, in_channels, in_h, in_w = x.shape
        
        # 计算输出尺寸
        out_h = (in_h + 2*self.padding - self.kernel_size) // self.stride + 1
        out_w = (in_w + 2*self.padding - self.kernel_size) // self.stride + 1
        
        # 填充
        if self.padding > 0:
            x_padded = np.pad(x, ((0,0), (0,0), (self.padding,self.padding), 
                                (self.padding,self.padding)), mode='constant')
        else:
            x_padded = x
            
        # 初始化输出
        output = np.zeros((batch_size, self.out_channels, out_h, out_w))
        
        # 卷积操作
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for h in range(out_h):
                    for w in range(out_w):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        
                        receptive_field = x_padded[b, :, h_start:h_end, w_start:w_end]
                        output[b, c_out, h, w] = np.sum(
                            receptive_field * self.weights[c_out]) + self.bias[c_out]
        
        self.cache = (x, x_padded)
        return output
    
    def backward(self, dout, learning_rate):
        """反向传播"""
        x, x_padded = self.cache
        batch_size, in_channels, in_h, in_w = x.shape
        
        # 初始化梯度
        dx_padded = np.zeros_like(x_padded)
        dw = np.zeros_like(self.weights)
        db = np.zeros_like(self.bias)
        
        out_h, out_w = dout.shape[2], dout.shape[3]
        
        # 计算梯度
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                db[c_out] += np.sum(dout[b, c_out])
                for h in range(out_h):
                    for w in range(out_w):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        
                        receptive_field = x_padded[b, :, h_start:h_end, w_start:w_end]
                        dw[c_out] += dout[b, c_out, h, w] * receptive_field
                        dx_padded[b, :, h_start:h_end, w_start:w_end] += (
                            dout[b, c_out, h, w] * self.weights[c_out])
        
        # 去除填充
        if self.padding > 0:
            dx = dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dx = dx_padded
        
        # 更新参数
        self.weights -= learning_rate * dw / batch_size
        self.bias -= learning_rate * db / batch_size
        
        return dx

class MaxPool2D:
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        self.cache = None
        
    def forward(self, x):
        batch_size, channels, in_h, in_w = x.shape
        
        out_h = (in_h - self.kernel_size) // self.stride + 1
        out_w = (in_w - self.kernel_size) // self.stride + 1
        
        output = np.zeros((batch_size, channels, out_h, out_w))
        mask = np.zeros_like(x)
        
        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_h):
                    for w in range(out_w):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        
                        receptive_field = x[b, c, h_start:h_end, w_start:w_end]
                        output[b, c, h, w] = np.max(receptive_field)
                        
                        # 记录最大值位置
                        max_pos = np.unravel_index(
                            np.argmax(receptive_field), receptive_field.shape)
                        mask[b, c, h_start + max_pos[0], w_start + max_pos[1]] = 1
        
        self.cache = mask
        return output
    
    def backward(self, dout):
        mask = self.cache
        dx = np.zeros_like(mask)
        
        batch_size, channels, out_h, out_w = dout.shape
        
        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_h):
                    for w in range(out_w):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        
                        # 只将梯度传递到最大值位置
                        dx[b, c, h_start:h_end, w_start:w_end] += (
                            dout[b, c, h, w] * mask[b, c, h_start:h_end, w_start:w_end])
        
        return dx

class ReLU:
    def __init__(self):
        self.cache = None
        
    def forward(self, x):
        self.cache = x
        return np.maximum(0, x)
    
    def backward(self, dout):
        x = self.cache
        dx = dout * (x > 0)
        return dx

class Flatten:
    def __init__(self):
        self.input_shape = None
        
    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)
    
    def backward(self, dout):
        return dout.reshape(self.input_shape)

class Linear:
    def __init__(self, in_features, out_features):
        scale = np.sqrt(2.0 / in_features)
        self.weights = np.random.randn(out_features, in_features) * scale
        self.bias = np.zeros((out_features, 1))
        self.cache = None
        
    def forward(self, x):
        self.cache = x
        return self.weights @ x.T + self.bias
        
    def backward(self, dout, learning_rate):
        x = self.cache
        batch_size = x.shape[0]
        
        dw = dout @ x
        db = np.sum(dout, axis=1, keepdims=True)
        dx = dout.T @ self.weights
        
        self.weights -= learning_rate * dw.T / batch_size
        self.bias -= learning_rate * db / batch_size
        
        return dx

class SoftmaxCrossEntropy:
    def __init__(self):
        self.cache = None
        
    def forward(self, x, y):
        # Softmax
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        softmax = exp_x / np.sum(exp_x, axis=0, keepdims=True)
        
        # Cross entropy loss
        batch_size = x.shape[1]
        loss = -np.sum(y * np.log(softmax + 1e-8)) / batch_size
        
        self.cache = (softmax, y, batch_size)
        return loss
    
    def backward(self):
        softmax, y, batch_size = self.cache
        return (softmax - y) / batch_size

class SimpleCNN:
    def __init__(self):
        self.layers = [
            Conv2D(1, 8, kernel_size=3, padding=1),  # 28x28x1 -> 28x28x8
            ReLU(),
            MaxPool2D(2, stride=2),                  # 28x28x8 -> 14x14x8
            
            Conv2D(8, 16, kernel_size=3, padding=1), # 14x14x8 -> 14x14x16
            ReLU(),
            MaxPool2D(2, stride=2),                  # 14x14x16 -> 7x7x16
            
            Flatten(),
            Linear(7*7*16, 128),
            ReLU(),
            Linear(128, 10)
        ]
        
        self.loss_fn = SoftmaxCrossEntropy()
        
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, dout, learning_rate):
        # 反向传播经过所有层
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                if isinstance(layer, (Conv2D, Linear)):
                    dout = layer.backward(dout, learning_rate)
                else:
                    dout = layer.backward(dout)
        return dout
    
    def train(self, x, y, learning_rate):
        # 前向传播
        logits = self.forward(x)
        
        # 计算损失
        loss = self.loss_fn.forward(logits, y.T)
        
        # 反向传播
        dout = self.loss_fn.backward()
        self.backward(dout, learning_rate)
        
        return loss
    
    def predict(self, x):
        logits = self.forward(x)
        return np.argmax(logits, axis=0)
    
    def accuracy(self, x, y):
        predictions = self.predict(x)
        return np.mean(predictions == np.argmax(y, axis=1))


def load_mnist():
    """加载MNIST数据集"""
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    
    X = mnist.data.astype(np.float32).reshape(-1, 1, 28, 28) / 255.0
    y = mnist.target.astype(np.int32)
    
    # 转换为one-hot编码
    y_onehot = np.eye(10)[y]
    
    return X, y_onehot, y

def train_cnn():
    # 加载数据
    X, y_onehot, y = load_mnist()
    
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42)
    
    # 创建模型
    model = SimpleCNN()
    
    # 训练参数
    epochs = 5
    batch_size = 64
    learning_rate = 0.01
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    print("Starting training...")
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        # 随机打乱数据
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        # 小批量训练
        for i in range(0, len(X_train), batch_size):
            batch_x = X_train_shuffled[i:i+batch_size]
            batch_y = y_train_shuffled[i:i+batch_size]
            
            loss = model.train(batch_x, batch_y, learning_rate)
            epoch_loss += loss
            num_batches += 1
            
            if i % 1000 == 0:
                print(f"Epoch {epoch+1}, Batch {i//batch_size}, Loss: {loss:.4f}")
        
        # 计算准确率
        train_acc = model.accuracy(X_train[:1000], y_train[:1000])  # 用部分数据计算
        test_acc = model.accuracy(X_test[:1000], y_test[:1000])
        
        train_losses.append(epoch_loss / num_batches)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Loss: {epoch_loss/num_batches:.4f}")
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print("-" * 50)
    
    return model, train_losses, train_accuracies, test_accuracies

# 训练模型
model, train_losses, train_accuracies, test_accuracies = train_cnn()

def plot_results(train_losses, train_accuracies, test_accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(test_accuracies, label='Test Accuracy')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# 绘制训练结果
plot_results(train_losses, train_accuracies, test_accuracies)

def test_model(model, X_test, y_test, num_samples=10):
    """测试模型并显示一些预测结果"""
    indices = np.random.choice(len(X_test), num_samples)
    
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(indices):
        plt.subplot(1, num_samples, i+1)
        
        # 显示图像
        img = X_test[idx].reshape(28, 28)
        plt.imshow(img, cmap='gray')
        
        # 预测
        prediction = model.predict(X_test[idx:idx+1])
        true_label = np.argmax(y_test[idx])
        
        color = 'green' if prediction == true_label else 'red'
        plt.title(f'True: {true_label}\nPred: {prediction[0]}', color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 加载测试数据
X, y_onehot, y = load_mnist()
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# 测试模型
test_model(model, X_test, y_test)