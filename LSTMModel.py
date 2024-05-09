
import torch.nn as nn
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
        self.lstm2 = nn.LSTM(50, 50, batch_first=True)
        self.lstm3 = nn.LSTM(50, 50, batch_first=True)
        self.dense = nn.Linear(50, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = x[:, -1, :]  # 获取最后一个时间步的输出
        return self.dense(x)

