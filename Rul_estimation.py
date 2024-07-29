import numpy as np
import scienceplots
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

'''
分析数据：第一列是单位，第二列是时间h，sensor1-sensor23一共23种监测数据
         最后一列是反应超声流量寿命的健康指标
任务：通过sensor1-sensor23来预测最后一列
'''

# 设置pandas显示选项
pd.set_option('display.max_rows', None)                # 显示所有行
pd.set_option('display.max_columns', None)             # 显示所有列
pd.set_option('display.max_colwidth', None)            # 显示完整的列内容
pd.set_option('display.width', None)                   # 自动调整控制台宽度
pd.set_option('display.expand_frame_repr', False)      # 防止自动换行


# 数据预处理
def data_preprocessing(x_data, y_data, window_size, step):
    data_x = []
    data_y = []
    for i in range(0, len(x_data) - window_size, step):
        data_x.append(x_data[i:i + window_size])    # 取窗口
        data_y.append(y_data[i + window_size])

    return np.array(data_x).astype(float), np.array(data_y).astype(float)


# 定义数据集
class HealthDataset(Dataset):
    # 类初始化方法
    def __init__(self, data_input, data_target):
        self.data_input = data_input
        self.data_target = data_target

    #以怎样形式获取数据集（怎样从数据里面把数据集挑出来）（index是索引）
    def __getitem__(self, index):
        data_input = torch.tensor(self.data_input[index], dtype=torch.float)      # 转换为tensor，指定数据类型
        data_target = torch.tensor(self.data_target[index], dtype=torch.float)
        return data_input, data_target

    # 获取数据集长度
    def __len__(self):
        return len(self.data_input) # 因为输入输出一样长


# 构建模型
class HealthModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()   # 继承父类
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)   # LSTM层（特征数量，隐藏层大小，几层LSTM,）
        self.linear = nn.Linear(hidden_size, output_size)                            # 全连接层（隐藏层，输出）

    # 输入后怎么进行计算
    def forward(self, input_seq):
        output, _ = self.lstm(input_seq)    # 只用最后输出，其他两个用不上
        pred = self.linear(output)
        pred = pred[:, -1, :].squeeze()     # 最后一个时间步长做预测值，为了算loss，把一个维度去掉
        return pred


if __name__ == '__main__':
    df = pd.read_excel(r"data.xlsx")  # 读取数据
 
    # 把sensor1-sensor23和健康预测数据分离出来
    sensors = df.drop(columns=["Unit", "Time(h)", "Health Indicator"]).to_numpy()
    indicator = df["Health Indicator"].to_numpy()   # 健康预测数据

    # sensor1-sensor23有的0.几，有的1000多，模型可能学的不是很好。所以用归一化方式，最大最小归一化到0-1之间
    scaler = MinMaxScaler()   # 创建类的对象，调用类
    sensors = scaler.fit_transform(sensors) # 用归一化方式，最大最小归一化到0-1之间 

    window_size = 10                       # 定义窗口大小
    step = 1                               # 滑动一步
    batch_size = 8
    input_size = sensors.shape[1]          # 有多少列
    hidden_size = 512
    output_size = 1                        # 输出
    num_layers = 2
    epoch = 1000
    lr = 0.001
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_x, data_y = data_preprocessing(sensors, indicator, window_size, step)

    # 划分为训练和测试，不打乱训练0.9，测试0.1；再划分验证集0.1，训练0.8
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.1, shuffle=False)  
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=1 / 9, shuffle=False)
 
    # 把划分好的数据传入数据集里去

    #训练数据集
    train_dataset = HealthDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # 验证数据集
    valid_dataset = HealthDataset(x_valid, y_valid)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 测试数据集
    test_dataset = HealthDataset(x_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)


    # 创建模型对象
    model = HealthModel(input_size, hidden_size, num_layers, output_size).to(device)

    # 定义优化器和损失函数
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_valid_loss = np.inf
    patience = 30
    count = 0
    for e in range(epoch):
        model.train()
        train_losses = []
        for data_input, data_target in train_dataloader:
            data_input, data_target = data_input.to(device), data_target.to(device)
            # 前馈
            optimizer.zero_grad()
            out = model(data_input)
            loss = criterion(out, data_target)
            # 反馈
            loss.backward()
            # 更新
            optimizer.step()

            train_losses.append(loss.cpu().item())

        model.eval()
        valid_losses = []
        with torch.no_grad():
            for data_input, data_target in valid_dataloader:
                data_input, data_target = data_input.to(device), data_target.to(device)
                out = model(data_input)
                loss = criterion(out, data_target)
                valid_losses.append(loss.cpu().item())

        avg_train_loss = np.mean(train_losses)
        avg_valid_loss = np.mean(valid_losses)

        print(f"Epoch {e + 1} Train Loss: {avg_train_loss:} Valid Loss: {avg_valid_loss}")

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
        else:
            count += 1

        if count > patience:
            print("Early stopping")
            torch.save(model, r'LSTMmodel.pth')
            break

    pred = []
    real = []
    model.eval()
    with torch.no_grad():
        for data_input, data_target in test_dataloader:
            data_input, data_target = data_input.to(device), data_target.to(device)
            out = model(data_input)
            pred.append(out.cpu().item())
            real.append(data_target.cpu().item())

    r2 = r2_score(real, pred)
    mse = mean_squared_error(real, pred)

    with plt.style.context(['science', "no-latex"]):
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(range(len(pred)), pred, label="Pred", color="red", linestyle='dashed')
        ax.plot(range(len(pred)), real, label="Real", color="black")
        ax.legend()
        ax.set(xlabel='Time')
        ax.set(ylabel='Rul')
        ax.autoscale(tight=True)
        fig.savefig('fig1.jpg', dpi=300)

        plt.show()
