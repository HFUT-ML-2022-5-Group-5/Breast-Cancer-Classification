#任务二：神经网络

# 预测乳腺癌,使用pandas对数据集合进行加载
import pandas as pd

import numpy as np
df = pd.read_csv("./breast_cancer2.csv")
print(df)

# 每行数据有30个乳腺病理特征, 最后一列表示是否患有乳腺癌
X = df[df.columns[0:-1]].values
Y = df[df.columns[-1]].values
print(X.shape,Y.shape)


from sklearn.preprocessing import StandardScaler
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
Clus_dataSet



from sklearn.model_selection import train_test_split
# 按照0.8 和 0.2 的比例随机划分数据集合
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,
                             test_size=0.2,random_state=17511)
print(f'X_train.shape={X_train.shape}')
print(f'Y_train.shape={Y_train.shape}')
print(f'X_test.shape={X_test.shape}')
print(f'Y_test.shape={Y_test.shape}')

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# 对特征进行标准化
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)



import torch
import numpy as np

X_train = torch.from_numpy(X_train.astype(np.float32))
Y_train = torch.from_numpy(Y_train.astype(np.float32))

X_test = torch.from_numpy(X_test.astype(np.float32))
Y_test = torch.from_numpy(Y_test.astype(np.float32))

# 将标记集合 Y_train 和 Y_test 转成2维
Y_train = Y_train.view(Y_train.shape[0],1)
Y_test = Y_test.view(Y_test.shape[0],1)
print(Y_train.size(),Y_test.size())


# 自定义模型
class MyModel(torch.nn.Module):
    def __init__(self,in_features):
        super(MyModel,self).__init__()   #调用父类的构造函数！
        # 搭建自己的神经网络
        # 1.构建线性层
        self.linear = torch.nn.Linear(in_features,1)
        # 2.构建激活函数层
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        """重写了父类的forward函数，正向传播"""
        pred = self.linear(x)
        out = self.sigmoid(pred)
        return out


# 损失函数公式定义
loss = torch.nn.BCELoss()

# 学习率，迭代次数
learning_rate = 0.1
num_epochs = 100

# 获取样本量和特征数，创建模型
n_samples, n_features = X.shape
model = MyModel(n_features)

# 创建优化器，
optimizer = torch.optim.SGD(
    model.parameters(), lr=learning_rate)

# 打印模型、打印模型参数
print(model)
print(list(model.parameters()))

threshold_value = 0.00001
for epoch in range(num_epochs):
    y_pred = model(X_train)  # 正向传播，调用forward()方法
    ls = loss(y_pred,Y_train)   # 计算损失（标量值）
    ls.backward()     # 反向传播
    optimizer.step()     # 更新权重
    optimizer.zero_grad()    # 清空梯度
    if epoch%5 == 0:
        print(f"epoch:{epoch},loss={ls.item():.4f}")
    if ls.item() <= threshold_value:
        break;
print("模型训练完成! loss={0}".format(ls))

with torch.no_grad():       # 无需向后传播（非训练过程）
    y_pred = model(X_test)
    # 上面计算出来的结果是0-1之间的数,将数据进行四舍五入,得到0或1
    y_pred_cls = y_pred.round()
    # 统计结果
acc = y_pred_cls.eq(Y_test).sum().numpy()/ float(Y_test.shape[0])
print(f"准确率:{acc.item():.4f}")
