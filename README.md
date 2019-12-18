# 银行离职人员预测模型

![Bank Customers](https://gss0.baidu.com/94o3dSag_xI4khGko9WTAnF6hhy/zhidao/pic/item/c8ea15ce36d3d539ee64cf8d3787e950352ab027.jpg "Bank Customers")

该项目可以预测客户何时可能离开银行。 它使用Keras和Tensorflow以及Kaggle数据集。
离开银行的客户可能对业务非常不利，特别是因为可以避免这种情况。该项目使用神经网络和10,000个客户信息的数据集来预测客户是否要离开银行。
此模型旨在帮助指导人们建立自己的等效网络。它尚未准备好进行部署，应该进一步个性化和完善。

## 实作

要使用该项目，请执行kehu_jianmo程序

```
python kehu_jianmo.py
````

  

## 使用Keras和Tensorflow构建神经网络

## 概观 

- Keras是Tensorflow的扩展程序，它提供了良好的错误输出消息。
- 该网络定义如下:
- 顺序神经网络
- 该代码库使用“密集”功能（classifier.add（dense ..）
- ADAM超参数 (Hyper Parameters) 可降低损失函数
- 因为我们才有两个输出值所以我们可以使用 `Binary_crossentropy` 
- 对于两个或多个值，请使用 `categorical_crossentropy`

## 网络汇总

1. 11 `输入参数`  
2. 6 神经元 `L1`  
3. 6 神经元 `L2`   
4. 1 `输出`  



`pd.get_dummies` 将文本字段转换为数字。例如

> `Spain`   = 1 0 0   
> `Germany` = 0 1 0  
> `France`  = 0 0 1

`drop_first=True` 表示删除了第一个字段，即SPAIN. .剩余的数据可以推断出该删除的字段


# 资料翻译

| CreditScore        | Geography           | Gender  | Age  | Tenure  | Balance  | NumOfProducts  | HasCrCard  |IsActiveMember  | EstimatedSalary  | Exited  |
| ------------- |-------------| -----|-------------| -------------| -------------| -------------| -------------| -------------| -------------| -------------| 
| 信用评分     | 地点| 性别 | 年龄 | 任期 | 银行存款余额 | 产品数量 | 有信用卡吗？ | 是活跃会员吗？ | 估计薪水 | 离开银行了吗 |


```
#----------------------------------------------------------------------------------------------
#                                               导入库
#----------------------------------------------------------------------------------------------

# 导入Numpy库
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 导入Keras库和包
import keras
from keras.models import Sequential
from keras.layers import Dense

# 将数据集分为训练集和测试集
from sklearn.model_selection import train_test_split

# 特征缩放 （正态分布）从－1到1 
from sklearn.preprocessing import StandardScaler

# 制作混淆矩阵
from sklearn.metrics import confusion_matrix,accuracy_score

```
  

```
#----------------------------------------------------------------------------------------------
#                                               数据预处理
#----------------------------------------------------------------------------------------------
# 导入数据集
dataset = pd.read_csv('BankCustomers.csv')
X = dataset.iloc[:, 3:13] # 全表
y = dataset.iloc[:, 13]   # 输出值

print('打印表预览')
print(X.head())
print('')
print('客户数量是：')
print(len(y))
print('')

# 将分类特征转换为虚拟变量
states=pd.get_dummies(X['Geography'],drop_first=True)
gender=pd.get_dummies(X['Gender'],drop_first=True)

# 连接其余的虚拟变量列
X=pd.concat([X,states,gender],axis=1)

# 删除列，因为不再需要

X=X.drop(['Geography','Gender'],axis=1)


#dataset.iloc[:, 3:13].values
print(states.head())
```
 
## 输出值 

```
打印表预览
   CreditScore Geography  Gender  Age  Tenure    Balance  NumOfProducts  \
0          619    France  Female   42       2       0.00              1   
1          608     Spain  Female   41       1   83807.86              1   
2          502    France  Female   42       8  159660.80              3   
3          699    France  Female   39       1       0.00              2   
4          850     Spain  Female   43       2  125510.82              1   

   HasCrCard  IsActiveMember  EstimatedSalary  
0          1               1        101348.88  
1          0               1        112542.58  
2          1               0        113931.57  
3          0               0         93826.63  
4          1               1         79084.10  

客户数量是：
10000
```


```
# 将数据集分为训练集和测试集
# X = 全表
# y = 输出值
feature_train, feature_test, label_train, label_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# 特征缩放 （正态分布）从－1到1 
sc = StandardScaler()
feature_train = sc.fit_transform(feature_train)
feature_test = sc.transform(feature_test)
```
  
```
#----------------------------------------------------------------------------------------------
#                                    定义和训练神经网络
#----------------------------------------------------------------------------------------------

# 初始化人工神经网络
classifier = Sequential()

# 添加输入层和第一个隐藏层
classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))

# 添加第二个隐藏层
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))

# 添加输出层
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

# 编译神经网络
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# 将ANN拟合到训练集
classifier.fit(feature_train, label_train, batch_size = 10, nb_epoch = 100)
```

```
#----------------------------------------------------------------------------------------------
#                                    准确性和混淆矩阵
#----------------------------------------------------------------------------------------------

# 预测测试集结果
label_pred = classifier.predict(feature_test)
label_pred = (label_pred > 0.5) # FALSE/TRUE depending on above or below 50%

cm = confusion_matrix(label_test, label_pred)  
accuracy=accuracy_score(label_test,label_pred)
```

```
print(classifier.summary())
```

## 输出值

```
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_4 (Dense)              (None, 6)                 72        
_________________________________________________________________
dense_5 (Dense)              (None, 6)                 42        
_________________________________________________________________
dense_6 (Dense)              (None, 1)                 7         
=================================================================
Total params: 121
Trainable params: 121
Non-trainable params: 0
_________________________________________________________________
None
```

# 准确性和混淆矩阵

```
print(cm)
print(accuracy)
```


## 致谢

数据集可以在kaggle上找到, [这里](https://www.kaggle.com/demohit/predict-your-customer-will-leave-bank/data)

