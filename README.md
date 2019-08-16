# Breast_Cancer_Detection
乳腺癌识别，2019年毕业设计

```
data  
│
└───40
│   └───test
│       └───B
│       └───M
│   └───valid
│       └───B
│       └───M
└───100
│   └─── ...
│
└───200
│   └─── ...
│
└───400
    └─── ...
```
`requires tensorflow >= 1.13`


data：该目录包含了训练所需的图像数据文件，其中不同放大倍数(40×、100×、200×、400×)的训练集、验证集如上所示的层次结构进行存储。下载地址：<https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/>

log：该目录用来保存训练结果。

handle.py：该文件根据数据存储的路径，将数据的标签进行提取。

alexnet.py：该文件定义了AlexNet的结构，以供main.py使用。

main.py：该文件定义了网络结构，并将训练数据导入后对网络进行训练，训练结束后保存模型并对模型进行测试。

predict.py：训练完成后对部分样本进行预测。
