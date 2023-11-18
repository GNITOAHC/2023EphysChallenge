# 2023EphysChallenge

## Pakage Usage

#### 參考網址
https://arxiv.org/pdf/1912.08278.pdf

我們採用卷積神經網絡（CNN）與量子神經網路（QNN）組合而成一個Hybird Model實作這個專案

1.Convolutional Neural Network（CNN）部分：
在這個專案中，使用了預訓練的ResNet18深度卷積神經網路模路
```
weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
model_hybrid = torchvision.models.resnet18(weights=weights)
```
```
ResNet18（Residual Network with 18 layers）是一種深度神經網路結構
使用這個模型可以解決深度神經網路中的梯度消失和梯度爆炸等問題
ResNet 通過引入殘差塊（Residual Blocks）的結構，使得模型能夠更容易地進行訓練
```

2.QNN（Quantum Neural Network）的部分

以下為實作步驟

(1)引入套件和框架： 首先引入 Pennylane 和 PyTorch 等框架，Pennylane 用於搭建和執行量子電路，而 PyTorch 用於建構深度學習模型。

(2)定義量子電路的層： 實現量子電路的基本組件，包括 Hadamard 閘、RY 旋轉閘和 CNOT 交織閘。這些組件將用於構建變分量子電路。

(3)定義 QNode： 創建一個 QNode 函數，該函數表示變分量子電路的運算流程。這包括將權重重塑、初始化 Hadamard 閘、對輸入特徵應用 RY 旋轉閘、構建可訓練的交織層和旋轉層，最後計算期望值。

(4)混合式模型中的 QNN： 創建一個混合式模型，該模型包含了一個預處理層、QNN 部分和一個後處理層。預處理層負責降維，QNN 部分進行量子計算，後處理層生成預測結果。

(5)訓練結果評估： 最後，通過在目標域數據上進行性能評估，確定模型在目標任務上的表現。這可以包括損失率(loss)、精確度(acc)等指標

## Development OS
```
MacOS
Windows 11
```

## Hardware
##### Only CPU
```
Mac M1
AMD Ryzen 5600U
```

## Run Code
##### Only CPU
>Terminal
```
python model.py
```
