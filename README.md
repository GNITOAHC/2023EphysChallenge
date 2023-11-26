# 2023EphysChallenge

> [!WARNING]
> This repo is archived.

## Pakage Usage

#### 參考網址
https://arxiv.org/pdf/1912.08278.pdf

### 專案架構介紹

我們採用卷積神經網絡（CNN）與量子神經網路（QNN）組合而成一個Hybird Model實作這個專案

1.Convolutional Neural Network（CNN）部分：

在這個專案中，使用了預訓練的ResNet18深度卷積神經網路模型
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


3.使用Transfer Learning的優勢
這種機器學習方法，將其中一個任務學到的知識應用於解決另一個相關的任務
在傳統的機器學習中，模型通常是在特定的數據集上進行訓練，然後應用於相似的任務。然而，當數據集相對較小或者相對於模型的複雜性不足時，傳統的訓練方法可能會受到限制

因此我們選擇採用Transfer Learning實作

### 專案實作函式介紹

1.與CIFAR-100數據集相關的function：

```
def labels_to_filtered(labels):
    """Maps CIFAR labels (3,88) to the index of filtered_labels"""
    return [filtered_labels.index(label) for label in labels]
#依競賽規則取得index 3, 88訓練資料
```

```
相關函式
dataset_sizes：訓練集和驗證集的大小信息
dataloaders：初始化了訓練集和驗證集的 PyTorch
DataLoader，用於在訓練模型時加載數據
```

2.量子神經網絡（QNN）相關函式

用於定義量子神經網絡（Quantum Neural Network）的層次
以下使用 PennyLane 套件實現

```
#定義量子神經網絡層
H_layer：包含了一個層的單比特Hadamard閘口。
RY_layer：包含了一個層的帶參數的Bit且環繞繞y軸旋轉的閘。
entangling_layer：包含了一個層的CNOT閘口，結構是簡單的CNOT閘連接，交替奇偶index的閘。
```
## 專案demo結果
我們使用ResNet18模型(訓練時間最短、效果最佳)與以下參數，可得到95.5%的最佳準確率
以下為我們設定的最佳參數
```
n_qubits = 4                # Number of qubits
step = 0.0006               # Learning rate
batch_size = 4              # Number of samples for each training step
num_epochs = 10          # Number of training epochs
q_depth = 7                 # Depth of the quantum circuit (number of variational layers)
gamma_lr_scheduler = 0.1    # Learning rate reduction applied every 10 epochs.
q_delta = 0.01              # Initial spread of random quantum weights
start_time = time.time()    # Start of the computation timer
```
這組配置中，使用了
較小的學習率 (step)
較小的批次大小 (batch_size)
相對較大的量子電路深度 (q_depth)
以及較長的訓練周期 (num_epochs)。這樣的配置可能有助於模型更好地學習複雜的量子特徵，但也可能導致較長的訓練時間。不過我們在本專案中使用resnet18模型，因此可減少訓練時間，增加精準度與效能

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
