# 多層感知機 (MLP) 訓練過程推導：持續改進的核心

多層感知機（MLP）是深度學習的基礎。其「持續改進」的機制，完全依賴於**梯度下降**和**反向傳播**這兩個數學優化過程。

## 1. 核心流程：前向傳播 (Forward Propagation)

這是模型進行預測的過程。數據從輸入層依序傳遞到輸出層。

$$
\text{Net}_j = \sum_{i=1}^n (w_{ij} \cdot x_i) + b_j
$$
$$
\text{Output}_j = f(\text{Net}_j)
$$

其中：
*   $w_{ij}$：從上一層節點 $i$ 到本層節點 $j$ 的權重 (Weight)。
*   $x_i$：上一層的輸出值。
*   $b_j$：節點 $j$ 的偏差 (Bias)。
*   $f(\cdot)$：激活函數 (如 ReLU)。

## 2. 評估：損失函數 (Loss Function)

模型預測後，需要量化預測的錯誤程度。在分類任務中，我們通常使用**交叉熵損失 (Cross-Entropy Loss)**。

$$
L(\hat{y}, y) = - \sum_{k} y_k \cdot \log(\hat{y}_k)
$$

其中 $y$ 是真實標籤，$\hat{y}$ 是模型的預測機率。**最小化 $L$ 就是 AI 的目標。**

## 3. 核心機制：反向傳播 (Backpropagation)

這是實現「改進」的關鍵步驟。它的目的是計算損失函數 $L$ 對**每一個權重 $w_{ij}$ 的梯度**（即偏導數）。

**目標：** 計算 $\frac{\partial L}{\partial w_{ij}}$

這個過程從輸出層開始，利用**鏈式法則 (Chain Rule)** 逐層將錯誤信號（梯度）傳播回輸入層。

$$
\frac{\partial L}{\partial w_{ij}} = \frac{\partial L}{\partial \text{Net}_j} \cdot \frac{\partial \text{Net}_j}{\partial w_{ij}}
$$

這個梯度 $\frac{\partial L}{\partial w_{ij}}$ 告訴我們，如果 $w_{ij}$ 增加，損失 $L$ 會如何變化。

## 4. 持續改進：權重更新 (Weight Update / Optimization)

有了梯度後，我們就可以使用**梯度下降法 (Gradient Descent)** 來調整權重，實現「持續改進」。

$$
w_{ij}^{\text{new}} = w_{ij}^{\text{old}} - \text{LR} \cdot \frac{\partial L}{\partial w_{ij}}
$$

*   $\text{LR}$：**學習率 (Learning Rate)**，決定了改進的步長。
*   **每一次**執行這個更新公式，模型就完成了一次**參數調整 (改進)**。
*   將整個數據集重複訓練多輪（Epochs），就是**持續改進**的整個過程。

---
**總結：** MLP 的訓練是通過不斷重複**前向傳播**來發現錯誤，再通過**反向傳播**計算錯誤來源，最後用**梯度下降**調整參數的**迭代循環**。