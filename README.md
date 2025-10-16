
````markdown
# NKUST_-HW1: MLP 與 SVM 訓練過程的比較與實作

本專案對比 **多層感知器 (MLP)** 與 **支持向量機 (SVM)** 的訓練過程，並深入探討機器學習中的**梯度下降 (Gradient Descent)** 最佳化原理。

---

## 1. 核心數學原理：梯度下降 (Gradient Descent)

所有參數更新的核心公式皆為**迭代最佳化**，旨在找到權重向量 $\vec{w}$ 使得損失函數 $L(\vec{w})$ 最小化：

$$\vec{w}^{*} = \vec{w} + \Delta \vec{w}$$

其中，參數調整量 $\Delta \vec{w}$ 依賴於學習率 $\eta$ 和損失函數的負梯度 $\nabla L(\vec{w})$：
$$\Delta \vec{w} = - \eta \nabla L(\vec{w})$$

---

## 2. MLP 訓練過程：非凸優化與反向傳播 (mlp.py)

MLP 的訓練是透過**反向傳播 (Backpropagation)** 演算法來高效計算複雜複合函數的梯度。

### 數學推導 (核心：鏈式法則)

*   **目標函數：** 最小化交叉熵損失 $L$。
*   **梯度計算：** 透過**鏈式法則**，計算 $\frac{\partial L}{\partial \vec{W}^{[l]}}$。
*   **程式中 $\vec{w}^{*} = \vec{w} + \Delta \vec{w}$ 的實現：**
    *   `loss.backward()`：執行反向傳播，自動計算 $\nabla L$。
    *   `optimizer.step()`：根據 $\nabla L$ 和優化器（如 Adam）的公式更新 $\vec{w}$。

### Python 執行結果 (單一 Epoch 驗證機制)

這證明了 PyTorch 的自動微分與優化器機制正確運作。

```bash
PS E:\SVM_MLP_traing> python .\mlp.py
開始訓練一個 Epoch...
--------------------
Loss: 0.7474  # 損失值，證明計算正確
--------------------
訓練完成。
````

-----

## 3\. SVM 訓練過程：軟間隔與次梯度下降 (svm.py)

本實作採用了**軟間隔 SVM** 的 **原始問題** 形式，透過 **Hinge Loss (鉸鏈損失)** 函數，並使用 **隨機梯度下降 (SGD)** 來求解。

### 數學推導 (核心：Hinge Loss 次梯度)

  * **目標函數 $J(\vec{w})$：**
    $$J(\vec{w}, b) = \frac{1}{2} ||\vec{w}||^2 + C \sum_{i=1}^{N} \max(0, 1 - y_i (\vec{w}^T \vec{x}_i + b))$$
  * **梯度 $\nabla J(\vec{w})$ (次梯度)：** 根據數據點是否滿足間隔條件（$y_i (\vec{w}^T \vec{x}_i + b) \geq 1$）分段計算梯度 $\nabla J$。
      * 程式中的 `if/else` 邏輯精確地實現了這一分段次梯度計算。

### Python 執行結果 (最終權重 W, b 輸出)

證明手動實現的 SGD 成功找到最佳化後的權重和偏置。

```bash
PS E:\SVM_MLP_traing> python .\svm.py
SVM 訓練完成！
最終權重 W: [-1.4312307   1.82241949]  # 決策邊界的法向量
最終偏置 b: -1.7000000000000004
```

-----

## 4\. 總結與對比 (Conclusion)

| 特性 | MLP (多層感知器) | SVM (支持向量機) |
| :--- | :--- | :--- |
| **目標問題** | 高度非凸優化 | 軟間隔凸優化 |
| **梯度方法** | 反向傳播 (Backpropagation) | 次梯度下降 (Subgradient Descent) |
| **Python 實現** | 框架自動微分 | 手動實現 `if/else` 邏輯 |

兩種方法都成功地通過迭代 $\vec{w}^{*} = \vec{w} - \eta \nabla L(\vec{w})$ 找到了符合各自模型訓練目標的最佳參數。

```
```
