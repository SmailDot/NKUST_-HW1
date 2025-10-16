# NKUST_-HW1: MLP 與 SVM 訓練過程的比較與實作

本專案在對比 **多層感知器 (MLP)** 與 **支持向量機 (SVM)** 的訓練過程，並深入探討機器學習中的**梯度下降 (Gradient Descent)** 最佳化原理。

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
