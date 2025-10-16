# NKUST_-HW1: MLP 與 SVM 訓練過程的對比與實作

對比 **多層感知器 (MLP)** 與 **支持向量機 (SVM)** 的訓練過程，深入探討機器學習中的**梯度下降 (Gradient Descent)** 最佳化原理。

---

## 1. 核心原理：梯度下降與山谷漫步 (Gradient Descent)

### 1.1 核心概念：山谷漫步的哲學 (由淺入深)

我們可以將**損失函數 $L(\vec{w})$** 想像成一座凹凸不平的「山谷」，而模型的參數 $\vec{w}$ 則是我們在山谷中的**位置**。訓練的目標就是找到山谷的**最低點** (即最小化 $L(\vec{w})$)。

*   **參數更新公式：** 每次更新就是我們邁出的一小步。
    $$\vec{w}^{*} = \vec{w} + \Delta \vec{w}$$
*   **調整量 $\Delta \vec{w}$：** 這一小步的方向和大小由**梯度 (Gradient)** 決定。梯度 $\nabla L(\vec{w})$ 指向山坡最陡峭的**上升**方向，因此我們必須朝著它的**反方向**走。
    $$\Delta \vec{w} = - \eta \nabla L(\vec{w})$$
    ($\eta$ 是學習率，控制步長。)

### 1.2 視覺化：損失曲面上的下降路徑

<img width="429" height="408" alt="94Uog" src="https://github.com/user-attachments/assets/952c2534-36fb-4b70-97e9-9530b389e343" />

---

## 2. MLP 訓練過程：非凸優化與反向傳播 (mlp.py)

MLP 是一個複合函數。要找到深層參數的梯度，必須將輸出層的誤差逐層向後分配。

### 數學推導 (核心：鏈式法則)

*   **目標函數：** 最小化交叉熵損失 $L$。
*   **核心機制：** **反向傳播 (Backpropagation)**。它是一種高效應用微積分**鏈式法則**的演算法，將 $\nabla L$ 倒推回每一層權重 $\vec{W}^{[l]}$。
*   **更新實現：** 在 PyTorch 等框架中，此複雜推導被自動化。

### Python 實作 $\vec{w}^{*} = \vec{w} + \Delta \vec{w}$

| 步驟 | 程式碼實現 | 說明 |
| :--- | :--- | :--- |
| **計算 $\Delta \vec{w}$** | `loss.backward()` | 執行反向傳播，自動計算 $\nabla L$ (即 $-\Delta \vec{w}/\eta$)。 |
| **應用 $\vec{w}^{*} = \vec{w} + \Delta \vec{w}$** | `optimizer.step()` | 讀取梯度，並根據優化器公式（例如 Adam）更新 $\vec{w}$ 的值。**此行即是公式的實現。** |

### Python 執行結果 (單一 Epoch 驗證機制)

```bash
PS E:\SVM_MLP_traing> python .\mlp.py
開始訓練一個 Epoch...
--------------------
Loss: 0.7474  # 證明前向/反向傳播迴圈正確
--------------------
訓練完成。
````

-----

## 3\. SVM 訓練過程：軟間隔與次梯度下降 (svm.py)

SVM 的核心是**最大化間隔 (Maximum Margin)**，這體現在目標函數的第一項上。

### 數學推導 (核心：Hinge Loss 次梯度)

  * **目標函數 $J(\vec{w})$：**
    $$J(\vec{w}, b) = \underbrace{\frac{1}{2} ||\vec{w}||^2}_{\text{最大化間隔}} + \underbrace{C \sum_{i=1}^{N} \max(0, 1 - y_i (\vec{w}^T \vec{x}_i + b))}_{\text{鉸鏈損失}}$$
  * **梯度 $\nabla J(\vec{w})$ (次梯度)：** 由於 Hinge Loss 不完全可微，使用**次梯度**分段計算 $\nabla J$。
    $$\nabla J(\vec{w}) = \begin{cases} \vec{w} & \text{if } y_i (\vec{w}^T \vec{x}_i + b) \geq 1 \\ \vec{w} - C y_i \vec{x}_i & \text{if } y_i (\vec{w}^T \vec{x}_i + b) < 1 \end{cases}$$

### Python 實作 $\vec{w}^{*} = \vec{w} + \Delta \vec{w}$

本實作手動編寫了 SGD 迴圈，清晰展示了 $\vec{w}$ 的更新。

```python
# --- 程式碼片段：實現次梯度計算與更新 ---

# 1. 梯度計算 (if/else 實現分段次梯度)
if decision >= 1:
    dw = w       
    # ...
else:
    # 僅在誤判或邊界內時，才加入懲罰項 - C * y_i * x_i
    dw = w - C * yi * xi 
    # ...

# 2. 應用更新： w* = w + Delta_w
w -= learning_rate * dw # w_new = w_old - eta * dw
# w 和 b 變數此時被更新為 w* 和 b*
```

### Python 執行結果 (最終權重 W, b 輸出)

```bash
PS E:\SVM_MLP_traing> python .\svm.py
SVM 訓練完成！
最終權重 W: [-1.4312307   1.82241949]  
最終偏置 b: -1.7000000000000004
```

-----

## 4\. 總結與對比 (Conclusion)

| 特性 | MLP (多層感知器) | SVM (支持向量機) |
| :--- | :--- | :--- |
| **損失函數** | 交叉熵 (Cross-Entropy) | 鉸鏈損失 (Hinge Loss) |
| **梯度方法** | 反向傳播 (Backpropagation) | 次梯度下降 (Subgradient Descent) |
| **問題性質** | 高度非凸優化 (容易局部最優) | 軟間隔凸優化 (GD 可找到全局最優) |
| **Python 實現** | 框架自動微分 | 手動實現 `if/else` 邏輯 |

兩種方法皆成功通過 $\vec{w}^{*} = \vec{w} - \eta \nabla L(\vec{w})$ 的迭代，解決了各自模型複雜的最佳化問題。
