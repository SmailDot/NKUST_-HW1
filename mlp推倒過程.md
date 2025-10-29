# 多層感知器 (Multi-Layer Perceptron, MLP) 訓練推導

MLP 是一種前饋神經網路，其訓練的核心在於**反向傳播 (Backpropagation)** 演算法。這個演算法本質上是使用**梯度下降 (Gradient Descent)** 來最小化損失函數 $E$，而反向傳播就是一種高效計算梯度 $\nabla E$ 的方法，它巧妙地應用了微積分的**鏈式法則 (Chain Rule)**。

### 1. 符號定義

為簡潔起見，我們以一個具有**單一隱藏層**的 MLP 為例 (這足以說明反向傳播的核心思想)：

* $x$: 輸入向量 (來自輸入層)。
* $W^{(1)}$: 從輸入層到隱藏層的權重矩陣。$W_{ij}^{(1)}$ 表示從輸入單元 $i$ 到隱藏單元 $j$ 的權重。
* $b^{(1)}$: 隱藏層的偏置 (bias) 向量。
* $z^{(1)}$: 隱藏層的**加權輸入** (weighted input)，$z^{(1)} = W^{(1)T}x + b^{(1)}$。
* $\sigma(\cdot)$: 隱藏層的**活化函數** (activation function)，例如 Sigmoid 或 ReLU。
* $a^{(1)}$: 隱藏層的**輸出** (activation)，$a^{(1)} = \sigma(z^{(1)})$。
* $W^{(2)}$: 從隱藏層到輸出層的權重矩陣。$W_{jk}^{(2)}$ 表示從隱藏單元 $j$ 到輸出單元 $k$ 的權重。
* $b^{(2)}$: 輸出層的偏置向量。
* $z^{(2)}$: 輸出層的加權輸入，$z^{(2)} = W^{(2)T}a^{(1)} + b^{(2)}$。
* $\sigma_{\text{out}}(\cdot)$: 輸出層的活化函數 (例如 Softmax 或 Sigmoid)。
* $\hat{y}$: 模型的預測輸出，$a^{(2)} = \hat{y} = \sigma_{\text{out}}(z^{(2)})$。
* $y$: 實際的目標標籤。
* $E$: 損失函數 (Loss Function)。例如均方誤差 (MSE): $E = \frac{1}{2} \sum_k (y_k - \hat{y}_k)^2$。
* $\eta$: 學習率 (learning rate)。

### 2. 訓練目標

目標是找到一組 $W^{(1)}, b^{(1)}, W^{(2)}, b^{(2)}$，使得損失函數 $E$ 最小。我們使用梯度下降法更新參數：

$$
W \leftarrow W - \eta \frac{\partial E}{\partial W}
$$

所以，關鍵任務是計算 $E$ 對**每一層**權重和偏置的偏微分 (梯度)。

### 3. 前向傳播 (Forward Propagation)

在計算梯度之前，必須先執行一次前向傳播，計算出預測值 $\hat{y}$ 並得到所有中間變數 ($z^{(1)}, a^{(1)}, z^{(2)}$) 的值。

1.  **隱藏層計算：**
    $z^{(1)} = W^{(1)T}x + b^{(1)}$
    $a^{(1)} = \sigma(z^{(1)})$

2.  **輸出層計算：**
    $z^{(2)} = W^{(2)T}a^{(1)} + b^{(2)}$
    $\hat{y} = \sigma_{\text{out}}(z^{(2)})$

3.  **計算損失：**
    $E = L(y, \hat{y})$

### 4. 反向傳播 (Backward Propagation)

從最後一層開始，使用鏈式法則，將「誤差」逐層反向傳遞。

#### 4.1 輸出層的梯度 ( $W^{(2)}$ 和 $b^{(2)}$ )

想計算 $E$ 對 $W_{jk}^{(2)}$ (連接隱藏單元 $j$ 到輸出單元 $k$ 的權重) 的梯度：

$$
\frac{\partial E}{\partial W_{jk}^{(2)}} = \frac{\partial E}{\partial \hat{y}_k} \cdot \frac{\partial \hat{y}_k}{\partial z_k^{(2)}} \cdot \frac{\partial z_k^{(2)}}{\partial W_{jk}^{(2)}}
$$

分解這三項：

1.  $\frac{\partial E}{\partial \hat{y}_k}$: 損失函數對預測輸出的偏導。
    * (若 $E$ 為 MSE: $\frac{\partial E}{\partial \hat{y}_k} = (\hat{y}_k - y_k)$)
2.  $\frac{\partial \hat{y}_k}{\partial z_k^{(2)}}$: 輸出層活化函數的微分，$\sigma_{\text{out}}'(z_k^{(2)})$。
3.  $\frac{\partial z_k^{(2)}}{\partial W_{jk}^{(2)}}$:
    * 因為 $z_k^{(2)} = \sum_j W_{jk}^{(2)} a_j^{(1)} + b_k^{(2)}$
    * 所以 $\frac{\partial z_k^{(2)}}{\partial W_{jk}^{(2)}} = a_j^{(1)}$ (即隱藏層 $j$ 單元的輸出)。

為了簡化表達式，定義輸出層的**誤差項 (delta)** $\delta_k^{(2)}$：

$$
\delta_k^{(2)} \equiv \frac{\partial E}{\partial z_k^{(2)}} = \frac{\partial E}{\partial \hat{y}_k} \cdot \frac{\partial \hat{y}_k}{\partial z_k^{(2)}}
$$

* (對於 MSE：$\delta_k^{(2)} = (\hat{y}_k - y_k) \cdot \sigma_{\text{out}}'(z_k^{(2)})$)

將 $\delta_k^{(2)}$ 代回原式，我們得到 $W^{(2)}$ 的梯度：

$$
\frac{\partial E}{\partial W_{jk}^{(2)}} = \delta_k^{(2)} \cdot a_j^{(1)}
$$

對於偏置 $b_k^{(2)}$，因為 $\frac{\partial z_k^{(2)}}{\partial b_k^{(2)}} = 1$，所以：

$$
\frac{\partial E}{\partial b_k^{(2)}} = \delta_k^{(2)}
$$

#### 4.2 隱藏層的梯度 ( $W^{(1)}$ 和 $b^{(1)}$ )

接著，計算 $E$ 對 $W_{ij}^{(1)}$ (連接輸入單元 $i$ 到隱藏單元 $j$ 的權重) 的梯度：

$$
\frac{\partial E}{\partial W_{ij}^{(1)}} = \left( \sum_k \frac{\partial E}{\partial z_k^{(2)}} \cdot \frac{\partial z_k^{(2)}}{\partial a_j^{(1)}} \right) \cdot \frac{\partial a_j^{(1)}}{\partial z_j^{(1)}} \cdot \frac{\partial z_j^{(1)}}{\partial W_{ij}^{(1)}}
$$

* **關鍵步驟：** 隱藏單元 $j$ 的輸出 $a_j^{(1)}$ 會影響**所有**的輸出單元 $k$ (通過 $z_k^{(2)}$)。因此，我們必須將所有來自輸出層的誤差加總 ($\sum_k$)。

再次分解這個鏈式法則：

1.  $\frac{\partial E}{\partial z_k^{(2)}}$: 這就是剛才定義的 $\delta_k^{(2)}$。
2.  $\frac{\partial z_k^{(2)}}{\partial a_j^{(1)}}$:
    * 因為 $z_k^{(2)} = \sum_j W_{jk}^{(2)} a_j^{(1)} + b_k^{(2)}$
    * 所以 $\frac{\partial z_k^{(2)}}{\partial a_j^{(1)}} = W_{jk}^{(2)}$ (即連接 $j$ 和 $k$ 的權重)。
3.  $\frac{\partial a_j^{(1)}}{\partial z_j^{(1)}}$: 隱藏層活化函數的微分，$\sigma'(z_j^{(1)})$。
4.  $\frac{\partial z_j^{(1)}}{\partial W_{ij}^{(1)}}$:
    * 因為 $z_j^{(1)} = \sum_i W_{ij}^{(1)} x_i + b_j^{(1)}$
    * 所以 $\frac{\partial z_j^{(1)}}{\partial W_{ij}^{(1)}} = x_i$ (即輸入 $i$)。

定義隱藏層的**誤差項** $\delta_j^{(1)}$：

$$
\delta_j^{(1)} \equiv \frac{\partial E}{\partial z_j^{(1)}} = \left( \sum_k \frac{\partial E}{\partial z_k^{(2)}} \cdot \frac{\partial z_k^{(2)}}{\partial a_j^{(1)}} \right) \cdot \frac{\partial a_j^{(1)}}{\partial z_j^{(1)}}
$$
$$
\delta_j^{(1)} = \left( \sum_k \delta_k^{(2)} W_{jk}^{(2)} \right) \cdot \sigma'(z_j^{(1)})
$$

* **核心洞察：** 隱藏層的誤差 $\delta^{(1)}$，可以由下一層（輸出層）的誤差 $\delta^{(2)}$ 乘以它們之間的權重 $W^{(2)}$，再乘以本地的活化函數微分 $\sigma'$ 得到。這就是「反向傳播」這個名字的由來。

現在，可以計算 $W^{(1)}$ 的梯度：

$$
\frac{\partial E}{\partial W_{ij}^{(1)}} = \delta_j^{(1)} \cdot x_i
$$

對於偏置 $b_j^{(1)}$，因為 $\frac{\partial z_j^{(1)}}{\partial b_j^{(1)}} = 1$，所以：

$$
\frac{\partial E}{\partial b_j^{(1)}} = \delta_j^{(1)}
$$

### 5. 權重更新 (Gradient Descent)

計算出所有梯度後，使用梯度下降來更新所有參數：

$$
W_{jk}^{(2)} \leftarrow W_{jk}^{(2)} - \eta \frac{\partial E}{\partial W_{jk}^{(2)}} \quad \left( = W_{jk}^{(2)} - \eta \cdot \delta_k^{(2)} \cdot a_j^{(1)} \right)
$$
$$
b_k^{(2)} \leftarrow b_k^{(2)} - \eta \frac{\partial E}{\partial b_k^{(2)}} \quad \left( = b_k^{(2)} - \eta \cdot \delta_k^{(2)} \right)
$$
$$
W_{ij}^{(1)} \leftarrow W_{ij}^{(1)} - \eta \frac{\partial E}{\partial W_{ij}^{(1)}} \quad \left( = W_{ij}^{(1)} - \eta \cdot \delta_j^{(1)} \cdot x_i \right)
$$
$$
b_j^{(1)} \leftarrow b_j^{(1)} - \eta \frac{\partial E}{\partial b_j^{(1)}} \quad \left( = b_j^{(1)} - \eta \cdot \delta_j^{(1)} \right)
$$

這個過程會被重複執行 (多個 epochs)，直到模型的損失 $E$ 收斂到最小值。
