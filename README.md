非常樂意協助您完成這項任務。作為 AI 助手，我無法直接存取您的電腦或 GitHub 帳號來執行上傳操作，但我將為您提供完整的 **Terminal 指令**和一份專業的 **`README.md` 內容**。

請您按照以下步驟在您的電腦上執行操作：

### 步驟一：Git 指令操作 (將檔案上傳至 GitHub)

請在您的專案目錄 `E:\SVM_MLP_traing` 下打開您的 **Git Bash** 或 **PowerShell** 終端機，依序執行以下指令：

| 指令 | 說明 |
| :--- | :--- |
| **`cd E:\SVM_MLP_traing`** | 進入您的專案資料夾。 |
| **`git init`** | (如果資料夾是新的) 初始化為 Git 儲存庫。 |
| **`git remote add origin https://github.com/SmailDot/NKUST_-HW1.git`** | 告訴 Git 遠端儲存庫的地址。 |
| **`(請在此處創建 README.md 檔案)`** | 將步驟二提供的內容貼入名為 `README.md` 的檔案中。 |
| **`git add .`** | 將目前目錄下所有新建立/修改的檔案 (`mlp.py`, `svm.py`, `README.md`) 加入暫存區。 |
| **`git commit -m "feat: Finish HW1 - MLP and SVM Training Derivations"`** | 提交變更，並附上本次提交的說明。 |
| **`git push -u origin master`** | 將本地的變更推送到遠端儲存庫的 `master` 分支（如果您的主分支是 `main`，請將 `master` 改為 `main`）。 |
| **(輸入您的 GitHub 帳號/密碼或 Token)** | 完成上傳。 |

### 步驟二：`README.md` 檔案內容 (Markdown 格式)

請建立一個名為 `README.md` 的純文字檔案，並將以下 Markdown 內容完整複製貼上。這份文件詳細介紹了您的作業內容、數學推導和實作結果。

````markdown
# NKUST_-HW1: MLP 與 SVM 訓練過程的比較與實作

本專案旨在對比 **多層感知器 (MLP)** 與 **支持向量機 (SVM)** 的訓練過程，並深入探討機器學習中的**梯度下降 (Gradient Descent)** 最佳化原理。

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
