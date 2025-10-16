import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# --- 1. 模型定義 ---
# 建立一個簡單的兩層網路 (MLP)
class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super(SimpleMLP, self).__init__()
        # 這是我們要訓練的參數 W 和 b
        # nn.Linear 是一個線性層： y = W*x + b
        self.linear = nn.Linear(input_dim, 1) 
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 前向傳播過程
        return self.sigmoid(self.linear(x))

# --- 2. 準備數據與初始化 ---
INPUT_DIM = 10
N_SAMPLES = 100
LEARNING_RATE = 0.01

# 模擬隨機數據 (X) 和標籤 (y)
X = torch.randn(N_SAMPLES, INPUT_DIM) 
y = torch.randint(0, 2, (N_SAMPLES, 1)).float() 

# 實例化模型
model = SimpleMLP(INPUT_DIM)

# --- 3. 修復 NameError：定義損失函數和優化器 ---
# 損失函數：二元交叉熵損失 (用於分類)
criterion = nn.BCELoss() 

# 優化器：將模型參數傳遞給 Adam，並設定學習率 (eta)
# *** 這裡 'optimizer' 物件被定義 ***
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) 

# --- 4. 訓練迴圈 (核心實作) ---
print("開始訓練一個 Epoch...")
print("-" * 20)

# 在訓練迴圈中，每一次迴圈都實現了一次 w* = w + Delta_w 的更新

for epoch in range(1):
    
    # [步驟 1: 清除舊梯度]
    # 清除上一步計算的梯度 dL/dW，防止累積
    optimizer.zero_grad() 
    
    # [步驟 2: 前向傳播] (計算 y_hat)
    output = model(X) 
    
    # [步驟 3: 計算損失 L]
    loss = criterion(output, y)
    
    # ----------------------------------------------------
    # --- 數學推導核心：反向傳播 (計算 dL/dW) ---
    # loss.backward() 實現了鏈式法則 (反向傳播)
    # 框架將計算出的梯度 (nabla L) 存到參數的 .grad 屬性中
    loss.backward() 
    
    # --- 應用更新：w* = w + Delta_w 的實作 ---
    # optimizer.step() 讀取 .grad 屬性 (即 -Delta_w/eta)，
    # 並根據優化器公式更新模型參數 W 和 b (實現 w* = w - eta * nabla L)
    optimizer.step()
    # ----------------------------------------------------
    
    print(f"Loss: {loss.item():.4f}")

print("-" * 20)
print("訓練完成。")