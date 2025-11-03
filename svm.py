import numpy as np

# 假設 data 和 labels (y_i 必須是 -1 或 1)
# X (N samples, D features), y (N samples)

def train_svm_sgd(X, y, learning_rate=0.01, C=1.0, epochs=100):
    num_samples, num_features = X.shape
    # 初始化 w (D 維向量) 和 b (偏置)
    w = np.zeros(num_features)
    b = 0

    for epoch in range(epochs):
        # 遍歷所有數據點 (Stochastic Gradient Descent)
        for i in range(num_samples):
            xi = X[i]
            yi = y[i]

            # 計算梯度 (次梯度)
            # 決策函數：yi * (w^T * xi + b)
            decision = yi * (np.dot(w, xi) + b)

            # --- 這是數學推導的核心實作 ---
            if decision >= 1:
                # 情況一：已正確分類且在間隔之外 (Loss = 0)
                dw = w
                db = 0
            else:
                # 情況二：在間隔內或誤判 (Loss > 0)
                # J'(w) = w - C * y_i * x_i
                dw = w - C * yi * xi
                db = -C * yi # 對偏置 b 的梯度

            # === w* = w + Delta_w 的實作 (梯度下降) ===
            # w_new = w_old - eta * dw
            w -= learning_rate * dw
            b -= learning_rate * db
            
    return w, b

if __name__ == '__main__':
    # 產生測試數據
    np.random.seed(0)
    X_test = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2], [1, 1]])
    y_test = np.array([1, 1, 1, -1, -1, -1]) 
    
    # 呼叫訓練函數
    w_final, b_final = train_svm_sgd(X_test, y_test, learning_rate=0.01, C=10.0, epochs=100)
    
    # 輸出結果
    print("SVM 訓練完成！")
    print(f"最終權重 W: {w_final}")
    print(f"最終偏置 b: {b_final}")