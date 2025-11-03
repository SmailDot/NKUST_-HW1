import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import numpy as np

# --- 導入 SVM 相關庫 ---
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons # 用於 SVM 決策邊界可視化

# --- 1. 模型定義 (MLP - 多層感知機) ---
class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# --- 2. 數據載入 (共用) ---
@st.cache_data
def load_data():
    """載入並轉換 MNIST 數據集，使用 Streamlit cache 避免重複下載。"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    return train_dataset

# --- 3. MLP 訓練函數：核心在於實時更新可視化數據 (持續改進) ---
def train_model_and_visualize(num_epochs, lr, batch_size, placeholder_loss, placeholder_acc):
    
    # 設備設定：優先使用 GPU (CUDA)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.info(f"⚡️ 訓練設備: {device}. (MLP 模型)")

    train_dataset = load_data()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型、損失函數和優化器
    model = SimpleNN(input_size=28*28, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    accuracy_history = []
    
    # 設置繪圖區域
    fig_loss, ax_loss = plt.subplots(figsize=(5, 3))
    fig_acc, ax_acc = plt.subplots(figsize=(5, 3))
    
    st.markdown("---")
    st.subheader("🤖 MLP 模型 (深度學習) 持續改進過程 - 實時可視化")
    st.markdown("請觀察：**Loss 曲線** 應持續**下降**；**Accuracy 曲線** 應持續**上升**。")

    
    # 訓練循環
    for epoch in range(num_epochs):
        
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0

        for i, (images, labels) in enumerate(train_loader):
            
            # 準備數據並移動到正確的設備
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            
            # 1. 前向傳播 -> 2. 計算損失
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 3. 反向傳播 -> 4. 權重更新 (持續改進)
            optimizer.zero_grad() 
            loss.backward()       
            optimizer.step()      

            # 統計本輪數據
            epoch_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            epoch_total += labels.size(0)
            epoch_correct += (predicted == labels).sum().item()

            # 每隔 100 步更新一次可視化
            if (i+1) % 100 == 0:
                loss_history.append(loss.item())
                accuracy_history.append(100 * (predicted == labels).sum().item() / labels.size(0))

                # --- 實時更新可視化 ---
                ax_loss.clear(); ax_loss.plot(loss_history, color='red'); 
                ax_loss.set_title(f'Loss Curve (Epoch {epoch+1})'); ax_loss.set_xlabel('Steps (x100)'); ax_loss.set_ylabel('Loss Value');
                placeholder_loss.pyplot(fig_loss)

                ax_acc.clear(); ax_acc.plot(accuracy_history, color='blue');
                ax_acc.set_title(f'Accuracy Curve (Epoch {epoch+1})'); ax_acc.set_xlabel('Steps (x100)'); ax_acc.set_ylabel('Accuracy (%)');
                ax_acc.set_ylim([np.min(accuracy_history)-5 if accuracy_history and np.min(accuracy_history) > 0 else 80, 100])
                placeholder_acc.pyplot(fig_acc)
                time.sleep(0.01) 
        
        # --- Epoch 結束後的總結與說明 (展示改進結果) ---
        avg_loss = epoch_loss / epoch_total
        avg_accuracy = 100 * epoch_correct / epoch_total
        
        with st.expander(f"✅ **第 {epoch+1} 輪 MLP 訓練總結與分析**", expanded=True):
            st.metric(label="本輪平均損失 (Loss)", value=f"{avg_loss:.4f}")
            st.metric(label="本輪平均準確度 (Accuracy)", value=f"{avg_accuracy:.2f}%")
            
            st.markdown("---")
            st.markdown(f"**AI 學習進度說明 (第 {epoch+1} 輪):**")
            
            if avg_accuracy < 80:
                 st.warning(f"當前準確度低，AI 正在**基礎調整**。")
            elif avg_accuracy < 90:
                st.info(f"當前準確度為 {avg_accuracy:.2f}%，模型已進入**穩定改進階段**。")
            else:
                st.success(f"模型**改進效果顯著**。現在的挑戰是維持高準確度並避免過度擬合。")

        st.markdown("---")
        
    st.success("MLP 訓練完成！模型已停止改進。")


# --- 4. SVM 訓練對比函數 (一次性優化) ---

@st.cache_data
def load_data_for_svm():
    """載入並準備適合 SVM 的數據 (MNIST 數據集)"""
    train_dataset = datasets.MNIST('./data', train=True, download=True, 
                                   transform=transforms.Compose([transforms.ToTensor()]))
    
    # 轉換為 numpy 數組，並限制數據量，因為 SVM 訓練速度較慢
    X = train_dataset.data.numpy().reshape(-1, 28*28)[:5000] # 只取 5000 樣本
    y = train_dataset.targets.numpy()[:5000]
    
    X = X / 255.0 # 標準化數據
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def run_svm_comparison(C_param, gamma_param):
    """訓練 SVM 並計算準確度"""
    
    X_train, X_test, y_train, y_test = load_data_for_svm()
    
    st.markdown("---")
    st.subheader("📊 SVM 模型 (傳統分類) 訓練結果")
    
    try:
        svm_model = SVC(C=C_param, gamma=gamma_param, kernel='rbf', verbose=False)
        
        svm_status = st.empty()
        svm_status.info("正在訓練 SVM 模型 (這是**一次性優化**過程，請稍候...)")
        
        start_time = time.time()
        svm_model.fit(X_train, y_train) 
        end_time = time.time()

        svm_status.success(f"SVM 訓練完成！耗時: {end_time - start_time:.2f} 秒")

        y_pred = svm_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        st.metric(label="SVM 最終準確度 (在測試集上)", value=f"{accuracy * 100:.2f}%")
        st.markdown("**學習機制說明：** SVM 的目標是最大化決策邊緣，通常透過**二次規劃**來求解，而不是像 MLP 那樣進行持續的梯度迭代。因此，其結果是固定的。")
        
    except Exception as e:
        st.error(f"SVM 訓練出錯: {e}")

# --- 5. SVM 決策邊界可視化函數 (幾何優化) ---
def visualize_svm_boundary(C_param, gamma_param):
    """可視化低維度數據集上 SVM 決策邊界的形成 (非實時迭代)"""
    
    st.markdown("---")
    st.subheader("🖼️ SVM 決策邊界可視化 (低維度模擬)")
    st.markdown("為了可視化，我們使用一個**二維模擬數據集**。SVM 的優化目標：**最大化決策邊緣 (Margin)**，並識別**支持向量**。")
    
    # 創建一個非線性可分的模擬數據集 (兩個月牙形)
    X, y = make_moons(n_samples=200, noise=0.1, random_state=42)
    
    # 訓練 SVM
    svm_model = SVC(C=C_param, gamma=gamma_param, kernel='rbf')
    
    status_placeholder = st.empty()
    status_placeholder.info("正在訓練 SVM 並繪製決策邊界 (一次性幾何優化)")
    
    svm_model.fit(X, y)
    status_placeholder.success("SVM 優化完成！決策邊界已找到。")

    # --- 繪製決策邊界 ---
    
    # 設置網格範圍
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    # 調整網格密度為 0.01 (更細緻)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # 預測網格上的點，並獲取距離 (Distance) 以繪製邊緣
    Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 獲取距離：用於繪製邊緣 (Margin)
    # decision_function 返回每個點到決策邊界 Signed Distance
    W = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # 調整 figsize (長寬比)
    fig, ax = plt.subplots(figsize=(6, 5)) 
    
    # 繪製背景顏色 (決策區域)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    
    # 繪製決策邊界 (W=0) 和邊緣 (W=1 和 W=-1)
    ax.contour(xx, yy, W, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], 
               levels=[-1, 0, 1])
    
    # 繪製數據點
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    
    # 繪製支持向量 (Support Vectors)
    sv = svm_model.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=150, facecolors='none', edgecolors='green', linewidths=1.5, label='Support Vectors')
    
    ax.set_title(f'SVM 決策邊界 (C={C_param:.2f}, Gamma={gamma_param:.3f})')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()
    
    # 關鍵：使用 use_container_width=True 讓圖表適應 Streamlit 欄位寬度
    st.pyplot(fig, use_container_width=True) 
    
    st.markdown("---")
    st.markdown("#### 圖表解讀 (SVM 的「優化」):")
    st.markdown("1. **實線 (Margin=0, 黑線)：** 這是最終的**決策邊界**。它位於兩條虛線的正中央。")
    st.markdown("2. **虛線 (Margin=±1, 黑虛線)：** 兩條虛線之間的區域就是 SVM 試圖**最大化**的**決策邊緣 (Margin)**。")
    st.markdown("3. **支持向量 (綠色圓圈)：** 這些點是**唯一**決定決策邊界和邊緣位置的數據點。SVM 的優化目標是：**找到一條邊界，讓綠色圓圈到它的距離最大化。**")
    st.markdown(f"4. **模型穩定性：** SVM 選擇的邊界 (黑實線) 不會像您畫的紅線那樣緊貼數據點，這是因為 SVM 追求**穩定性 (邊緣最大化)**，這通常比追求訓練集 100% 準確度更重要。")
# --- 6. Streamlit UI 界面 ---
def main():
    st.set_page_config(layout="wide")
    st.title("🧠 機器學習過程可視化：持續改進 (MLP) vs. 一次優化 (SVM)")
    st.markdown("---")
    
    # --- 側邊欄控制項 ---
    st.sidebar.title("參數控制中心")
    
    # MLP 參數設定
    st.sidebar.header("⚙️ MLP 訓練參數 (持續改進)")
    num_epochs = st.sidebar.slider("訓練輪數 (Epochs)", 1, 10, 3, key='mlp_epochs')
    with st.sidebar.expander("❓ 訓練輪數 (Epochs) 說明"):
        st.markdown("定義模型完整地掃描一遍所有訓練數據的次數。")
    
    lr = st.sidebar.slider("學習率 (Learning Rate)", 0.001, 0.1, 0.01, format="%f", key='mlp_lr')
    with st.sidebar.expander("❓ 學習率 (Learning Rate) 說明"):
        st.markdown("定義每次參數調整（改進）的步長。")
    
    batch_size = st.sidebar.slider("批量大小 (Batch Size)", 32, 256, 128, key='mlp_batch')
    with st.sidebar.expander("❓ 批量大小 (Batch Size) 說明"):
        st.markdown("定義每次計算梯度和調整權重時所使用的數據量。")

    st.sidebar.markdown("---")
    
    # SVM 參數設定
    st.sidebar.header("⚙️ SVM 模型參數 (一次優化)")
    C_param = st.sidebar.slider("SVM C 參數 (正則化)", 0.1, 10.0, 1.0, format="%f", key='svm_c')
    with st.sidebar.expander("❓ C 參數說明"):
        st.markdown("C 參數決定了對錯誤分類樣本的懲罰程度。C 越高，模型越複雜，邊緣越窄。")
        
    gamma_param = st.sidebar.slider("SVM Gamma 參數 (核函數影響範圍)", 0.001, 0.1, 0.01, format="%f", key='svm_gamma')
    with st.sidebar.expander("❓ Gamma 參數說明"):
        st.markdown("Gamma 定義了單個訓練樣本的影響範圍。Gamma 越高，影響範圍越小，模型可能過度擬合。")


    # --- 執行按鈕 ---
    st.sidebar.markdown("---")
    
    st.subheader("💡 選擇您想執行的動作：")
    
    col_btn_mlp, col_btn_svm, col_btn_svm_viz = st.columns(3)
    
    if col_btn_mlp.button("🚀 開始 MLP 訓練 (持續改進)"):
        # 創建兩欄用於 MLP 實時訓練圖表
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📉 MLP 損失曲線 (Loss Curve)")
            placeholder_loss = st.empty()
            
        with col2:
            st.markdown("#### 📈 MLP 準確度曲線 (Accuracy Curve)")
            placeholder_acc = st.empty()
        
        train_model_and_visualize(num_epochs, lr, batch_size, placeholder_loss, placeholder_acc)

    if col_btn_svm.button("🔬 運行 SVM 準確度對比"):
        run_svm_comparison(C_param, gamma_param)

    if col_btn_svm_viz.button("🖼️ SVM 決策邊界可視化"):
        visualize_svm_boundary(C_param, gamma_param)

if __name__ == "__main__":
    main()