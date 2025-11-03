# 🚀 機器學習過程可視化：持續改進 (MLP) vs. 一次優化 (SVM)

本專案提供一個互動式的 Streamlit 應用程式，旨在透過實時可視化，清晰地展示現代深度學習模型（MLP）的**「持續改進」**機制，並與傳統機器學習模型（SVM）的**「一次性優化」**機制進行對比。

## ✨ 專案亮點

*   **📈 MLP 實時訓練可視化：** 實時繪製 Loss（損失）和 Accuracy（準確度）曲線，動態展示 AI 如何通過**反向傳播**和**梯度下降**持續學習和改進。
*   **⚙️ 互動式超參數調整：** 允許用戶實時調整學習率、訓練輪數等超參數，並觀察它們如何影響模型的改進過程和收斂結果。
*   **🔬 SVM 訓練對比：** 展示 SVM 的一次性訓練時間和最終準確度，與 MLP 的迭代學習形成對比。
*   **🖼️ SVM 決策邊界可視化：** 在低維度數據上繪製 SVM 的**決策邊界**、**決策邊緣 (Margin)** 和**支持向量**，直觀解釋 SVM 的幾何優化原理。

## 🛠️ 如何運行專案

### 1. 克隆專案

```bash
git clone https://github.com/SmailDot/NKUST_-HW1_MLP-SVM-Train.git
cd NKUST_-HW1_MLP-SVM-Train
```

### 2.安裝依賴
使用 requirements.txt 來管理所有必要的 Python 庫。
```bash
pip install -r requirements.txt
```
### 3. 啟動 Streamlit 應用程式

```bash
streamlit run app.py
```
程式將自動開啟 (http://localhost:8501)

### 4.📂 文件結構說明

| 文件名 | 說明 |
| --- | --- |
| app.py | 核心應用程式：整合了 MLP 訓練、SVM 訓練和所有 Streamlit UI 邏輯。 |
| requirements.txt | Python 依賴包列表 (streamlit, torch, scikit-learn 等)。 |
| README.md | 專案概覽與運行指南。 |
| MLP推導過程.md | 詳細解釋 MLP 的梯度下降與反向傳播原理。 |
| SVM推導過程.md | 詳細解釋 SVM 的邊緣最大化與二次規劃原理。 |

### 5.💡 學習目標
透過這個應用程式和文件，您可以清晰地掌握兩種模型的本質差異：

**深度學習 (MLP)**： 基於微積分和迭代的持續優化。

**傳統機器學習 (SVM)**： 基於幾何學和一次性數學求解的優化。

