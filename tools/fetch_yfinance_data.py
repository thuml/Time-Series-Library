import yfinance as yf
import pandas as pd
import numpy as np
import os

print("1. 開始從 yfinance 下載台積電資料...")
# 下載台積電（台股代碼 2330.TW，美股代碼 TSM）
ticker = "2330.TW"
df = yf.download(ticker, start="2015-01-01", end="2025-01-01")

# 修正 MultiIndex columns 問題（新版 yfinance 會產生）
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# 重設 index，讓 date 變成一般欄位
df = df.reset_index()
df = df.rename(columns={"Date": "date"})

# 選擇需要的基礎欄位，並確保時間格式正確
df = df[["date", "Open", "High", "Low", "Close", "Volume"]].copy()
df["date"] = pd.to_datetime(df["date"])

print("2. 正在計算對數報酬率 (Log Return)...")
# 計算對數報酬率，乘以 100 讓數值大小更適合神經網路學習
df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1)) * 100

print("3. 正在計算 5 日滾動標準差 (Target Volatility)...")
# 🌟 關鍵修改：用 5 天滾動標準差作為波動率的代理變數
df['Target_Vol'] = df['Log_Return'].rolling(window=5).std()

# 刪除因為 shift(1) 和 rolling(5) 而在最前面幾天產生的 NaN 缺失值
df = df.dropna()

print("4. 整理欄位格式並準備存檔...")
# 選擇最終的欄位順序！(重點：Target_Vol 必須在最後一個，Log_Return 在倒數第二個)
final_columns = ['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Log_Return', 'Target_Vol']
df_final = df[final_columns]

# 建立資料夾並儲存
os.makedirs("./dataset/yfinance", exist_ok=True)
# 檔名設為 TSMC.csv，代表這是經過平滑處理的資料
output_path = "./dataset/yfinance/TSMC.csv" 
df_final.to_csv(output_path, index=False)

print("-" * 50)
print(f"✅ 成功！已儲存 {len(df_final)} 筆處理後的資料至 {output_path}")
print("-" * 50)
print("【資料預覽 (前 5 筆)】:")
print(df_final.head())
print("\n【資料型態】:")
print(df_final.dtypes)