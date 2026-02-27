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

print("2. 正在計算特徵：對數報酬率與目標波動率...")
# 計算對數報酬率，乘以 100 讓數值大小更適合神經網路學習
df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1)) * 100
# 🌟 目標變數：用 5 天滾動標準差作為波動率的代理變數
df['Target_Vol'] = df['Log_Return'].rolling(window=5).std()

print("3. 正在進行特徵去勢 (Detrending)，轉換為平穩特徵...")
# 創造平穩特徵，取代原始的絕對股價與絕對成交量
df['High_Low_Spread'] = (df['High'] - df['Low']) / df['Open'] * 100     # 當日振幅 (%)
df['Close_Open_Spread'] = (df['Close'] - df['Open']) / df['Open'] * 100 # 當日實體K棒漲跌幅 (%)
df['Vol_Change'] = np.log(df['Volume'] + 1).diff()                      # 成交量對數變化率 (平穩)

# 刪除因為 shift(1)、rolling(5) 與 diff() 而在最前面幾天產生的 NaN 缺失值
df = df.dropna()

print("4. 整理欄位格式並準備存檔...")
# 🚨 選擇最終的欄位順序！(重點：拔除 Open/High/Low/Close/Volume)
# 確保 Log_Return 在倒數第二個，Target_Vol 在最後一個！
final_columns = ['date', 'High_Low_Spread', 'Close_Open_Spread', 'Vol_Change', 'Log_Return', 'Target_Vol']
df_final = df[final_columns]

# 建立資料夾並儲存
os.makedirs("./dataset/yfinance", exist_ok=True)
# 直接覆蓋原檔名 TSMC.csv
output_path = "./dataset/yfinance/TSMC.csv" 
df_final.to_csv(output_path, index=False)

print("-" * 50)
print(f"✅ 成功！已儲存 {len(df_final)} 筆「全平穩特徵」資料至 {output_path}")
print("-" * 50)
print("【資料預覽 (前 5 筆)】:")
print(df_final.head())
print("\n【資料型態】:")
print(df_final.dtypes)




# import yfinance as yf
# import pandas as pd
# import numpy as np
# import os

# print("1. 開始從 yfinance 下載台積電資料...")
# # 下載台積電（台股代碼 2330.TW，美股代碼 TSM）
# ticker = "2330.TW"
# df = yf.download(ticker, start="2015-01-01", end="2025-01-01")

# # 修正 MultiIndex columns 問題（新版 yfinance 會產生）
# if isinstance(df.columns, pd.MultiIndex):
#     df.columns = df.columns.get_level_values(0)

# # 重設 index，讓 date 變成一般欄位
# df = df.reset_index()
# df = df.rename(columns={"Date": "date"})

# # 選擇需要的基礎欄位，並確保時間格式正確
# df = df[["date", "Open", "High", "Low", "Close", "Volume"]].copy()
# df["date"] = pd.to_datetime(df["date"])

# print("2. 正在計算對數報酬率 (Log Return)...")
# # 計算對數報酬率，乘以 100 讓數值大小更適合神經網路學習
# df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1)) * 100

# print("3. 正在計算 5 日滾動標準差 (Target Volatility)...")
# # 🌟 關鍵修改：用 5 天滾動標準差作為波動率的代理變數
# df['Target_Vol'] = df['Log_Return'].rolling(window=5).std()

# # 刪除因為 shift(1) 和 rolling(5) 而在最前面幾天產生的 NaN 缺失值
# df = df.dropna()

# print("4. 整理欄位格式並準備存檔...")
# # 選擇最終的欄位順序！(重點：Target_Vol 必須在最後一個，Log_Return 在倒數第二個)
# final_columns = ['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Log_Return', 'Target_Vol']
# df_final = df[final_columns]

# # 建立資料夾並儲存
# os.makedirs("./dataset/yfinance", exist_ok=True)
# # 檔名設為 TSMC.csv，代表這是經過平滑處理的資料
# output_path = "./dataset/yfinance/TSMC.csv" 
# df_final.to_csv(output_path, index=False)

# print("-" * 50)
# print(f"✅ 成功！已儲存 {len(df_final)} 筆處理後的資料至 {output_path}")
# print("-" * 50)
# print("【資料預覽 (前 5 筆)】:")
# print(df_final.head())
# print("\n【資料型態】:")
# print(df_final.dtypes)