import subprocess
import time

def run_experiments():
    # 💡 將你所有想執行的指令，一行一行字串放在這個 list 裡面
    commands = [
        "python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/yfinance/ --data_path TSMC.csv --model_id TSMC_48_7 --model cch_LSTMAttention --use_norm 1 --data custom --features MS --target Target_Vol --freq b --seq_len 96 --label_len 48 --pred_len 7 --e_layers 2 --d_layers 1 --factor 3 --enc_in 5 --dec_in 5 --c_out 1 --d_model 32 --d_ff 64 --n_heads 4 --dropout 0.2 --embed timeF --loss MAE --patience 10 --des Exp --itr 1 --train_epochs 30 --batch_size 32 --learning_rate 0.0005 --lradj cosine"
    ]

    total_experiments = len(commands)
    print(f"📦 批次任務啟動：共計 {total_experiments} 個實驗準備執行...\n")

    for i, cmd in enumerate(commands):
        print("=" * 80)
        print(f"🚀 開始執行實驗 [{i+1}/{total_experiments}]")
        print(f"指令內容:\n{cmd}")
        print("=" * 80)
        
        start_time = time.time()
        
        try:
            # 呼叫系統終端機執行指令，這會將執行過程的 log 實時印在你的畫面上
            subprocess.run(cmd, shell=True, check=True)
            
            elapsed_time = time.time() - start_time
            print(f"\n✅ 實驗 [{i+1}/{total_experiments}] 成功執行完畢！耗時: {elapsed_time/60:.2f} 分鐘\n")
            
            # 每個實驗跑完後休息 3 秒，讓 GPU 釋放記憶體
            time.sleep(3) 

        except subprocess.CalledProcessError as e:
            # 如果某個實驗發生 Bug 崩潰，會捕捉錯誤並印出，然後直接停止後續實驗
            print(f"\n❌ 實驗 [{i+1}/{total_experiments}] 發生錯誤而中斷！")
            print(f"系統回傳錯誤碼: {e.returncode}")
            print("批次任務已停止。")
            break
        except KeyboardInterrupt:
            # 讓你可以用 Ctrl+C 隨時中斷整個批次腳本
            print("\n🛑 接收到使用者中斷指令，批次任務結束。")
            break

if __name__ == "__main__":
    run_experiments()