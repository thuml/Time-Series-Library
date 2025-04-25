from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
import mlflow
from tqdm import tqdm
import glob
import bentoml
from datetime import datetime

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

        basic_col = [
            "Open",
            "High",
            "Low",
            "Volume",
            "Vwap",
            "snp_index_vwap",
            "snp_index_open",
            "snp_index_high",
            "snp_index_low",
            "snp_index_close",
            "snp_index_volume",
            "Close"
        ]

        self.selected_columns = basic_col


    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
 

    def vali(self, vali_data, vali_loader, criterion, epoch, flag):
        total_loss = []

        feature_loss = [0.0] * len(self.selected_columns)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, stock_name) in enumerate(tqdm(vali_loader)):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss.item())

                # feature
                for idx in range(pred.shape[2]):
                    ft_loss = criterion(pred[:, :, idx], true[:, :, idx])
                    feature_loss[idx] += ft_loss.item()

        # epoch당 feature별 평균 loss
        for idx in range(len(feature_loss)):

            avg_ft_loss = feature_loss[idx] / len(vali_loader)

            self.writer.add_scalar(
                f"feature{idx} - {self.selected_columns[idx]}/{flag}_loss",
                avg_ft_loss,
                epoch + 1,
            )

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        # test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints+"/"+self.args.model+"/"+self.args.model_id)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            feature_loss = [0.0] * len(self.selected_columns)

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, stock_name) in enumerate(tqdm(train_loader)):

                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    self.writer.add_scalar(
                        "train/step_loss",
                        loss.item(),
                        epoch * len(train_loader) + i,
                    )

                    # feature만큼 반복
                    for idx in range(outputs.shape[2]):
                        ft_loss = criterion(outputs[:, :, idx], batch_y[:, :, idx])
                        feature_loss[idx] += ft_loss.item()

                        # feature별 step loss
                        self.writer.add_scalar(
                            f"feature{idx} - {self.selected_columns[idx]}/train_step_loss",
                            ft_loss.item(),
                            epoch * len(train_loader) + i,
                        )

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
            
            # epoch당 feature별 평균 loss
            for idx in range(len(feature_loss)):
                avg_ft_loss = feature_loss[idx] / len(train_loader)
                self.writer.add_scalar(
                    f"feature{idx} - {self.selected_columns[idx]}/train_loss",
                    avg_ft_loss,
                    epoch + 1,
                )

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            vali_loss = self.vali(vali_data, vali_loader, criterion, epoch, flag="val")
            mlflow.log_metric("vali_loss", vali_loss, step=epoch)
            # test_loss = self.vali(test_data, test_loader, criterion)
            # mlflow.log_metric("test_loss", test_loss, step=epoch)

            self.writer.add_scalar("train/loss", train_loss, epoch + 1)
            self.writer.add_scalar("val/loss", vali_loss, epoch + 1)

            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            
            early_stopping(vali_loss, self.model, path, epoch)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # 경로 설정
        checkpoint_dir = os.path.join(
            f"/data/pcw_workspace/Time-Series-Library/checkpoints/{self.args.model}/{self.args.model_id}/"

        )

        # checkpoint 불러오기 (checkpoint_epoch_loss.pth 형식)
        best_model_path = glob.glob(os.path.join(checkpoint_dir, "checkpoint*.pth"))

        # load : 저장된 모델 로드
        # load_state_dict : 현재 모델에 적용
        if best_model_path:
            # [0]으로 해야 가장 최근 checkpoint로 나옴
            print(f"loading checkpoint model {best_model_path[0]}")
            checkpoint = torch.load(best_model_path[0])
            self.model.load_state_dict(checkpoint)
            # bento_model = bentoml.pytorch.save_model(f"{self.model}", self.model)
            # print(f"Model saved to BentoML with name {self.model}.")

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        # 경로 설정
        checkpoint_dir = os.path.join(
            f"/data/pcw_workspace/Time-Series-Library/checkpoints/{self.args.model}/{self.args.model_id}/"
        )

        # checkpoint 불러오기 (checkpoint_epoch_loss.pth 형식)
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint*.pth"))

        if test and checkpoint_files:
            # [0]으로 해야 가장 최근 checkpoint로 나옴
            print(f"loading model {checkpoint_files[0]}")
            self.model.load_state_dict(torch.load(checkpoint_files[0]))

        preds = []
        trues = []
        png_folder_path = f"./test_results/{self.args.model}/{self.args.model_id}/"
        os.makedirs(png_folder_path, exist_ok=True)

        self.model.eval()
        mse_list = []
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader)):

                if len(batch) == 7:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, stock_name, seq_x_dates, seq_y_dates = batch
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, stock_name = batch
                    seq_x_dates, seq_y_dates = None, None # date 정보 없음
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

                if i % 20 == 0:
                    date_labels = []
                    if seq_x_dates is not None and seq_y_dates is not None:
                        full_dates = np.concatenate([seq_x_dates, seq_y_dates], axis=0)
                        date_labels = [dt[0] for dt in full_dates]
                        date_str = seq_x_dates[0][0] # 'YYYY-MM-DD'
                    else:
                        # batch_x_mark를 날짜 문자열로 변환
                        first_mark = batch_x_mark[0, 0, :3].detach().cpu().numpy()  # [year, month, day]
                        date_str = datetime(int(first_mark[0]), int(first_mark[1]), int(first_mark[2])).strftime("%Y-%m-%d")

                        # seq + pred 길이만큼만
                        for t in range(self.args.seq_len + self.args.pred_len):
                            # batch_x_mark에서 날짜 가져옴
                            if t < batch_x_mark.shape[1]:
                                mark = batch_x_mark[0, t, :3]
                            # batch_y_mark에서 날짜 가져오면서 인덱스 조정
                            else:
                                mark = batch_y_mark[0, t - batch_x_mark.shape[1], :3]
                            # 년, 월, 일 정보
                            y, m, d = mark.detach().cpu().numpy()
                            date = datetime(int(y), int(m), int(d))
                            date_labels.append(date.strftime('%y-%m-%d'))

                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    # 파일 이름 지정
                    stock_str = str(stock_name[0]) if isinstance(stock_name, (list, tuple, np.ndarray)) else str(stock_name)
                    file_name = f"{stock_str}_{date_str}_{i}.png"
                    mse=visual(gt, pd, os.path.join(png_folder_path, file_name), stock_str, date_labels)
                    # mse list 저장
                    mse_list.append((mse, file_name))
                    # MLflow에 artifact로 저장
                    #mlflow.log_artifact(os.path.join(folder_path, str(i) + '.png'))
                    

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        result_folder_path = (f"./results/{self.args.model}/{self.args.model_id}/")
        os.makedirs(result_folder_path, exist_ok=True)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        np.save(result_folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(result_folder_path + 'pred.npy', preds)
        np.save(result_folder_path + 'true.npy', trues)

        #######
        mse_folder_path = f"mse_results/{self.args.model}"
        os.makedirs(mse_folder_path, exist_ok=True)

        file_path = f"{mse_folder_path}/{self.args.model_id}_mse_result.txt"

        f = open(file_path, 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae ))
        f.write('\n')
        f.write('\n')

        # mse list (best, worst 확인)
        mse_list.sort()

        print("Best 3 MSE (Lowest MSE):")
        for mse, file_name in mse_list[:3]:
            print(f"file_name = {file_name}, MSE = {mse:.6f}")

        print("Worst 3 MSE (Highest MSE):")
        for mse, file_name in mse_list[-3:]:
            print(f"file_name = {file_name}, MSE = {mse:.6f}")

        # txt 파일에 MSE(Best, Worst) + MSE list write
        f.write("Best 3 MSE (Lowest MSE):\n")
        for mse, file_name in mse_list[:3]:
            f.write(f"file_name = {file_name}, MSE = {mse:.6f}\n")

        f.write("Worst 3 MSE (Highest MSE):\n")
        for mse, file_name in mse_list[-3:]:
            f.write(f"file_name = {file_name}, MSE = {mse:.6f}\n")

        f.write("\n")

        f.write("MSE list(sort):\n")
        for mse, file_name in mse_list:
            f.write(f"file_name = {file_name}, MSE = {mse:.6f}\n")
        f.write("\n")

        f.close()

        return
