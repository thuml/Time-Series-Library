import os
import pandas as pd
import random
from torch.utils.data import BatchSampler
from tqdm import tqdm
import math


class StockSampler(BatchSampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # 데이터 미리 로딩 (메모리 효율성)
        self.stock_data = self._load_stock_data()

    def _load_stock_data(self):
        stock_data = {}

        start_samples = 0

        # 종목 하나씩 불러옴
        for file_name, (data, data_stamp) in self.dataset.stock_data.items():

            num_days = len(data)
            if num_days >= self.dataset.seq_len + self.dataset.pred_len:
                # 유효한 sample 마지막 index
                num_samples = (
                    num_days - self.dataset.seq_len - self.dataset.pred_len + 1
                )
                # start_samples ~ num_samples까지 index 설정
                stock_data[file_name] = list(
                    range(start_samples, start_samples + num_samples)
                )
                # 다음 종목으로 넘어갈때 start_samples 변경
                start_samples += num_days

        return stock_data

    # 각 종목마다 배치가 몇개 나오는지 반환
    def _get_batches_for_stock(self, stock_name, stock_idx_list):
        if self.shuffle:
            random.shuffle(stock_idx_list)
        # stock_idx_list : 유효한 sample 수

        # batch_size 너무 크면 num_batches 0 나올수도 있음(!!)
        ## 나중에 assert 처리

        num_batches = math.ceil(
            len(stock_idx_list) / self.batch_size
        )  # drop last False일때

        # num_batches = len(stock_idx_list) // self.batch_size  # drop last True일때

        batches = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            batches.append(stock_idx_list[start_idx : start_idx + self.batch_size])

        return batches

    # sampler에서 index 반환 -> dataset에서 getitem 호출할때 사용
    def __iter__(self):
        indices = []

        # 유효한 sample수 만큼 tqdm 찍힘
        for stock_name, stock_idx_list in tqdm(self.stock_data.items()):
            # print(f"stock_name:{stock_name} / idx_list:{len(stock_idx_list)}")
            # 각 종목별로 배치 단위로 인덱스를 추출

            batches = self._get_batches_for_stock(stock_name, stock_idx_list)
            for batch in batches:
                batch_with_name = [(stock_name, idx) for idx in batch]
                indices.append(batch_with_name)

        return iter(indices)

    def __len__(self):
        # 전체 배치 수 계산(train, valid, test 각각)
        total_batches = sum(
            math.ceil(len(stock_idx_list) / self.batch_size)  # drop last False일때
            # len(stock_idx_list)//self.batch_size # drop last True일때
            for stock_idx_list in self.stock_data.values()
        )
        return total_batches