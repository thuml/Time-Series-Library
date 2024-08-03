from data_provider.data_loader import Dataset_IE_day

root_path = "/Users/guoyangyun/计量中心/16.分析报告/分析/智能报表/全社会用电量预测（省级）/cache"
size = [7 * 4, 7, 7]
features = "MS"
data_path = "preprocessed_hydl_tq_r_df.pkl"
target = "electricity_consumption"
scale = False
timeenc = 0
freq = "d"
province_name = "广东"
industry_name = "全社会用电总计"
train_start = "2022-01-01"
train_end = "2022-09-30"
test_start = "2022-12-01"
test_end = "2022-12-31"
cols = "date,wd,wd_max,wd_min,holiday_code,electricity_consumption"

train_data_loader = Dataset_IE_day(
    root_path=root_path,
    flag="train",
    size=size,
    features=features,
    data_path=data_path,
    target=target,
    scale=scale,
    timeenc=timeenc,
    freq=freq,
    province_name=province_name,
    industry_name=industry_name,
    train_start=train_start,
    train_end=train_end,
    test_start=test_start,
    test_end=test_end,
    cols=cols,
)
train_demo = train_data_loader.__getitem__(1)

val_data_loader = Dataset_IE_day(
    root_path=root_path,
    flag="val",
    size=size,
    features=features,
    data_path=data_path,
    target=target,
    scale=scale,
    timeenc=timeenc,
    freq=freq,
    province_name=province_name,
    industry_name=industry_name,
    train_start=train_start,
    train_end=train_end,
    test_start=test_start,
    test_end=test_end,
    cols=cols,
)
val_demo = val_data_loader.__getitem__(1)

test_data_loader = Dataset_IE_day(
    root_path=root_path,
    flag="test",
    size=size,
    features=features,
    data_path=data_path,
    target=target,
    scale=scale,
    timeenc=timeenc,
    freq=freq,
    province_name=province_name,
    industry_name=industry_name,
    train_start=train_start,
    train_end=train_end,
    test_start=test_start,
    test_end=test_end,
    cols=cols,
)
test_demo = test_data_loader.__getitem__(1)

pred_data_loader = Dataset_IE_day(
    root_path=root_path,
    flag="pred",
    size=size,
    features=features,
    data_path=data_path,
    target=target,
    scale=scale,
    timeenc=timeenc,
    freq=freq,
    province_name=province_name,
    industry_name=industry_name,
    train_start=train_start,
    train_end=train_end,
    test_start=test_start,
    test_end=test_end,
    cols=cols,
)
pred_demo = test_data_loader.__getitem__(1)
from IPython import embed

embed()
