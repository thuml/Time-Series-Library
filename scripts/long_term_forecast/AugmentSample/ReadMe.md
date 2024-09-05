# Augmentation Feature Roadbook

Hi there! For those who are interested in testing 
augmentation techniques in `Time-Series-Library`.

For now, we have embedded several augmentation methods
in this repo. We are still collecting publicly available 
augmentation algorithms, and we appreciate your valuable
advice!

```
The Implemented Augmentation Methods
1. jitter 
2. scaling 
3. permutation 
4. magwarp 
5. timewarp 
6. windowslice 
7. windowwarp 
8. rotation 
9. spawner 
10. dtwwarp 
11. shapedtwwarp 
12. wdba (Specially Designed for Classification tasks)
13. discdtw
```

## Usage

In this folder, we present two sample of shell scripts 
doing augmentation in `Forecasting` and `Classification`
tasks.

Take `Forecasting` task for example, we test multiple
augmentation algorithms on `EthanolConcentration` dataset
(a subset of the popular classification benchmark `UEA`) 
using `PatchTST` model.

```shell
export CUDA_VISIBLE_DEVICES=0

model_name=PatchTST

for aug in jitter scaling permutation magwarp timewarp windowslice windowwarp rotation spawner dtwwarp shapedtwwarp wdba discdtw discsdtw
do
echo using augmentation: ${aug}

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/EthanolConcentration/ \
  --model_id EthanolConcentration \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10 \
  --augmentation_ratio 1 \
  --${aug}
 done
```

Here, parameter `augmentation_ratio` represents how many
times do we want to perform our augmentation method.
Parameter `${aug}` represents a string of augmentation
type label. 

The example here only perform augmentation once, so we
can set `augmentation_ratio` to `1`, followed by one
augmentation type label. Trivially, you can set 
`augmentation_ratio` to an integer `num` followed by 
`num` augmentation type labels.

The augmentation code obeys the same prototype of 
`Time-Series-Library`. If you want to adjust other 
training parameters, feel free to add arguments to the
shell scripts and play around. The full list of parameters
can be seen in `run.py`.

## Contact Us!

This piece of code is written and maintained by 
[Yunzhong Qiu](https://github.com/DigitalLifeYZQiu). 
We thank [Haixu Wu](https://github.com/wuhaixu2016) and
[Jiaxiang Dong](https://github.com/dongjiaxiang) for 
insightful discussion and solid support.

If you have difficulties or find bugs in our code, please
contact us:
- Email: qiuyz24@mails.tsinghua.edu.cn