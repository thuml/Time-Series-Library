#!/bin/bash

SCRIPT_WEATHER_ITRANSFORMER="./scripts/imputation//Weather_script/iTransformer_seq_len_exp.sh"
SCRIPT_EETH_ITIMESFORMERCYCLEATTN="./scripts/imputation//ETT_script/iTimesformercycleattn_ETTh2_seq_len_exp.sh"
SCRIPT_EETH_ITIMESFORMER="./scripts/imputation//ETT_script/iTimesformer_ETTh2_seq_len_exp.sh"
SCRIPT_EETH_ITRANSFORMER="./scripts/imputation//ETT_script/iTransformer_ETTh2_seq_len_exp.sh"
SCRIPT_ETTH_TIMESNET="./scripts/imputation//ETT_script/TimesNet_ETTh2_seq_len_exp.sh"

bash "$SCRIPT_WEATHER_ITRANSFORMER"
bash "$SCRIPT_EETH_ITIMESFORMERCYCLEATTN"
bash "$SCRIPT_EETH_ITIMESFORMER"
bash "$SCRIPT_EETH_ITRANSFORMER"
bash "$SCRIPT_ETTH_TIMESNET"