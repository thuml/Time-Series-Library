import argparse
import os
import torch
import torch.backends
from utils.print_args import print_args
import random
import numpy as np

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--label_col', type=str, default='label', help='label column for CSV classification datasets')
    parser.add_argument('--drop_cols', type=str, default='',
                        help='comma-separated columns to exclude from CSV classification features')
    parser.add_argument('--window_step', type=int, default=1,
                        help='stride of the sliding window for CSV classification datasets')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='train split ratio for CSV classification datasets')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='validation split ratio for CSV classification datasets')
    parser.add_argument('--file_split_mode', type=str, default='shuffle',
                        help='CSV classification file split mode, options:[shuffle, sorted]')
    parser.add_argument('--split_seed', type=int, default=2,
                        help='random seed for file-level split in CSV classification')
    parser.add_argument('--window_label_mode', type=str, default='last',
                        help='CSV classification window label mode, options:[last, majority]')
    parser.add_argument('--enable_future_state_targets', action='store_true', default=False,
                        help='enable future-state targets for lead-time classification models')
    parser.add_argument('--label_shift', type=int, default=0,
                        help='future-state lead time in steps for classification targets')
    parser.add_argument('--enable_progression_targets', action='store_true', default=False,
                        help='enable auxiliary progression labels for structured classification tasks')
    parser.add_argument('--state_graph_profile', type=str, default='none',
                        help='state graph profile, options:[none, hoister_overspeed]')
    parser.add_argument('--warning_horizon', type=int, default=5,
                        help='future horizon in steps for worsening / fault warning targets')
    parser.add_argument('--time_bucket_steps', type=str, default='1,3,5,10',
                        help='comma-separated step boundaries for time-to-fault buckets')
    parser.add_argument('--fault_raw_label', type=str, default='3',
                        help='raw label value representing the terminal fault state')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--results_root', type=str, default='./results/', help='location of test result summaries')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--tv_dt', type=int, default=0, help='whether to use time variant dt for MambaSL')
    parser.add_argument('--tv_B', type=int, default=0, help='whether to use time variant B for MambaSL')
    parser.add_argument('--tv_C', type=int, default=0, help='whether to use time variant C for MambaSL')
    parser.add_argument('--use_D', type=int, default=0, help='whether to use D for MambaSL')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=96,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--use_class_weights', action='store_true', default=False,
                        help='use inverse-frequency class weights for classification')
    parser.add_argument('--use_balanced_sampler', action='store_true', default=False,
                        help='use WeightedRandomSampler for classification training')
    parser.add_argument('--sampler_power', type=float, default=1.0,
                        help='exponent applied to per-sample weights for balanced sampler')
    parser.add_argument('--minority_raw_label', type=str, default='',
                        help='raw label value to additionally boost in balanced sampler, e.g. 9')
    parser.add_argument('--minority_boost', type=float, default=1.0,
                        help='extra multiplier for minority_raw_label in balanced sampler')
    parser.add_argument('--cls_loss', type=str, default='ce',
                        help='classification loss type, options:[ce, focal]')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='focal loss gamma when --cls_loss focal')
    parser.add_argument('--classification_early_stop_metric', type=str, default='accuracy',
                        help='validation metric used for classification checkpoint selection: accuracy, macro_f1, balanced_accuracy, fault_macro_f1')
    parser.add_argument('--aux_hazard_weight', type=float, default=0.5,
                        help='loss weight for short-horizon hazard prediction')
    parser.add_argument('--aux_time_weight', type=float, default=0.3,
                        help='loss weight for time-to-fault bucket prediction')
    parser.add_argument('--aux_next_state_weight', type=float, default=0.3,
                        help='loss weight for next-state progression supervision')
    parser.add_argument('--aux_invalid_transition_weight', type=float, default=0.05,
                        help='loss weight for penalizing invalid state transitions')
    parser.add_argument('--sgto_current_weight', type=float, default=0.3,
                        help='auxiliary current-state classification weight for SGTONet')
    parser.add_argument('--sgto_boundary_weight', type=float, default=0.2,
                        help='boundary gate supervision weight for SGTONet')
    parser.add_argument('--sgto_graph_weight', type=float, default=0.02,
                        help='graph invalid-transition penalty weight for SGTONet')
    parser.add_argument('--sgto_align_weight', type=float, default=0.1,
                        help='future representation alignment weight for SGTONet')
    parser.add_argument('--sgto_boundary_soft_weight', type=float, default=0.4,
                        help='soft future-target loss weight on boundary windows for SGTONet')
    parser.add_argument('--sgto_boundary_beta', type=float, default=0.5,
                        help='mixture ratio for current vs future class on SGTO boundary windows')
    parser.add_argument('--sgto_proto_weight', type=float, default=0.15,
                        help='prototype classification loss weight for SGTO rare-class variants')
    parser.add_argument('--sgto_proto_sep_weight', type=float, default=0.05,
                        help='prototype separation margin loss weight for SGTO rare-class variants')
    parser.add_argument('--sgto_rare_gate_weight', type=float, default=0.1,
                        help='rare-class gate supervision weight for SGTO rare-class variants')
    parser.add_argument('--sgto_rare_pull_weight', type=float, default=0.15,
                        help='rare-class prototype pull loss weight for SGTO rare-class variants')
    parser.add_argument('--sgto_rare_pos_weight', type=float, default=4.0,
                        help='positive-class weight for rare-class gate supervision')
    parser.add_argument('--sgto_proto_margin', type=float, default=0.5,
                        help='margin used by prototype separation loss')
    parser.add_argument('--sgto_proto_logit_scale', type=float, default=8.0,
                        help='logit scale for cosine prototype classifier in SGTO rare-class variants')
    parser.add_argument('--sgto_proto_mix_weight', type=float, default=0.35,
                        help='mixture weight for prototype logits in SGTO rare-class variants')
    parser.add_argument('--sgto_rare_boost_scale', type=float, default=1.25,
                        help='logit boost strength applied to the configured rare class in SGTO rare-class variants')
    parser.add_argument('--sgto_rare_margin', type=float, default=0.6,
                        help='margin used by rare-class prototype margin loss')
    parser.add_argument('--sgto_rare_margin_weight', type=float, default=0.1,
                        help='rare-class prototype margin loss weight')
    parser.add_argument('--sgto_rare_align_weight', type=float, default=0.15,
                        help='rare-class future alignment loss weight')
    parser.add_argument('--sgto_rare_rank_weight', type=float, default=0.0,
                        help='ranking loss weight for separating exact rare positives from hard rare-neighborhood negatives')
    parser.add_argument('--sgto_rare_rank_margin', type=float, default=1.0,
                        help='logit margin used by the SGTO rare ranking loss')
    parser.add_argument('--sgto_rare_hard_negative_labels', type=str, default='5,7',
                        help='comma-separated current raw labels treated as hard negatives for rare ranking')
    parser.add_argument('--sgto_patch_stride', type=int, default=0,
                        help='patch stride for SGTO patch-token encoders; defaults to patch_len // 2 when <= 0')
    parser.add_argument('--sgto_dual_rare_fuse_weight', type=float, default=0.0,
                        help='optional rare-trigger logit fusion weight for dual-mode SGTO variants')
    parser.add_argument('--sgto_dual_rare_suppress_weight', type=float, default=0.0,
                        help='optional non-rare suppression weight for dual-mode SGTO variants')
    parser.add_argument('--sgto_dual_rare_context', type=str, default='attention',
                        help='rare-trigger context for dual-mode SGTO variants: attention, mean, hidden')
    parser.add_argument('--sgto_rare_fuse_weight', type=float, default=2.0,
                        help='binary rare-head fusion strength into the final class logits')
    parser.add_argument('--sgto_nonrare_suppress_weight', type=float, default=0.35,
                        help='suppression applied to non-rare logits when rare probability is high')
    parser.add_argument('--sgto_rare_broad_gate', action='store_true', default=False,
                        help='train rare binary branch on a broader rare-neighborhood target instead of exact rare class only')
    parser.add_argument('--sgto_rare_precursor_labels', type=str, default='5,7',
                        help='comma-separated raw labels treated as rare precursors on boundary windows')
    parser.add_argument('--sgto_rare_override', action='store_true', default=False,
                        help='override final predictions with the rare binary branch during evaluation')
    parser.add_argument('--sgto_rare_override_threshold', type=float, default=0.8,
                        help='rare branch probability threshold used by evaluation-time override')
    parser.add_argument('--sgto_rare_override_require_boundary', action='store_true', default=True,
                        help='require boundary windows for rare override')
    parser.add_argument('--sgto_rare_override_no_boundary', action='store_false',
                        dest='sgto_rare_override_require_boundary',
                        help='allow rare override on non-boundary windows')
    parser.add_argument('--sgto_rare_override_precursor_labels', type=str, default='5,7',
                        help='comma-separated current raw labels allowed to trigger rare override')
    parser.add_argument('--sgto_rare_override_auto_threshold', action='store_true', default=False,
                        help='select rare override threshold on the validation split before testing')
    parser.add_argument('--sgto_rare_override_objective', type=str, default='rare_f1',
                        help='validation objective for rare override threshold: rare_f1, macro_f1, balanced_accuracy, macro_plus_recall')
    parser.add_argument('--sgto_rare_override_threshold_min', type=float, default=0.05,
                        help='minimum threshold scanned for rare override calibration')
    parser.add_argument('--sgto_rare_override_threshold_max', type=float, default=0.95,
                        help='maximum threshold scanned for rare override calibration')
    parser.add_argument('--sgto_rare_override_threshold_steps', type=int, default=19,
                        help='number of thresholds scanned for rare override calibration')
    parser.add_argument('--sgto_rare_override_fallback_threshold', type=float, default=1.01,
                        help='threshold used when validation split has no rare-class samples')
    parser.add_argument('--sgto_rare_override_min_precision', type=float, default=0.0,
                        help='minimum validation rare precision required during threshold calibration')
    parser.add_argument('--sgto_rare_override_min_recall', type=float, default=0.0,
                        help='minimum validation rare recall required during threshold calibration')
    parser.add_argument('--sgto_rare_override_recall_bonus', type=float, default=0.05,
                        help='recall bonus used by macro_plus_recall calibration objective')
    parser.add_argument('--sgto_rare_override_min_softmax', type=float, default=0.0,
                        help='minimum rare softmax probability required for rare override')
    parser.add_argument('--sgto_rare_override_margin', type=float, default=-1.0,
                        help='allow override only when rare probability is within this margin of the best non-rare probability')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', action='store_true', default=True, help='use gpu (default: on)')
    parser.add_argument('--no_use_gpu', action='store_false', dest='use_gpu', help='disable gpu (force cpu)')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', action='store_true', default=False,
                        help='enable dtw metric (time consuming; default: off)')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')

    # GCN
    parser.add_argument('--node_dim', type=int, default=10, help='each node embbed to dim dimentions')
    parser.add_argument('--gcn_depth', type=int, default=2, help='')
    parser.add_argument('--gcn_dropout', type=float, default=0.3, help='')
    parser.add_argument('--propalpha', type=float, default=0.3, help='')
    parser.add_argument('--conv_channel', type=int, default=32, help='')
    parser.add_argument('--skip_channel', type=int, default=32, help='')

    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear: a linear layer for each variate(channel) individually')

    # TimeFilter
    parser.add_argument('--alpha', type=float, default=0.1, help='KNN for Graph Construction')
    parser.add_argument('--top_p', type=float, default=0.5, help='Dynamic Routing in MoE')
    parser.add_argument('--pos', type=int, choices=[0, 1], default=1, help='Positional Embedding. Set pos to 0 or 1')

    args = parser.parse_args()
    set_random_seed(args.seed)
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)


    if args.task_name == 'long_term_forecast':
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        from exp.exp_imputation import Exp_Imputation
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        from exp.exp_anomaly_detection import Exp_Anomaly_Detection
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        from exp.exp_classification import Exp_Classification
        Exp = Exp_Classification
    elif args.task_name == 'zero_shot_forecast':
        from exp.exp_zero_shot_forecasting import Exp_Zero_Shot_Forecast
        Exp = Exp_Zero_Shot_Forecast
    else:
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            run_seed = args.seed + ii
            set_random_seed(run_seed)
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.expand,
                args.d_conv,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)
            
            # Override setting for specific model to ensure proper checkpoint naming and logging
            if args.model == 'MambaSingleLayer' and args.task_name == 'classification':
                setting = f'{args.task_name}_CLS_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                        + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_ds{args.d_ff}' \
                        + f'_expand{args.expand}_dc{args.d_conv}_nk{args.num_kernels}' \
                        + f'_tvdt{int(args.tv_dt)}_tvB{int(args.tv_B)}_tvC{int(args.tv_C)}_useD{int(args.use_D)}_{args.des}_{ii}'

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            if args.use_gpu:
                if args.gpu_type == 'mps':
                    torch.backends.mps.empty_cache()
                elif args.gpu_type == 'cuda':
                    torch.cuda.empty_cache()
    else:
        exp = Exp(args)  # set experiments
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.expand,
            args.d_conv,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)
        
        # Override setting for specific model to ensure proper checkpoint naming and logging
        if args.model == 'MambaSingleLayer' and args.task_name == 'classification':
            setting = f'{args.task_name}_CLS_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                    + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_ds{args.d_ff}' \
                    + f'_expand{args.expand}_dc{args.d_conv}_nk{args.num_kernels}' \
                    + f'_tvdt{args.tv_dt}_tvB{args.tv_B}_tvC{args.tv_C}_useD{int(args.use_D)}_{args.des}_{ii}'

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        if args.use_gpu:
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
