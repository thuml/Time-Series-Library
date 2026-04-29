from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score

warnings.filterwarnings('ignore')


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.long().view(-1)
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        gather_index = targets.unsqueeze(1)
        log_pt = log_probs.gather(1, gather_index).squeeze(1)
        pt = probs.gather(1, gather_index).squeeze(1)

        focal_weight = (1.0 - pt).pow(self.gamma)
        if self.weight is not None:
            alpha_t = self.weight.gather(0, targets)
            loss = -alpha_t * focal_weight * log_pt
        else:
            loss = -focal_weight * log_pt

        if self.reduction == 'sum':
            return loss.sum()
        if self.reduction == 'none':
            return loss
        return loss.mean()


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        self.class_weight_tensor = None
        super(Exp_Classification, self).__init__(args)

    @staticmethod
    def _extract_state_targets(label):
        if label.dim() == 1:
            return label.long()
        return label[:, 0].long()

    @staticmethod
    def _unpack_batch(batch):
        if len(batch) == 3:
            batch_x, label, padding_mask = batch
            future_x = None
        elif len(batch) == 4:
            batch_x, label, padding_mask, future_x = batch
        else:
            raise ValueError(f"Unexpected classification batch size: {len(batch)}")
        return batch_x, label, padding_mask, future_x

    @staticmethod
    def _extract_model_logits(outputs):
        if isinstance(outputs, dict):
            if 'future_logits' in outputs:
                return outputs['future_logits']
            return outputs['logits']
        return outputs

    @staticmethod
    def _soft_target_cross_entropy(logits, soft_targets):
        return -(soft_targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

    def _resolve_raw_label_index(self, raw_label):
        class_names = list(getattr(self.args, 'class_names', []))
        normalized = []
        for name in class_names:
            try:
                numeric = float(name)
                normalized.append(int(numeric) if numeric.is_integer() else numeric)
            except (TypeError, ValueError):
                normalized.append(name)
        try:
            numeric = float(raw_label)
            target = int(numeric) if numeric.is_integer() else numeric
        except (TypeError, ValueError):
            target = raw_label
        if target in normalized:
            return normalized.index(target)
        return None

    def _apply_rare_override_predictions(self, outputs, label, predictions):
        if not bool(getattr(self.args, 'sgto_rare_override', False)):
            return predictions
        if not isinstance(outputs, dict) or 'rare_gate_logits' not in outputs:
            return predictions

        rare_class_index = outputs.get('rare_class_index', None)
        if rare_class_index is None or int(rare_class_index) < 0:
            rare_class_index = self._resolve_raw_label_index(getattr(self.args, 'minority_raw_label', ''))
        if rare_class_index is None or int(rare_class_index) < 0:
            return predictions

        threshold = float(getattr(self, '_calibrated_rare_override_threshold', getattr(self.args, 'sgto_rare_override_threshold', 0.8)))
        rare_scores = torch.sigmoid(outputs['rare_gate_logits'].view(-1))
        override_mask = rare_scores >= threshold

        min_softmax = float(getattr(self.args, 'sgto_rare_override_min_softmax', 0.0))
        margin = float(getattr(self.args, 'sgto_rare_override_margin', -1.0))
        if (min_softmax > 0.0 or margin >= 0.0) and 'future_logits' in outputs:
            probs = F.softmax(outputs['future_logits'], dim=-1)
            rare_probs = probs[:, int(rare_class_index)]
            if min_softmax > 0.0:
                override_mask &= rare_probs >= min_softmax
            if margin >= 0.0:
                nonrare_probs = probs.clone()
                nonrare_probs[:, int(rare_class_index)] = -1.0
                best_nonrare = nonrare_probs.max(dim=1).values
                override_mask &= rare_probs >= best_nonrare - margin

        if bool(getattr(self.args, 'sgto_rare_override_require_boundary', True)) and label.size(1) >= 3:
            override_mask &= label[:, 2].float() > 0.5

        precursor_raw = str(getattr(self.args, 'sgto_rare_override_precursor_labels', '')).split(',')
        precursor_indices = [
            idx for idx in (self._resolve_raw_label_index(item.strip()) for item in precursor_raw if item.strip())
            if idx is not None
        ]
        if precursor_indices and label.size(1) >= 2:
            precursor_mask = torch.zeros_like(override_mask, dtype=torch.bool)
            current_targets = label[:, 1].long()
            for precursor_idx in precursor_indices:
                precursor_mask |= current_targets == int(precursor_idx)
            override_mask &= precursor_mask

        adjusted = predictions.clone()
        adjusted[override_mask] = int(rare_class_index)
        return adjusted

    def _score_rare_override_threshold(self, trues, base_preds, rare_scores, boundary_flags, current_targets, rare_class_index, threshold):
        predictions = base_preds.copy()
        override_mask = rare_scores >= threshold

        if bool(getattr(self.args, 'sgto_rare_override_require_boundary', True)):
            override_mask &= boundary_flags > 0.5

        precursor_raw = str(getattr(self.args, 'sgto_rare_override_precursor_labels', '')).split(',')
        precursor_indices = [
            idx for idx in (self._resolve_raw_label_index(item.strip()) for item in precursor_raw if item.strip())
            if idx is not None
        ]
        if precursor_indices:
            precursor_mask = np.zeros_like(override_mask, dtype=bool)
            for precursor_idx in precursor_indices:
                precursor_mask |= current_targets == int(precursor_idx)
            override_mask &= precursor_mask

        predictions[override_mask] = int(rare_class_index)

        rare_true = trues == int(rare_class_index)
        rare_pred = predictions == int(rare_class_index)
        true_positive = np.logical_and(rare_true, rare_pred).sum()
        pred_positive = rare_pred.sum()
        true_count = rare_true.sum()
        precision = true_positive / pred_positive if pred_positive > 0 else 0.0
        recall = true_positive / true_count if true_count > 0 else 0.0
        rare_f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        macro_f1 = f1_score(trues, predictions, average='macro', zero_division=0)
        balanced_acc = balanced_accuracy_score(trues, predictions)

        objective = str(getattr(self.args, 'sgto_rare_override_objective', 'rare_f1')).lower()
        if objective == 'macro_f1':
            score = macro_f1
        elif objective == 'balanced_accuracy':
            score = balanced_acc
        elif objective == 'macro_plus_recall':
            score = macro_f1 + float(getattr(self.args, 'sgto_rare_override_recall_bonus', 0.05)) * recall
        else:
            score = rare_f1

        min_precision = float(getattr(self.args, 'sgto_rare_override_min_precision', 0.0))
        min_recall = float(getattr(self.args, 'sgto_rare_override_min_recall', 0.0))
        if precision < min_precision or recall < min_recall:
            score = -1.0

        return {
            'threshold': float(threshold),
            'score': float(score),
            'macro_f1': float(macro_f1),
            'balanced_accuracy': float(balanced_acc),
            'class9_precision': float(precision),
            'class9_recall': float(recall),
            'class9_f1': float(rare_f1),
            'pred_rare': int(pred_positive),
        }

    def _calibrate_rare_override_threshold(self):
        if not bool(getattr(self.args, 'sgto_rare_override', False)):
            return None
        if not bool(getattr(self.args, 'sgto_rare_override_auto_threshold', False)):
            return None

        val_data, val_loader = self._get_data(flag='VAL')
        rare_class_index = self._resolve_raw_label_index(getattr(self.args, 'minority_raw_label', ''))
        if rare_class_index is None:
            return None

        trues = []
        base_preds = []
        rare_scores = []
        boundary_flags = []
        current_targets = []

        self.model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch_x, label, padding_mask, future_x = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                future_x = future_x.float().to(self.device) if future_x is not None else None
                outputs = self.model(batch_x, padding_mask, future_x, None)
                if not isinstance(outputs, dict) or 'rare_gate_logits' not in outputs:
                    return None

                logits = self._extract_model_logits(outputs).detach()
                preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
                trues.append(self._extract_state_targets(label).detach().cpu().numpy())
                base_preds.append(preds.detach().cpu().numpy())
                rare_scores.append(torch.sigmoid(outputs['rare_gate_logits'].view(-1)).detach().cpu().numpy())
                if label.size(1) >= 3:
                    boundary_flags.append(label[:, 2].float().detach().cpu().numpy())
                    current_targets.append(label[:, 1].long().detach().cpu().numpy())
                else:
                    boundary_flags.append(np.zeros(label.size(0), dtype=np.float32))
                    current_targets.append(np.full(label.size(0), -1, dtype=np.int64))

        trues = np.concatenate(trues, axis=0)
        base_preds = np.concatenate(base_preds, axis=0)
        rare_scores = np.concatenate(rare_scores, axis=0)
        boundary_flags = np.concatenate(boundary_flags, axis=0)
        current_targets = np.concatenate(current_targets, axis=0)

        if (trues == int(rare_class_index)).sum() == 0:
            threshold = float(getattr(self.args, 'sgto_rare_override_fallback_threshold', 1.01))
            self._calibrated_rare_override_threshold = threshold
            self._rare_override_calibration = {
                'threshold': threshold,
                'score': 0.0,
                'macro_f1': float(f1_score(trues, base_preds, average='macro', zero_division=0)),
                'balanced_accuracy': float(balanced_accuracy_score(trues, base_preds)),
                'class9_precision': 0.0,
                'class9_recall': 0.0,
                'class9_f1': 0.0,
                'pred_rare': 0,
            }
            print(f'rare override calibration: no rare samples in VAL, fallback threshold={threshold:.3f}')
            return threshold

        thresholds = np.linspace(
            float(getattr(self.args, 'sgto_rare_override_threshold_min', 0.05)),
            float(getattr(self.args, 'sgto_rare_override_threshold_max', 0.95)),
            int(getattr(self.args, 'sgto_rare_override_threshold_steps', 19)),
        )
        candidates = [
            self._score_rare_override_threshold(
                trues,
                base_preds,
                rare_scores,
                boundary_flags,
                current_targets,
                rare_class_index,
                threshold,
            )
            for threshold in thresholds
        ]
        # Prefer higher thresholds when validation scores tie to reduce false positives.
        best = max(candidates, key=lambda item: (item['score'], item['class9_precision'], item['threshold']))
        self._calibrated_rare_override_threshold = best['threshold']
        self._rare_override_calibration = best
        print(
            'rare override calibration: '
            f"threshold={best['threshold']:.3f}, "
            f"objective={best['score']:.4f}, "
            f"val_class9_precision={best['class9_precision']:.4f}, "
            f"val_class9_recall={best['class9_recall']:.4f}"
        )
        return best['threshold']

    def _compute_sgto_loss(self, outputs, label, criterion):
        future_targets = label[:, 0].long()
        current_targets = label[:, 1].long()
        boundary_targets = label[:, 2].float()

        future_loss = criterion(outputs['future_logits'], future_targets)
        current_loss = criterion(outputs['current_logits'], current_targets)
        boundary_loss = F.binary_cross_entropy_with_logits(outputs['boundary_logits'].view(-1), boundary_targets)
        graph_loss = outputs.get('invalid_transition_penalty', torch.zeros((), device=future_loss.device))

        boundary_soft_loss = torch.zeros((), device=future_loss.device)
        boundary_mask = boundary_targets > 0.5
        if boundary_mask.any():
            soft_beta = float(getattr(self.args, 'sgto_boundary_beta', 0.5))
            soft_targets = torch.zeros(
                boundary_mask.sum(),
                outputs['future_logits'].shape[-1],
                device=future_loss.device,
                dtype=outputs['future_logits'].dtype,
            )
            soft_targets.scatter_(1, current_targets[boundary_mask].unsqueeze(1), soft_beta)
            soft_targets.scatter_add_(
                1,
                future_targets[boundary_mask].unsqueeze(1),
                torch.full((boundary_mask.sum(), 1), 1.0 - soft_beta, device=future_loss.device, dtype=outputs['future_logits'].dtype),
            )
            boundary_soft_loss = self._soft_target_cross_entropy(outputs['future_logits'][boundary_mask], soft_targets)

        align_loss = torch.zeros((), device=future_loss.device)
        if 'target_future_hidden' in outputs:
            pred_hidden = F.normalize(outputs['future_hidden'], dim=-1)
            target_hidden = F.normalize(outputs['target_future_hidden'].detach(), dim=-1)
            align_loss = F.mse_loss(pred_hidden, target_hidden)

        prototype_loss = torch.zeros((), device=future_loss.device)
        prototype_sep_loss = torch.zeros((), device=future_loss.device)
        rare_gate_loss = torch.zeros((), device=future_loss.device)
        rare_pull_loss = torch.zeros((), device=future_loss.device)
        rare_margin_loss = torch.zeros((), device=future_loss.device)
        rare_align_loss = torch.zeros((), device=future_loss.device)
        rare_rank_loss = torch.zeros((), device=future_loss.device)

        if 'prototype_logits' in outputs:
            prototype_loss = criterion(outputs['prototype_logits'], future_targets)

            target_logits = outputs['prototype_logits'].gather(1, future_targets.unsqueeze(1)).squeeze(1)
            masked_proto_logits = outputs['prototype_logits'].clone()
            masked_proto_logits.scatter_(1, future_targets.unsqueeze(1), -1e9)
            hardest_negative = masked_proto_logits.max(dim=1).values
            proto_margin = float(getattr(self.args, 'sgto_proto_margin', 0.5))
            prototype_sep_loss = F.relu(proto_margin - (target_logits - hardest_negative)).mean()

        rare_class_index = outputs.get('rare_class_index', -1)
        if rare_class_index is None:
            rare_class_index = -1
        if rare_class_index >= 0 and 'rare_gate_logits' in outputs:
            exact_rare_mask = future_targets == int(rare_class_index)
            broad_gate = bool(getattr(self.args, 'sgto_rare_broad_gate', False))
            if broad_gate:
                rare_mask_bool = exact_rare_mask | (current_targets == int(rare_class_index))
                precursor_raw = str(getattr(self.args, 'sgto_rare_precursor_labels', '5,7')).split(',')
                precursor_indices = [
                    idx for idx in (self._resolve_raw_label_index(item.strip()) for item in precursor_raw if item.strip())
                    if idx is not None
                ]
                if precursor_indices:
                    precursor_mask = torch.zeros_like(boundary_targets, dtype=torch.bool)
                    for precursor_idx in precursor_indices:
                        precursor_mask |= (current_targets == int(precursor_idx))
                    rare_mask_bool |= precursor_mask & (boundary_targets > 0.5)
                rare_targets = rare_mask_bool.float()
            else:
                rare_targets = (future_targets == int(rare_class_index)).float()
            pos_weight = torch.tensor(
                float(getattr(self.args, 'sgto_rare_pos_weight', 4.0)),
                device=future_loss.device,
            )
            rare_gate_loss = F.binary_cross_entropy_with_logits(
                outputs['rare_gate_logits'].view(-1),
                rare_targets,
                pos_weight=pos_weight,
            )

            rare_mask = rare_targets > 0.5
            if rare_mask.any() and 'future_hidden' in outputs and 'prototypes' in outputs:
                norm_hidden = F.normalize(outputs['future_hidden'][rare_mask], dim=-1)
                norm_proto = F.normalize(outputs['prototypes'][int(rare_class_index)].unsqueeze(0), dim=-1)
                rare_pull_loss = 1.0 - (norm_hidden * norm_proto).sum(dim=-1).mean()

            if 'prototype_logits' in outputs:
                rare_proto_logits = outputs['prototype_logits'][:, int(rare_class_index)]
                masked_proto_logits = outputs['prototype_logits'].clone()
                masked_proto_logits[:, int(rare_class_index)] = -1e9
                best_other = masked_proto_logits.max(dim=1).values
                rare_margin = float(getattr(self.args, 'sgto_rare_margin', 0.6))
                if rare_mask.any():
                    rare_margin_loss = rare_margin_loss + F.relu(
                        rare_margin - (rare_proto_logits[rare_mask] - best_other[rare_mask])
                    ).mean()
                nonrare_mask = ~rare_mask
                if nonrare_mask.any():
                    rare_margin_loss = rare_margin_loss + F.relu(
                        rare_margin + rare_proto_logits[nonrare_mask] - best_other[nonrare_mask]
                    ).mean()

            if rare_mask.any() and 'target_future_hidden' in outputs:
                pred_hidden = F.normalize(outputs['future_hidden'][rare_mask], dim=-1)
                target_hidden = F.normalize(outputs['target_future_hidden'][rare_mask].detach(), dim=-1)
                rare_align_loss = F.mse_loss(pred_hidden, target_hidden)

            hard_negative_raw = str(getattr(self.args, 'sgto_rare_hard_negative_labels', '5,7')).split(',')
            hard_negative_indices = [
                idx for idx in (self._resolve_raw_label_index(item.strip()) for item in hard_negative_raw if item.strip())
                if idx is not None
            ]
            hard_negative_mask = torch.zeros_like(exact_rare_mask, dtype=torch.bool)
            for hard_negative_idx in hard_negative_indices:
                hard_negative_mask |= current_targets == int(hard_negative_idx)
            hard_negative_mask &= (boundary_targets > 0.5) & (~exact_rare_mask)
            if not hard_negative_indices:
                hard_negative_mask = (boundary_targets > 0.5) & (~exact_rare_mask)
            if exact_rare_mask.any() and hard_negative_mask.any():
                rare_gate_logits = outputs['rare_gate_logits'].view(-1)
                positive_logits = rare_gate_logits[exact_rare_mask]
                negative_logits = rare_gate_logits[hard_negative_mask]
                rare_rank_margin = float(getattr(self.args, 'sgto_rare_rank_margin', 1.0))
                rare_rank_loss = F.relu(
                    rare_rank_margin - positive_logits.unsqueeze(1) + negative_logits.unsqueeze(0)
                ).mean()

        return (
            future_loss
            + float(getattr(self.args, 'sgto_current_weight', 0.3)) * current_loss
            + float(getattr(self.args, 'sgto_boundary_weight', 0.2)) * boundary_loss
            + float(getattr(self.args, 'sgto_graph_weight', 0.02)) * graph_loss
            + float(getattr(self.args, 'sgto_align_weight', 0.1)) * align_loss
            + float(getattr(self.args, 'sgto_boundary_soft_weight', 0.4)) * boundary_soft_loss
            + float(getattr(self.args, 'sgto_proto_weight', 0.15)) * prototype_loss
            + float(getattr(self.args, 'sgto_proto_sep_weight', 0.05)) * prototype_sep_loss
            + float(getattr(self.args, 'sgto_rare_gate_weight', 0.1)) * rare_gate_loss
            + float(getattr(self.args, 'sgto_rare_pull_weight', 0.15)) * rare_pull_loss
            + float(getattr(self.args, 'sgto_rare_margin_weight', 0.1)) * rare_margin_loss
            + float(getattr(self.args, 'sgto_rare_align_weight', 0.15)) * rare_align_loss
            + float(getattr(self.args, 'sgto_rare_rank_weight', 0.0)) * rare_rank_loss
        )

    def _compute_classification_loss(self, outputs, label, criterion):
        if not isinstance(outputs, dict):
            return criterion(outputs, self._extract_state_targets(label))

        if 'future_logits' in outputs and label.size(1) >= 3:
            return self._compute_sgto_loss(outputs, label, criterion)

        if {'hazard_logits', 'time_logits', 'next_state_log_probs'}.issubset(outputs.keys()) and label.size(1) >= 4:
            state_targets = self._extract_state_targets(label)
            total_loss = criterion(outputs['logits'], state_targets)

            hazard_targets = label[:, 1].float()
            time_targets = label[:, 2].long()
            next_state_targets = label[:, 3].long()

            hazard_loss = F.binary_cross_entropy_with_logits(
                outputs['hazard_logits'].view(-1), hazard_targets
            )
            time_loss = F.cross_entropy(outputs['time_logits'], time_targets)
            next_state_loss = F.nll_loss(outputs['next_state_log_probs'], next_state_targets)

            invalid_penalty = outputs['invalid_transition_penalty']

            total_loss = total_loss \
                + float(getattr(self.args, 'aux_hazard_weight', 0.5)) * hazard_loss \
                + float(getattr(self.args, 'aux_time_weight', 0.3)) * time_loss \
                + float(getattr(self.args, 'aux_next_state_weight', 0.3)) * next_state_loss \
                + float(getattr(self.args, 'aux_invalid_transition_weight', 0.05)) * invalid_penalty
            return total_loss

        state_targets = self._extract_state_targets(label)
        return criterion(outputs['logits'], state_targets)

    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.dec_in = self.args.enc_in
        self.args.c_out = self.args.enc_in
        self.args.num_class = len(train_data.class_names)
        self.args.class_names = list(train_data.class_names)
        self.args.num_time_buckets = int(getattr(train_data, 'time_bucket_count', 0))
        # model init
        model = self.model_dict[self.args.model](self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        class_weights = None
        if getattr(self.args, 'use_class_weights', False):
            train_data, _ = self._get_data(flag='TRAIN')
            if getattr(train_data, 'class_weights', None) is not None:
                class_weights = torch.tensor(train_data.class_weights, dtype=torch.float32, device=self.device)
        self.class_weight_tensor = class_weights

        loss_name = str(getattr(self.args, 'cls_loss', 'ce')).lower()
        if loss_name == 'focal':
            gamma = float(getattr(self.args, 'focal_gamma', 2.0))
            criterion = FocalLoss(weight=class_weights, gamma=gamma)
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                batch_x, label, padding_mask, future_x = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                future_x = future_x.float().to(self.device) if future_x is not None else None

                outputs = self.model(batch_x, padding_mask, future_x, None)

                pred = self._extract_model_logits(outputs).detach()
                loss = self._compute_classification_loss(outputs, label, criterion)
                total_loss.append(loss.item())

                preds.append(pred)
                trues.append(self._extract_state_targets(label))

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = F.softmax(preds, dim=1)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)
        macro_f1 = f1_score(trues, predictions, average='macro', zero_division=0)
        balanced_acc = balanced_accuracy_score(trues, predictions)
        class_names = [str(name) for name in getattr(vali_data, 'class_names', list(range(self.args.num_class)))]
        raw_label_to_idx = {str(raw_label): idx for idx, raw_label in enumerate(class_names)}
        fault_indices = [raw_label_to_idx[key] for key in ['3', '7', '9'] if key in raw_label_to_idx]
        fault_macro_f1 = (
            f1_score(trues, predictions, labels=fault_indices, average='macro', zero_division=0)
            if fault_indices else float('nan')
        )
        metrics = {
            'accuracy': float(accuracy),
            'macro_f1': float(macro_f1),
            'balanced_accuracy': float(balanced_acc),
            'fault_macro_f1': float(fault_macro_f1),
        }

        self.model.train()
        return total_loss, accuracy, metrics

    def _select_classification_checkpoint_score(self, metrics):
        metric_name = str(getattr(self.args, 'classification_early_stop_metric', 'accuracy')).lower()
        aliases = {
            'acc': 'accuracy',
            'balanced_acc': 'balanced_accuracy',
            'bal_acc': 'balanced_accuracy',
            'fault_f1': 'fault_macro_f1',
        }
        metric_name = aliases.get(metric_name, metric_name)
        if metric_name not in metrics:
            raise ValueError(
                f"Unsupported classification_early_stop_metric={metric_name}. "
                "Use accuracy, macro_f1, balanced_accuracy, or fault_macro_f1."
            )
        return float(metrics[metric_name]), metric_name

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='VAL')
        test_data, test_loader = self._get_data(flag='TEST')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, batch in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x, label, padding_mask, future_x = self._unpack_batch(batch)

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                future_x = future_x.float().to(self.device) if future_x is not None else None

                outputs = self.model(batch_x, padding_mask, future_x, None)
                loss = self._compute_classification_loss(outputs, label, criterion)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy, val_metrics = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy, test_metrics = self.vali(test_data, test_loader, criterion)
            val_score, val_score_name = self._select_classification_checkpoint_score(val_metrics)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} "
                "Vali {5}: {6:.3f} Test Loss: {7:.3f} Test Acc: {8:.3f} Test Macro-F1: {9:.3f}"
                .format(
                    epoch + 1, train_steps, train_loss, vali_loss, val_accuracy,
                    val_score_name, val_score, test_loss, test_accuracy, test_metrics['macro_f1']
                ))
            early_stopping(-val_score, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth'), map_location=self.device)
            )

        preds = []
        base_pred_indices = []
        pred_indices = []
        rare_score_rows = []
        current_indices = []
        boundary_flags = []
        trues = []
        results_root = getattr(self.args, 'results_root', './results/')
        folder_path = os.path.abspath(os.path.join(results_root, setting))
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        calibrated_threshold = self._calibrate_rare_override_threshold()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch_x, label, padding_mask, future_x = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                future_x = future_x.float().to(self.device) if future_x is not None else None

                outputs = self.model(batch_x, padding_mask, future_x, None)

                batch_logits = self._extract_model_logits(outputs).detach()
                batch_base_predictions = torch.argmax(F.softmax(batch_logits, dim=1), dim=1)
                batch_predictions = self._apply_rare_override_predictions(outputs, label, batch_base_predictions)

                if isinstance(outputs, dict) and 'rare_gate_logits' in outputs:
                    rare_score_rows.append(torch.sigmoid(outputs['rare_gate_logits'].view(-1)).detach().cpu())
                else:
                    rare_score_rows.append(torch.full((batch_logits.size(0),), float('nan')))

                preds.append(batch_logits)
                base_pred_indices.append(batch_base_predictions.detach().cpu())
                pred_indices.append(batch_predictions.detach().cpu())
                trues.append(self._extract_state_targets(label))
                if label.dim() > 1 and label.size(1) >= 2:
                    current_indices.append(label[:, 1].long().detach().cpu())
                else:
                    current_indices.append(torch.full((label.size(0),), -1, dtype=torch.long))
                if label.dim() > 1 and label.size(1) >= 3:
                    boundary_flags.append(label[:, 2].float().detach().cpu())
                else:
                    boundary_flags.append(torch.zeros(label.size(0), dtype=torch.float32))

        preds = torch.cat(preds, 0)
        base_pred_indices = torch.cat(base_pred_indices, 0)
        pred_indices = torch.cat(pred_indices, 0)
        rare_score_rows = torch.cat(rare_score_rows, 0)
        current_indices = torch.cat(current_indices, 0)
        boundary_flags = torch.cat(boundary_flags, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)

        predictions = pred_indices.numpy()
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)
        macro_f1 = f1_score(trues, predictions, average='macro', zero_division=0)
        weighted_f1 = f1_score(trues, predictions, average='weighted', zero_division=0)
        balanced_acc = balanced_accuracy_score(trues, predictions)

        class_names = [str(name) for name in getattr(test_data, 'class_names', list(range(self.args.num_class)))]
        class_indices = list(range(len(class_names)))
        report = classification_report(
            trues,
            predictions,
            labels=class_indices,
            target_names=class_names,
            digits=4,
            zero_division=0
        )
        report_dict = classification_report(
            trues,
            predictions,
            labels=class_indices,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        conf_mat = confusion_matrix(trues, predictions, labels=class_indices)

        raw_label_to_idx = {str(raw_label): idx for idx, raw_label in enumerate(class_names)}
        fault_indices = [raw_label_to_idx[key] for key in ['3', '7', '9'] if key in raw_label_to_idx]
        if fault_indices:
            fault_macro_f1 = f1_score(
                trues, predictions, labels=fault_indices, average='macro', zero_division=0
            )
        else:
            fault_macro_f1 = float('nan')

        class9_metrics = report_dict.get('9', {})
        class9_precision = float(class9_metrics.get('precision', 0.0))
        class9_recall = float(class9_metrics.get('recall', 0.0))
        class9_f1 = float(class9_metrics.get('f1-score', 0.0))
        class9_support = float(class9_metrics.get('support', 0.0))
        calibration = getattr(self, '_rare_override_calibration', {})
        calibrated_threshold = (
            float(calibrated_threshold)
            if calibrated_threshold is not None
            else float(getattr(self.args, 'sgto_rare_override_threshold', 0.8))
        )

        # result save
        print('accuracy:{}'.format(accuracy))
        print('macro_f1:{}'.format(macro_f1))
        print('weighted_f1:{}'.format(weighted_f1))
        print('balanced_accuracy:{}'.format(balanced_acc))
        print('fault_macro_f1:{}'.format(fault_macro_f1))
        print('class9_precision:{}'.format(class9_precision))
        print('class9_recall:{}'.format(class9_recall))
        print('class9_f1:{}'.format(class9_f1))
        if bool(getattr(self.args, 'sgto_rare_override', False)):
            print('rare_override_threshold:{}'.format(calibrated_threshold))
        print(report)
        result_file = os.path.join(folder_path, 'metrics.txt')
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        with open(result_file, 'a', encoding='utf-8') as f:
            f.write(setting + "  \n")
            f.write('accuracy:{}'.format(accuracy))
            f.write('\n')
            f.write('macro_f1:{}'.format(macro_f1))
            f.write('\n')
            f.write('weighted_f1:{}'.format(weighted_f1))
            f.write('\n')
            f.write('balanced_accuracy:{}'.format(balanced_acc))
            f.write('\n')
            f.write('fault_macro_f1:{}'.format(fault_macro_f1))
            f.write('\n')
            f.write('class9_precision:{}'.format(class9_precision))
            f.write('\n')
            f.write('class9_recall:{}'.format(class9_recall))
            f.write('\n')
            f.write('class9_f1:{}'.format(class9_f1))
            f.write('\n')
            if bool(getattr(self.args, 'sgto_rare_override', False)):
                f.write('rare_override_threshold:{}'.format(calibrated_threshold))
                f.write('\n')
            f.write(report)
            f.write('\n')
            f.write('\n')

        summary_df = pd.DataFrame([
            {
                'setting': setting,
                'accuracy': accuracy,
                'macro_f1': macro_f1,
                'weighted_f1': weighted_f1,
                'balanced_accuracy': balanced_acc,
                'fault_macro_f1': fault_macro_f1,
                'class9_precision': class9_precision,
                'class9_recall': class9_recall,
                'class9_f1': class9_f1,
                'class9_support': class9_support,
                'rare_override_threshold': calibrated_threshold if bool(getattr(self.args, 'sgto_rare_override', False)) else '',
                'rare_override_val_precision': calibration.get('class9_precision', ''),
                'rare_override_val_recall': calibration.get('class9_recall', ''),
                'rare_override_val_f1': calibration.get('class9_f1', ''),
            }
        ])
        summary_df.to_csv(os.path.join(folder_path, 'summary.csv'), index=False)

        pred_label_names = class_names
        prediction_rows = pd.DataFrame({
            'true_index': trues,
            'base_pred_index': base_pred_indices.numpy(),
            'pred_index': predictions,
            'current_index': current_indices.numpy(),
            'boundary_flag': boundary_flags.numpy(),
            'true_label': [pred_label_names[idx] for idx in trues],
            'base_pred_label': [pred_label_names[idx] for idx in base_pred_indices.numpy()],
            'pred_label': [pred_label_names[idx] for idx in predictions],
            'current_label': [
                pred_label_names[idx] if 0 <= idx < len(pred_label_names) else ''
                for idx in current_indices.numpy()
            ],
            'rare_score': rare_score_rows.numpy(),
            'rare_override_applied': base_pred_indices.numpy() != predictions,
        })
        sample_metadata = getattr(test_data, 'sample_metadata', None)
        if sample_metadata is not None and len(sample_metadata) == len(prediction_rows):
            prediction_rows = pd.concat([pd.DataFrame(sample_metadata), prediction_rows], axis=1)

        prediction_rows.to_csv(os.path.join(folder_path, 'pred.csv'), index=False)
        pd.DataFrame(conf_mat, index=class_names, columns=class_names).to_csv(
            os.path.join(folder_path, 'cm.csv')
        )
        return
