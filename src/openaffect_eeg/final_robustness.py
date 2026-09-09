"""Bounded v9 support sensitivity, separate from historical experiment modules."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
from importlib.metadata import version
import time

import numpy as np
import pandas as pd

from openaffect_eeg.baselines import regression_metrics
from openaffect_eeg.identity_exposure import (
    _fallback_prior, _participant_offsets, leave_one_out_prior, predict_prior,
)


def coverage_assignments(trials, *, folds=5, seed=20260908):
    """Cover every eligible identity once; not every crossed trial pair."""
    columns = ["trial_uid", "subject_uid", "stimulus_uid"]
    if trials[columns].isna().any().any() or trials.trial_uid.duplicated().any():
        raise ValueError("Missing or duplicate identities")
    if folds < 3:
        raise ValueError("At least three identity blocks are required")
    blocks = {}
    for axis in columns[1:]:
        values = trials[axis].astype(str).unique()
        if len(values) < folds * 2:
            raise ValueError("Insufficient identities for supported coverage blocks")
        ordered = sorted(values, key=lambda uid: hashlib.sha256(f"{seed}:{axis}:{uid}".encode()).hexdigest())
        blocks[axis] = {uid: i % folds for i, uid in enumerate(ordered)}
    p = trials.subject_uid.astype(str).map(blocks["subject_uid"]).to_numpy()
    s = trials.stimulus_uid.astype(str).map(blocks["stimulus_uid"]).to_numpy()
    result = []
    for fold in range(folds):
        val = (fold + 1) % folds
        roles = np.full(len(trials), "excluded", dtype=object)
        roles[(p != fold) & (p != val) & (s != fold) & (s != val)] = "train"
        roles[(p == val) & (s == val)] = "validation"
        roles[(p == fold) & (s == fold)] = "test"
        assignment = pd.DataFrame({"trial_uid": trials.trial_uid.to_numpy(), "split": roles})
        if min((roles == part).sum() for part in ("train", "validation", "test")) < 4:
            raise ValueError(f"Insufficient support in coverage fold {fold}")
        result.append(assignment)
    return result


def population_targets(trials, assignment, targets):
    assigned = trials.merge(assignment[["trial_uid", "split"]], on="trial_uid", validate="1:1")
    if len(assigned) != len(trials):
        raise ValueError("Assignment must cover all eligible trials")
    train = assigned.loc[assigned.split.eq("train")]
    val = assigned.loc[assigned.split.eq("validation")]
    fit = pd.concat([train, val], ignore_index=True)
    names = tuple(t.removeprefix("target_") for t in targets)

    def prepare(source, requested):
        y = source[list(targets)].to_numpy(float)
        loo = leave_one_out_prior(y, source.stimulus_uid.to_numpy(),
                                 _fallback_prior(source, y.mean(0), target_names=names))
        prior, _ = predict_prior(y, source.stimulus_uid.to_numpy(), requested.stimulus_uid.to_numpy(),
                                 _fallback_prior(requested, y.mean(0), target_names=names))
        return y, loo, prior

    train_y, train_loo, val_prior = prepare(train, val)
    fit_y, fit_loo, all_prior = prepare(fit, trials)
    return dict(train=train, validation=val, fit=fit, train_y=train_y, train_loo=train_loo,
                val_y=val[list(targets)].to_numpy(float), val_prior=val_prior,
                fit_y=fit_y, fit_loo=fit_loo, all_prior=all_prior)


def prediction_table(trials, assignment, targets, prior, direct, residual):
    """Apply exactly the same complete calibration trials to every predictor."""
    joined = trials.merge(assignment[["trial_uid", "split"]], on="trial_uid", validate="1:1")
    positions = pd.Index(trials.trial_uid)
    test = joined.loc[joined.split.eq("test")].copy()
    cal = joined.loc[joined.split.eq("calibration")].copy()
    ti, ci = positions.get_indexer(test.trial_uid), positions.get_indexer(cal.trial_uid)
    output = test[["trial_uid", "subject_uid", "stimulus_uid", *targets]].reset_index(drop=True)
    for name, values in (("prior", prior), ("eeg", direct), ("combined", prior + residual)):
        population = name if name == "prior" else name + "_population"
        calibrated = values[ti] + _participant_offsets(cal, values[ci], test,
                                                      subject_column="subject_uid", target_columns=targets)
        for index, target in enumerate(targets):
            suffix = target.removeprefix("target_")
            output[f"{population}_{suffix}"] = values[ti, index]
            output[f"{name}_personalized_{suffix}"] = calibrated[:, index]
    return output


@dataclass(frozen=True)
class GPURegressionConfig:
    crop_samples: int
    learning_rate: float
    max_epochs: int = 100
    patience: int = 15
    batch_size: int = 128
    weight_decay: float = 0.0001
    dropout: float = 0.25

    def __post_init__(self):
        if min(self.crop_samples, self.max_epochs, self.patience, self.batch_size) < 1:
            raise ValueError("Invalid training dimensions")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Invalid learning rate")


def fit_gpu_regression(tensors, train_indices, train_y, validation_indices, validation_y,
                       fit_indices, fit_y, *, seed, config, checkpoint):
    """Official EEGNet, arbitrary genuine targets, GPU-resident trial batches."""
    import torch
    from braindecode.models import EEGNet

    if version("braindecode") != "1.2.0":
        raise ValueError("The frozen adapter requires braindecode 1.2.0")
    if not tensors.is_cuda or tensors.ndim != 3 or config.crop_samples > tensors.shape[2]:
        raise ValueError("Expected CUDA trials at least as long as the crop")
    train_indices, validation_indices, fit_indices = map(np.asarray, (train_indices, validation_indices, fit_indices))
    if np.intersect1d(train_indices, validation_indices).size:
        raise ValueError("Train/validation trial overlap")
    if set(fit_indices) != set(train_indices) | set(validation_indices):
        raise ValueError("Refit must contain train and validation only")
    targets = train_y.shape[1]
    for ids, y in ((train_indices, train_y), (validation_indices, validation_y), (fit_indices, fit_y)):
        if y.shape != (len(ids), targets) or not np.isfinite(y).all() or len(set(ids)) != len(ids):
            raise ValueError("Malformed target or index support")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    device = tensors.device
    starts = list(range(0, tensors.shape[2] - config.crop_samples + 1, config.crop_samples))
    if starts[-1] != tensors.shape[2] - config.crop_samples:
        starts.append(tensors.shape[2] - config.crop_samples)

    def normalization(ids, y):
        block = tensors[torch.as_tensor(ids, device=device)]
        mean = block.mean(dim=(0, 2), keepdim=True)
        scale = block.std(dim=(0, 2), keepdim=True, correction=0).clamp_min(1e-8)
        yy = torch.as_tensor(y, dtype=torch.float32, device=device)
        ym, ys = yy.mean(0), yy.std(0, correction=0).clamp_min(1e-8)
        return mean, scale, ym, ys

    def model():
        torch.manual_seed(int(seed))
        return EEGNet(n_chans=tensors.shape[1], n_outputs=targets, n_times=config.crop_samples,
                      F1=8, D=2, F2=16, kernel_length=64, depthwise_kernel_length=16,
                      drop_prob=config.dropout, final_layer_with_constraint=False).to(device)

    def predict(net, ids, norm):
        mean, scale, ym, ys = norm
        net.eval()
        chunks = []
        with torch.no_grad():
            for first in range(0, len(ids), config.batch_size):
                indices = torch.as_tensor(ids[first:first + config.batch_size], device=device)
                current = tensors[indices]
                predictions = [net((current[:, :, start:start + config.crop_samples] - mean) / scale)
                               for start in starts]
                chunks.append(torch.stack(predictions).mean(0) * ys + ym)
        return torch.cat(chunks).cpu().numpy()

    def train(ids, y, epochs, use_validation):
        norm = normalization(ids, y)
        mean, scale, ym, ys = norm
        net = model()
        optimizer = torch.optim.AdamW(net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        index_tensor = torch.as_tensor(ids, device=device)
        target_tensor = (torch.as_tensor(y, dtype=torch.float32, device=device) - ym) / ys
        history, best, selected_epochs, stale = [], float("inf"), 0, 0
        for epoch in range(epochs):
            net.train()
            generator = torch.Generator(device=device).manual_seed(int(seed) + epoch)
            permutation = torch.randperm(len(ids), generator=generator, device=device)
            total_loss = torch.zeros((), device=device)
            total_gradient = torch.zeros((), device=device)
            for first in range(0, len(ids), config.batch_size):
                order = permutation[first:first + config.batch_size]
                samples = tensors[index_tensor[order]]
                max_start = tensors.shape[2] - config.crop_samples
                offsets = torch.randint(max_start + 1, (len(order),), generator=generator, device=device)
                sample_index = offsets[:, None] + torch.arange(config.crop_samples, device=device)[None, :]
                cropped = torch.gather(samples, 2, sample_index[:, None, :].expand(-1, tensors.shape[1], -1))
                batch = (cropped - mean) / scale
                optimizer.zero_grad(set_to_none=True)
                estimate = net(batch)
                if estimate.shape != target_tensor[order].shape:
                    raise RuntimeError("EEGNet output does not match genuine target dimensions")
                loss = torch.nn.functional.mse_loss(estimate, target_tensor[order])
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
                total_loss += loss.detach() * len(order)
                total_gradient += gradient.detach()
            objective = float(total_loss / len(ids))
            if not np.isfinite(objective):
                raise FloatingPointError("Non-finite training objective")
            record = dict(epoch=epoch + 1, objective=objective, gradient_norm_sum=float(total_gradient))
            if use_validation:
                val_prediction = predict(net, validation_indices, norm)
                mae = float(np.abs(val_prediction - validation_y).mean())
                record.update(validation_mae=mae, validation_prediction_std=val_prediction.std(0).tolist())
                if mae < best - 1e-6:
                    best, selected_epochs, stale = mae, epoch + 1, 0
                else:
                    stale += 1
            history.append(record)
            if use_validation and stale >= config.patience:
                break
        return net, norm, history, selected_epochs, best

    started = time.monotonic()
    selected, _, history, epochs, best = train(train_indices, train_y, config.max_epochs, True)
    del selected
    if epochs < 1 or not np.isfinite(best):
        raise RuntimeError("No finite validation checkpoint")
    fitted, norm, refit_history, _, _ = train(fit_indices, fit_y, epochs, False)
    predictions = predict(fitted, np.arange(len(tensors)), norm)
    fit_metrics = regression_metrics(fit_y, predictions[fit_indices], target_names=tuple(f"target{i}" for i in range(targets)))
    torch.save(dict(state_dict=fitted.state_dict(), normalization=[v.detach().cpu() for v in norm],
                    config=asdict(config), target_count=targets, seed=int(seed), selected_epochs=epochs), checkpoint)
    metadata = dict(selected_epochs=epochs, validation_mae=best, history=history, refit_history=refit_history,
                    fit_metrics=fit_metrics, fit_prediction_std=predictions[fit_indices].std(0).tolist(),
                    elapsed_seconds=time.monotonic() - started, target_count=targets, config=asdict(config),
                    parameter_count=sum(p.numel() for p in fitted.parameters()))
    del fitted
    torch.cuda.empty_cache()
    return predictions, metadata
