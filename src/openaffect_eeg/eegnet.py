"""Compact EEGNet-style residual encoder with stimulus adversarial training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from openaffect_eeg.adversarial import gradient_reverse
from openaffect_eeg.baselines import regression_metrics


class StandardEEGNetRegression(nn.Module):
    """Version-pinned Braindecode EEGNet with raw two-output regression."""

    def __init__(self, channel_count: int, config: "EEGNetTrainingConfig"):
        super().__init__()
        from importlib.metadata import version
        from braindecode.models import EEGNet

        if version("braindecode") != "1.2.0":
            raise ValueError("This audited adapter requires braindecode==1.2.0")
        self.backbone = EEGNet(
            n_chans=channel_count, n_outputs=2, n_times=config.crop_samples,
            F1=config.f1, D=config.depth_multiplier, F2=config.f2,
            kernel_length=config.temporal_kernel,
            depthwise_kernel_length=config.separable_kernel,
            drop_prob=config.dropout, final_layer_with_constraint=False,
        )

    def forward(self, trials, *, adversary_strength=0.0):
        if adversary_strength != 0.0:
            raise ValueError("Standard regression EEGNet has no adversarial head")
        encoded = trials
        latent = None
        for name, module in self.backbone.named_children():
            if name == "final_layer":
                latent = encoded.flatten(1)
            encoded = module(encoded)
        if latent is None or encoded.shape != (len(trials), 2):
            raise RuntimeError("Unexpected standard EEGNet module/output contract")
        return encoded, encoded.new_zeros((len(trials), 1)), latent


class EEGNetAdversarialResidualNetwork(nn.Module):
    """EEGNet-style temporal/spatial encoder with affect and stimulus heads."""

    def __init__(
        self,
        *,
        channel_count: int,
        temporal_kernel: int,
        separable_kernel: int,
        f1: int,
        depth_multiplier: int,
        f2: int,
        latent_dim: int,
        stimulus_count: int,
        dropout: float,
    ):
        super().__init__()
        dimensions = (
            channel_count,
            f1,
            depth_multiplier,
            f2,
            latent_dim,
            stimulus_count,
        )
        if min(dimensions) < 1:
            raise ValueError("EEGNet dimensions must be positive")
        if min(temporal_kernel, separable_kernel) < 1:
            raise ValueError("EEGNet kernels must be positive")
        if temporal_kernel % 2 == 0 or separable_kernel % 2 == 0:
            raise ValueError("EEGNet kernels must be odd for length-preserving padding")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Dropout must be in [0, 1)")

        spatial_width = f1 * depth_multiplier
        self.temporal = nn.Sequential(
            nn.Conv2d(
                1,
                f1,
                kernel_size=(1, temporal_kernel),
                padding=(0, temporal_kernel // 2),
                bias=False,
            ),
            nn.BatchNorm2d(f1),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(
                f1,
                spatial_width,
                kernel_size=(channel_count, 1),
                groups=f1,
                bias=False,
            ),
            nn.BatchNorm2d(spatial_width),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(
                spatial_width,
                spatial_width,
                kernel_size=(1, separable_kernel),
                padding=(0, separable_kernel // 2),
                groups=spatial_width,
                bias=False,
            ),
            nn.Conv2d(spatial_width, f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.latent_projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(f2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )
        self.affect_head = nn.Linear(latent_dim, 2)
        self.stimulus_head = nn.Linear(latent_dim, stimulus_count)

    def forward(
        self,
        trials: torch.Tensor,
        *,
        adversary_strength: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if trials.ndim != 3:
            raise ValueError("EEGNet inputs must have shape batch x channels x samples")
        encoded = self.temporal(trials.unsqueeze(1))
        encoded = self.spatial(encoded)
        encoded = self.separable(encoded)
        latent = self.latent_projector(encoded)
        residual = self.affect_head(latent)
        reversed_latent = gradient_reverse(latent, strength=adversary_strength)
        stimulus_logits = self.stimulus_head(reversed_latent)
        return residual, stimulus_logits, latent


@dataclass(frozen=True)
class EEGNetTrainingConfig:
    """Training and cropped-inference settings for the EEGNet residual model."""

    temporal_kernel: int = 51
    separable_kernel: int = 15
    f1: int = 4
    depth_multiplier: int = 2
    f2: int = 8
    latent_dim: int = 16
    dropout: float = 0.25
    crop_samples: int = 1000
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    stimulus_loss_weight: float = 1.0
    max_epochs: int = 100
    patience: int = 12
    minimum_delta: float = 1e-6
    gradient_clip_norm: float = 5.0
    thread_count: int = 4
    device: str = "cpu"
    architecture: str = "compact"
    record_history: bool = False
    eegain_source: str | None = None
    sampling_rate: int = 100

    def __post_init__(self) -> None:
        positive = (
            self.temporal_kernel,
            self.separable_kernel,
            self.f1,
            self.depth_multiplier,
            self.f2,
            self.latent_dim,
            self.crop_samples,
            self.batch_size,
            self.max_epochs,
            self.patience,
            self.thread_count,
        )
        if min(positive) < 1:
            raise ValueError("EEGNet training sizes must be positive")
        if self.architecture not in {"compact", "braindecode_eegnet", "eegain_eegnet"}:
            raise ValueError("Unknown EEGNet architecture")
        if self.architecture == "compact" and (
            self.temporal_kernel % 2 == 0 or self.separable_kernel % 2 == 0
        ):
            raise ValueError("EEGNet kernels must be odd")
        if self.architecture in {"braindecode_eegnet", "eegain_eegnet"} and self.stimulus_loss_weight:
            raise ValueError("Standard EEGNet regression must disable adversarial loss")
        if self.architecture == "eegain_eegnet" and (not self.eegain_source or self.sampling_rate < 32):
            raise ValueError("EEGain requires a pinned source path and valid sampling rate")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("Dropout must be in [0, 1)")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("Optimizer parameters are invalid")
        if self.stimulus_loss_weight < 0.0:
            raise ValueError("Stimulus loss weight must be non-negative")
        if self.minimum_delta < 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("Stopping and clipping parameters are invalid")
        if not self.device:
            raise ValueError("Training device must not be empty")


@dataclass(frozen=True)
class EEGNetNormalization:
    """Statistics estimated exclusively from one assigned training partition."""

    channel_mean: np.ndarray
    channel_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray


def _trial_indices(indices: np.ndarray, *, trial_count: int) -> np.ndarray:
    selected = np.asarray(indices, dtype=int)
    if selected.ndim != 1 or not len(selected):
        raise ValueError("Trial indices must be a non-empty one-dimensional array")
    if selected.min() < 0 or selected.max() >= trial_count:
        raise ValueError("Trial indices are outside the tensor archive")
    if len(np.unique(selected)) != len(selected):
        raise ValueError("Trial indices must be unique")
    return selected


def _targets(targets: np.ndarray, *, row_count: int) -> np.ndarray:
    values = np.asarray(targets, dtype=float)
    if values.shape != (row_count, 2) or not np.isfinite(values).all():
        raise ValueError("Residual targets must be finite with shape (n_trials, 2)")
    return values


def prepare_eegnet_normalization(
    tensors: np.ndarray,
    trial_indices: np.ndarray,
    targets: np.ndarray,
    *,
    chunk_size: int = 16,
) -> EEGNetNormalization:
    """Estimate channel and target scaling from selected training trials only."""
    if tensors.ndim != 3 or not np.issubdtype(tensors.dtype, np.floating):
        raise ValueError("EEG tensors must be a floating 3D array")
    selected = _trial_indices(trial_indices, trial_count=len(tensors))
    responses = _targets(targets, row_count=len(selected))
    if chunk_size < 1:
        raise ValueError("Normalization chunk size must be positive")

    channel_sum = np.zeros(tensors.shape[1], dtype=np.float64)
    channel_square_sum = np.zeros(tensors.shape[1], dtype=np.float64)
    value_count = 0
    for start in range(0, len(selected), chunk_size):
        block = np.asarray(tensors[selected[start : start + chunk_size]], dtype=float)
        if not np.isfinite(block).all():
            raise ValueError("EEG tensors contain non-finite values")
        channel_sum += block.sum(axis=(0, 2))
        channel_square_sum += np.square(block).sum(axis=(0, 2))
        value_count += block.shape[0] * block.shape[2]
    channel_mean = channel_sum / value_count
    channel_variance = channel_square_sum / value_count - np.square(channel_mean)
    channel_scale = np.sqrt(np.maximum(channel_variance, 0.0))
    channel_scale[channel_scale < 1e-8] = 1.0
    target_mean = responses.mean(axis=0)
    target_scale = responses.std(axis=0)
    target_scale[target_scale < 1e-8] = 1.0
    return EEGNetNormalization(
        channel_mean=channel_mean,
        channel_scale=channel_scale,
        target_mean=target_mean,
        target_scale=target_scale,
    )


def _crop_starts(sample_count: int, crop_samples: int) -> list[int]:
    if crop_samples > sample_count:
        raise ValueError("EEG crop is longer than the stored trial")
    starts = list(range(0, sample_count - crop_samples + 1, crop_samples))
    final_start = sample_count - crop_samples
    if final_start not in starts:
        starts.append(final_start)
    return starts


class _CroppedTrainingDataset(Dataset):
    def __init__(
        self,
        tensors: np.ndarray,
        indices: np.ndarray,
        targets: np.ndarray,
        stimulus_labels: np.ndarray,
        normalization: EEGNetNormalization,
        *,
        crop_samples: int,
        seed: int,
    ):
        self.tensors = tensors
        self.indices = indices
        self.targets = targets
        self.stimulus_labels = stimulus_labels
        self.normalization = normalization
        self.crop_samples = crop_samples
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.indices)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        archive_index = int(self.indices[item])
        trial = self.tensors[archive_index]
        available = trial.shape[1] - self.crop_samples
        start = 0
        if available:
            start = (
                self.seed * 1_000_003
                + self.epoch * 10_007
                + archive_index * 101
            ) % (available + 1)
        crop = np.array(
            trial[:, start : start + self.crop_samples],
            dtype=np.float32,
            copy=True,
        )
        crop -= self.normalization.channel_mean[:, None]
        crop /= self.normalization.channel_scale[:, None]
        return (
            torch.from_numpy(crop),
            torch.as_tensor(self.targets[item], dtype=torch.float32),
            int(self.stimulus_labels[item]),
        )


def _new_eegnet(
    *,
    channel_count: int,
    stimulus_count: int,
    config: EEGNetTrainingConfig,
) -> EEGNetAdversarialResidualNetwork:
    if config.architecture == "braindecode_eegnet":
        return StandardEEGNetRegression(channel_count, config)
    if config.architecture == "eegain_eegnet":
        from openaffect_eeg.eegain_adapter import EEGainRegression
        return EEGainRegression(channel_count, config)
    return EEGNetAdversarialResidualNetwork(
        channel_count=channel_count,
        temporal_kernel=config.temporal_kernel,
        separable_kernel=config.separable_kernel,
        f1=config.f1,
        depth_multiplier=config.depth_multiplier,
        f2=config.f2,
        latent_dim=config.latent_dim,
        stimulus_count=stimulus_count,
        dropout=config.dropout,
    )


def _predict_standardized(
    model: EEGNetAdversarialResidualNetwork,
    tensors: np.ndarray,
    indices: np.ndarray,
    normalization: EEGNetNormalization,
    config: EEGNetTrainingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    selected = _trial_indices(indices, trial_count=len(tensors))
    starts = _crop_starts(tensors.shape[2], config.crop_samples)
    residual_rows = []
    latent_rows = []
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(selected), config.batch_size):
            batch_indices = selected[
                batch_start : batch_start + config.batch_size
            ]
            trials = np.asarray(tensors[batch_indices], dtype=np.float32)
            crops = np.stack(
                [
                    trials[:, :, start : start + config.crop_samples]
                    for start in starts
                ],
                axis=1,
            )
            crop_shape = crops.shape
            crops = crops.reshape(-1, crop_shape[2], crop_shape[3])
            crops -= normalization.channel_mean[None, :, None]
            crops /= normalization.channel_scale[None, :, None]
            residual, _, latent = model(
                torch.from_numpy(crops).to(device), adversary_strength=0.0
            )
            residual_rows.append(
                residual.detach()
                .cpu()
                .numpy()
                .reshape(len(batch_indices), len(starts), 2)
                .mean(1)
            )
            latent_rows.append(
                latent.detach()
                .cpu()
                .numpy()
                .reshape(len(batch_indices), len(starts), -1)
                .mean(1)
            )
    return np.concatenate(residual_rows), np.concatenate(latent_rows)


class FittedAdversarialEEGNet:
    """Fitted EEGNet and train-only statistics for trial-level inference."""

    def __init__(
        self,
        *,
        model: EEGNetAdversarialResidualNetwork,
        normalization: EEGNetNormalization,
        config: EEGNetTrainingConfig,
        best_epoch: int,
        validation_mae: float,
        stimulus_classes: np.ndarray,
        history: list[dict] | None = None,
    ):
        self.model = model.eval()
        self.normalization = normalization
        self.config = config
        self.best_epoch = int(best_epoch)
        self.validation_mae = float(validation_mae)
        self.stimulus_classes = stimulus_classes
        self.history = history or []

    def predict(self, tensors: np.ndarray, indices: np.ndarray) -> np.ndarray:
        standardized, _ = _predict_standardized(
            self.model,
            tensors,
            indices,
            self.normalization,
            self.config,
        )
        return (
            standardized * self.normalization.target_scale
            + self.normalization.target_mean
        )

    def transform(self, tensors: np.ndarray, indices: np.ndarray) -> np.ndarray:
        _, latent = _predict_standardized(
            self.model,
            tensors,
            indices,
            self.normalization,
            self.config,
        )
        return latent


def _train_eegnet(
    tensors: np.ndarray,
    train_indices: np.ndarray,
    train_targets: np.ndarray,
    train_stimulus: np.ndarray,
    *,
    adversary_strength: float,
    seed: int,
    config: EEGNetTrainingConfig,
    normalization: EEGNetNormalization,
    epoch_count: int,
    validation_indices: np.ndarray | None = None,
    validation_targets: np.ndarray | None = None,
) -> FittedAdversarialEEGNet:
    if not np.isfinite(adversary_strength) or adversary_strength < 0.0:
        raise ValueError("Adversary strength must be finite and non-negative")
    selected = _trial_indices(train_indices, trial_count=len(tensors))
    responses = _targets(train_targets, row_count=len(selected))
    if config.crop_samples > tensors.shape[2]:
        raise ValueError("EEG crop is longer than the stored trial")
    groups = np.asarray(train_stimulus).astype(str)
    if groups.shape != (len(selected),):
        raise ValueError("Stimulus identities have an incompatible shape")
    stimulus_classes, stimulus_labels = np.unique(groups, return_inverse=True)
    standardized_targets = (
        responses - normalization.target_mean
    ) / normalization.target_scale
    dataset = _CroppedTrainingDataset(
        tensors,
        selected,
        standardized_targets,
        stimulus_labels,
        normalization,
        crop_samples=config.crop_samples,
        seed=seed,
    )

    torch.set_num_threads(config.thread_count)
    torch.manual_seed(int(seed))
    model = _new_eegnet(
        channel_count=tensors.shape[1],
        stimulus_count=len(stimulus_classes),
        config=config,
    ).to(torch.device(config.device))
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    validation_selected = None
    validation_y = None
    if validation_indices is not None or validation_targets is not None:
        if validation_indices is None or validation_targets is None:
            raise ValueError("Validation indices and targets must be provided together")
        validation_selected = _trial_indices(
            validation_indices, trial_count=len(tensors)
        )
        validation_y = _targets(
            validation_targets, row_count=len(validation_selected)
        )
        if np.intersect1d(selected, validation_selected).size:
            raise ValueError("Training and validation trials must be disjoint")

    best_mae = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history = []
    for epoch in range(epoch_count):
        dataset.set_epoch(epoch)
        generator = torch.Generator().manual_seed(int(seed) + epoch)
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        model.train()
        objective_sum, gradient_sum, samples_seen, batch_count = 0.0, 0.0, 0, 0
        for batch, target, stimulus_label in loader:
            batch = batch.to(config.device)
            target = target.to(config.device)
            stimulus_label = stimulus_label.to(config.device)
            optimizer.zero_grad(set_to_none=True)
            residual, stimulus_logits, _ = model(
                batch, adversary_strength=adversary_strength
            )
            affect_loss = F.mse_loss(residual, target)
            stimulus_loss = (
                F.cross_entropy(stimulus_logits, stimulus_label)
                if config.stimulus_loss_weight else affect_loss.new_zeros(())
            )
            loss = affect_loss + config.stimulus_loss_weight * stimulus_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("EEGNet objective is not finite")
            loss.backward()
            gradient = nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            if not torch.isfinite(gradient):
                raise FloatingPointError("EEGNet gradient is not finite")
            optimizer.step()
            objective_sum += float(loss.detach()) * len(batch)
            gradient_sum += float(gradient)
            samples_seen += len(batch)
            batch_count += 1

        record = dict(epoch=epoch + 1, objective=objective_sum / samples_seen,
                      gradient_norm=gradient_sum / batch_count)
        if config.record_history:
            train_standardized, _ = _predict_standardized(
                model, tensors, selected, normalization, config)
            train_prediction = (train_standardized * normalization.target_scale
                                + normalization.target_mean)
            record["train_metrics"] = regression_metrics(responses, train_prediction)
            record["train_prediction_std"] = train_prediction.std(0).tolist()

        if validation_selected is None:
            if config.record_history:
                history.append(record)
            continue
        standardized_prediction, _ = _predict_standardized(
            model,
            tensors,
            validation_selected,
            normalization,
            config,
        )
        prediction = (
            standardized_prediction * normalization.target_scale
            + normalization.target_mean
        )
        validation_mae = float(np.mean(np.abs(prediction - validation_y)))
        if config.record_history:
            record["validation_metrics"] = regression_metrics(validation_y, prediction)
            record["validation_prediction_std"] = prediction.std(0).tolist()
            history.append(record)
        if validation_mae < best_mae - config.minimum_delta:
            best_mae = validation_mae
            best_epoch = epoch
            best_state = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    if validation_selected is None:
        best_epoch = epoch_count - 1
        standardized_prediction, _ = _predict_standardized(
            model,
            tensors,
            selected,
            normalization,
            config,
        )
        prediction = (
            standardized_prediction * normalization.target_scale
            + normalization.target_mean
        )
        best_mae = float(np.mean(np.abs(prediction - responses)))
    else:
        if best_state is None:
            raise RuntimeError("EEGNet training did not produce a finite checkpoint")
        model.load_state_dict(best_state)
    return FittedAdversarialEEGNet(
        model=model,
        normalization=normalization,
        config=config,
        best_epoch=best_epoch,
        validation_mae=best_mae,
        stimulus_classes=stimulus_classes,
        history=history,
    )


def fit_adversarial_eegnet(
    tensors: np.ndarray,
    train_indices: np.ndarray,
    train_targets: np.ndarray,
    train_stimulus: np.ndarray,
    validation_indices: np.ndarray,
    validation_targets: np.ndarray,
    *,
    adversary_strength: float,
    seed: int,
    config: EEGNetTrainingConfig | None = None,
    normalization: EEGNetNormalization | None = None,
) -> FittedAdversarialEEGNet:
    """Fit EEGNet with validation residual MAE early stopping."""
    settings = config or EEGNetTrainingConfig()
    scaling = normalization or prepare_eegnet_normalization(
        tensors, train_indices, train_targets
    )
    return _train_eegnet(
        tensors,
        train_indices,
        train_targets,
        train_stimulus,
        validation_indices=validation_indices,
        validation_targets=validation_targets,
        adversary_strength=adversary_strength,
        seed=seed,
        config=settings,
        normalization=scaling,
        epoch_count=settings.max_epochs,
    )


def fit_adversarial_eegnet_fixed_epochs(
    tensors: np.ndarray,
    train_indices: np.ndarray,
    train_targets: np.ndarray,
    train_stimulus: np.ndarray,
    *,
    adversary_strength: float,
    epoch_count: int,
    seed: int,
    config: EEGNetTrainingConfig | None = None,
    normalization: EEGNetNormalization | None = None,
) -> FittedAdversarialEEGNet:
    """Refit EEGNet on all development trials for a selected epoch count."""
    if not isinstance(epoch_count, int) or epoch_count < 1:
        raise ValueError("Epoch count must be a positive integer")
    settings = config or EEGNetTrainingConfig()
    scaling = normalization or prepare_eegnet_normalization(
        tensors, train_indices, train_targets
    )
    return _train_eegnet(
        tensors,
        train_indices,
        train_targets,
        train_stimulus,
        adversary_strength=adversary_strength,
        seed=seed,
        config=settings,
        normalization=scaling,
        epoch_count=epoch_count,
    )
