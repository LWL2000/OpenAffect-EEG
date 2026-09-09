"""Nonlinear EEG residual encoder with a gradient-reversal stimulus head."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Function
from torch.nn import functional as F


class _GradientReverse(Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = float(strength)
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.strength * gradient, None


def gradient_reverse(values: torch.Tensor, *, strength: float) -> torch.Tensor:
    """Keep the forward value and reverse its encoder gradient."""
    if not 0.0 <= strength:
        raise ValueError("Adversary strength must be non-negative")
    return _GradientReverse.apply(values, float(strength))


class AdversarialResidualNetwork(nn.Module):
    """Encode EEG features for affect regression and stimulus adversarial loss."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        stimulus_count: int,
        dropout: float,
    ):
        super().__init__()
        if min(input_dim, hidden_dim, latent_dim, stimulus_count) < 1:
            raise ValueError("Network dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Dropout must be in [0, 1)")
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )
        self.affect_head = nn.Linear(latent_dim, 2)
        self.stimulus_head = nn.Linear(latent_dim, stimulus_count)

    def forward(
        self,
        features: torch.Tensor,
        *,
        adversary_strength: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encoder(features)
        residual = self.affect_head(latent)
        reversed_latent = gradient_reverse(latent, strength=adversary_strength)
        stimulus_logits = self.stimulus_head(reversed_latent)
        return residual, stimulus_logits, latent


@dataclass(frozen=True)
class AdversarialTrainingConfig:
    """Hyperparameters shared by zero-adversary and selected-adversary models."""

    hidden_dim: int = 64
    latent_dim: int = 16
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    stimulus_loss_weight: float = 1.0
    max_epochs: int = 200
    patience: int = 25
    minimum_delta: float = 1e-6
    gradient_clip_norm: float = 5.0
    thread_count: int = 4

    def __post_init__(self) -> None:
        if min(self.hidden_dim, self.latent_dim, self.max_epochs, self.patience) < 1:
            raise ValueError("Network sizes, epochs, and patience must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("Dropout must be in [0, 1)")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("Optimizer parameters are invalid")
        if self.stimulus_loss_weight < 0.0:
            raise ValueError("Stimulus loss weight must be non-negative")
        if self.minimum_delta < 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("Stopping and clipping parameters are invalid")
        if self.thread_count < 1:
            raise ValueError("Thread count must be positive")


class FittedAdversarialResidualModel:
    """Fitted network plus train-only normalization statistics."""

    def __init__(
        self,
        *,
        model: AdversarialResidualNetwork,
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
        target_mean: np.ndarray,
        target_scale: np.ndarray,
        best_epoch: int,
        validation_mae: float,
        stimulus_classes: np.ndarray,
    ):
        self.model = model.eval()
        self.feature_mean = feature_mean
        self.feature_scale = feature_scale
        self.target_mean = target_mean
        self.target_scale = target_scale
        self.best_epoch = int(best_epoch)
        self.validation_mae = float(validation_mae)
        self.stimulus_classes = stimulus_classes

    def _features_tensor(self, features: np.ndarray) -> torch.Tensor:
        values = _feature_matrix(
            features,
            expected_width=len(self.feature_mean),
            minimum_rows=1,
        )
        standardized = (values - self.feature_mean) / self.feature_scale
        return torch.as_tensor(standardized, dtype=torch.float32)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict the experienced-affect residual in the original target scale."""
        with torch.no_grad():
            residual, _, _ = self.model(
                self._features_tensor(features), adversary_strength=0.0
            )
        return residual.numpy().astype(float) * self.target_scale + self.target_mean

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Return the learned EEG latent representation for identity probes."""
        with torch.no_grad():
            _, _, latent = self.model(
                self._features_tensor(features), adversary_strength=0.0
            )
        return latent.numpy().astype(float)


def _feature_matrix(
    features: np.ndarray,
    *,
    expected_width: int | None = None,
    minimum_rows: int = 2,
) -> np.ndarray:
    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or len(values) < minimum_rows:
        raise ValueError(f"Features must contain at least {minimum_rows} row(s)")
    if expected_width is not None and values.shape[1] != expected_width:
        raise ValueError("Requested features have an incompatible width")
    if not np.isfinite(values).all():
        raise ValueError("Features contain non-finite values")
    return values


def _target_matrix(targets: np.ndarray, *, expected_rows: int) -> np.ndarray:
    values = np.asarray(targets, dtype=float)
    if values.shape != (expected_rows, 2):
        raise ValueError("Residual targets must have shape (n_samples, 2)")
    if not np.isfinite(values).all():
        raise ValueError("Residual targets contain non-finite values")
    return values


def _normalization(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def fit_adversarial_residual_model(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    train_stimulus: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    *,
    adversary_strength: float,
    seed: int,
    config: AdversarialTrainingConfig | None = None,
) -> FittedAdversarialResidualModel:
    """Fit with validation MAE early stopping and return the best epoch."""
    if not np.isfinite(adversary_strength) or adversary_strength < 0.0:
        raise ValueError("Adversary strength must be finite and non-negative")
    settings = config or AdversarialTrainingConfig()
    train_x = _feature_matrix(train_features)
    train_y = _target_matrix(train_targets, expected_rows=len(train_x))
    validation_x = _feature_matrix(
        validation_features, expected_width=train_x.shape[1]
    )
    validation_y = _target_matrix(validation_targets, expected_rows=len(validation_x))
    groups = np.asarray(train_stimulus).astype(str)
    if groups.shape != (len(train_x),):
        raise ValueError("Stimulus identities have an incompatible shape")

    feature_mean, feature_scale = _normalization(train_x)
    target_mean, target_scale = _normalization(train_y)
    standardized_train_x = (train_x - feature_mean) / feature_scale
    standardized_train_y = (train_y - target_mean) / target_scale
    standardized_validation_x = (validation_x - feature_mean) / feature_scale
    stimulus_classes, stimulus_labels = np.unique(groups, return_inverse=True)

    torch.set_num_threads(settings.thread_count)
    torch.manual_seed(int(seed))
    model = AdversarialResidualNetwork(
        input_dim=train_x.shape[1],
        hidden_dim=settings.hidden_dim,
        latent_dim=settings.latent_dim,
        stimulus_count=len(stimulus_classes),
        dropout=settings.dropout,
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    train_x_tensor = torch.as_tensor(standardized_train_x, dtype=torch.float32)
    train_y_tensor = torch.as_tensor(standardized_train_y, dtype=torch.float32)
    stimulus_tensor = torch.as_tensor(stimulus_labels, dtype=torch.long)
    validation_x_tensor = torch.as_tensor(
        standardized_validation_x, dtype=torch.float32
    )

    best_mae = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    for epoch in range(settings.max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        residual, stimulus_logits, _ = model(
            train_x_tensor, adversary_strength=adversary_strength
        )
        affect_loss = F.mse_loss(residual, train_y_tensor)
        stimulus_loss = F.cross_entropy(stimulus_logits, stimulus_tensor)
        loss = affect_loss + settings.stimulus_loss_weight * stimulus_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip_norm)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            standardized_prediction, _, _ = model(
                validation_x_tensor, adversary_strength=0.0
            )
        prediction = (
            standardized_prediction.numpy().astype(float) * target_scale + target_mean
        )
        validation_mae = float(np.mean(np.abs(prediction - validation_y)))
        if validation_mae < best_mae - settings.minimum_delta:
            best_mae = validation_mae
            best_epoch = epoch
            best_state = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= settings.patience:
                break

    if best_state is None:
        raise RuntimeError("Adversarial training did not produce a finite checkpoint")
    model.load_state_dict(best_state)
    return FittedAdversarialResidualModel(
        model=model,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        best_epoch=best_epoch,
        validation_mae=best_mae,
        stimulus_classes=stimulus_classes,
    )


def fit_adversarial_residual_model_fixed_epochs(
    features: np.ndarray,
    targets: np.ndarray,
    stimulus: np.ndarray,
    *,
    adversary_strength: float,
    epoch_count: int,
    seed: int,
    config: AdversarialTrainingConfig | None = None,
) -> FittedAdversarialResidualModel:
    """Refit on all development rows for an already selected epoch count."""
    if not np.isfinite(adversary_strength) or adversary_strength < 0.0:
        raise ValueError("Adversary strength must be finite and non-negative")
    if not isinstance(epoch_count, int) or epoch_count < 1:
        raise ValueError("Epoch count must be a positive integer")
    settings = config or AdversarialTrainingConfig()
    train_x = _feature_matrix(features)
    train_y = _target_matrix(targets, expected_rows=len(train_x))
    groups = np.asarray(stimulus).astype(str)
    if groups.shape != (len(train_x),):
        raise ValueError("Stimulus identities have an incompatible shape")

    feature_mean, feature_scale = _normalization(train_x)
    target_mean, target_scale = _normalization(train_y)
    standardized_train_x = (train_x - feature_mean) / feature_scale
    standardized_train_y = (train_y - target_mean) / target_scale
    stimulus_classes, stimulus_labels = np.unique(groups, return_inverse=True)

    torch.set_num_threads(settings.thread_count)
    torch.manual_seed(int(seed))
    model = AdversarialResidualNetwork(
        input_dim=train_x.shape[1],
        hidden_dim=settings.hidden_dim,
        latent_dim=settings.latent_dim,
        stimulus_count=len(stimulus_classes),
        dropout=settings.dropout,
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    train_x_tensor = torch.as_tensor(standardized_train_x, dtype=torch.float32)
    train_y_tensor = torch.as_tensor(standardized_train_y, dtype=torch.float32)
    stimulus_tensor = torch.as_tensor(stimulus_labels, dtype=torch.long)

    for _ in range(epoch_count):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        residual, stimulus_logits, _ = model(
            train_x_tensor, adversary_strength=adversary_strength
        )
        affect_loss = F.mse_loss(residual, train_y_tensor)
        stimulus_loss = F.cross_entropy(stimulus_logits, stimulus_tensor)
        loss = affect_loss + settings.stimulus_loss_weight * stimulus_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip_norm)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        standardized_prediction, _, _ = model(
            train_x_tensor, adversary_strength=0.0
        )
    prediction = (
        standardized_prediction.numpy().astype(float) * target_scale + target_mean
    )
    training_mae = float(np.mean(np.abs(prediction - train_y)))
    return FittedAdversarialResidualModel(
        model=model,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        best_epoch=epoch_count - 1,
        validation_mae=training_mae,
        stimulus_classes=stimulus_classes,
    )
