"""Fine-tune the final LaBraM block on prespecified matched-resource corners.

This is a bounded sensitivity analysis: five original identity blocks, stimulus
doses 0/8, participant doses 0/8, one residual target construction, and one
declared optimization setting.  It does not replace the frozen-LaBraM grid.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from extract_labram_features import load_labram, repository_commit, sha256_file
from openaffect_eeg.final_robustness import population_targets, prediction_table
from openaffect_eeg.labram import EMO_64_CHANNELS, LABRAM_STANDARD_1020


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".incomplete.json")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def channel_ids(names: list[str]) -> list[int]:
    positions = {name: i + 1 for i, name in enumerate(LABRAM_STANDARD_1020)}
    result = [positions[str(name).upper()] for name in names]
    if len(result) != len(set(result)):
        raise ValueError("Channel positions must be unique")
    return [0, *result]


def build_model(repository: Path, checkpoint: Path, device: str, targets: int, seed: int):
    backbone, torch, metadata = load_labram(repository, checkpoint, device, seed=seed)
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    if not hasattr(backbone, "blocks") or len(backbone.blocks) < 1:
        raise ValueError("Official LaBraM transformer blocks were not found")
    for parameter in backbone.blocks[-1].parameters():
        parameter.requires_grad_(True)
    if hasattr(backbone, "norm"):
        for parameter in backbone.norm.parameters():
            parameter.requires_grad_(True)
    torch.manual_seed(seed)
    head = torch.nn.Linear(int(backbone.embed_dim), targets).to(device)
    return backbone, head, torch, metadata


def fit_residual(
    tensors,
    train_indices,
    train_y,
    validation_indices,
    validation_y,
    fit_indices,
    fit_y,
    *,
    channels,
    repository,
    checkpoint,
    seed,
    crop_samples,
    scale_factor,
    output,
    max_epochs=30,
    patience=5,
    batch_size=8,
):
    """Select epochs on train/validation, then refit on their union."""
    import torch
    import torch.nn.functional as functional

    device = "cuda"
    input_chans = channel_ids(channels)
    targets = train_y.shape[1]
    pretrained, _, _, load_meta = build_model(repository, checkpoint, device, targets, seed)
    pretrained_state = {key: value.detach().cpu() for key, value in pretrained.state_dict().items()}
    del pretrained
    torch.cuda.empty_cache()

    def new_model(run_seed):
        model, head, _, _ = build_model(repository, checkpoint, device, targets, run_seed)
        model.load_state_dict(pretrained_state)
        return model, head

    def crop(indices, generator, random):
        batch = tensors[indices]
        limit = batch.shape[-1] - crop_samples
        if limit < 0:
            raise ValueError("Crop exceeds the available trial")
        if random:
            starts = torch.randint(limit + 1, (len(indices),), generator=generator, device=device)
        else:
            starts = torch.full((len(indices),), limit // 2, device=device)
        offsets = starts[:, None] + torch.arange(crop_samples, device=device)[None, :]
        batch = torch.gather(batch, 2, offsets[:, None, :].expand(-1, batch.shape[1], -1))
        batch = batch * scale_factor
        batch = batch - batch.mean(dim=2, keepdim=True)
        batch = functional.interpolate(batch, size=crop_samples * 2, mode="linear", align_corners=False)
        return batch.reshape(len(indices), batch.shape[1], crop_samples // 100, 200)

    def predict(model, head, indices):
        model.eval(); head.eval()
        values = []
        for first in range(0, len(indices), batch_size):
            ids = torch.as_tensor(indices[first:first + batch_size], device=device)
            batch = crop(ids, None, False)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                tokens = model(batch, input_chans=input_chans, return_patch_tokens=True)
                values.append(head(tokens.mean(1)).float().cpu().numpy())
        return np.concatenate(values)

    def train(indices, targets_np, epochs, use_validation, run_seed):
        model, head = new_model(run_seed)
        backbone_parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW([
            {"params": backbone_parameters, "lr": 1e-5},
            {"params": head.parameters(), "lr": 1e-3},
        ], weight_decay=0.01)
        ids = torch.as_tensor(indices, device=device)
        targets_tensor = torch.as_tensor(targets_np, dtype=torch.float32, device=device)
        history, best, selected, stale = [], float("inf"), 0, 0
        for epoch in range(epochs):
            model.train(); head.train()
            generator = torch.Generator(device=device).manual_seed(run_seed + epoch)
            order = torch.randperm(len(ids), generator=generator, device=device)
            total = 0.0
            for first in range(0, len(ids), batch_size):
                pick = order[first:first + batch_size]
                batch = crop(ids[pick], generator, True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    tokens = model(batch, input_chans=input_chans, return_patch_tokens=True)
                    estimate = head(tokens.mean(1))
                    loss = functional.mse_loss(estimate.float(), targets_tensor[pick])
                loss.backward()
                torch.nn.utils.clip_grad_norm_([*backbone_parameters, *head.parameters()], 1.0)
                optimizer.step()
                total += float(loss.detach()) * len(pick)
            record = {"epoch": epoch + 1, "train_mse": total / len(ids)}
            if use_validation:
                val = predict(model, head, validation_indices)
                mae = float(np.abs(val - validation_y).mean())
                record["validation_mae"] = mae
                if mae < best - 1e-6:
                    best, selected, stale = mae, epoch + 1, 0
                else:
                    stale += 1
            history.append(record)
            if use_validation and stale >= patience:
                break
        return model, head, history, selected, best

    started = time.monotonic()
    selected_model, selected_head, selection, epochs, validation_mae = train(
        np.asarray(train_indices), train_y, max_epochs, True, seed
    )
    del selected_model, selected_head
    torch.cuda.empty_cache()
    if epochs < 1:
        raise RuntimeError("Fine-tuning did not select a finite epoch")
    fitted, fitted_head, refit, _, _ = train(
        np.asarray(fit_indices), fit_y, epochs, False, seed + 500000
    )
    prediction = predict(fitted, fitted_head, np.arange(len(tensors)))
    state = {"backbone": fitted.state_dict(), "head": fitted_head.state_dict()}
    torch.save(state, output)
    metadata = {
        "selected_epochs": epochs,
        "validation_mae": validation_mae,
        "selection_history": selection,
        "refit_history": refit,
        "elapsed_seconds": time.monotonic() - started,
        "checkpoint_load": load_meta,
        "trainable_parameters": sum(p.numel() for p in fitted.parameters() if p.requires_grad)
            + sum(p.numel() for p in fitted_head.parameters()),
        "optimization": {"backbone_lr": 1e-5, "head_lr": 1e-3, "weight_decay": 0.01,
                         "max_epochs": max_epochs, "patience": patience, "batch_size": batch_size},
        "adaptation": "100-Hz analysis tensors linearly resampled to LaBraM's 200-Hz patch interface",
        "input_scale_factor": scale_factor,
    }
    del fitted, fitted_head
    torch.cuda.empty_cache()
    return prediction, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("assignments_root", type=Path)
    parser.add_argument("model_repository", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    spec = cfg["datasets"][args.dataset]
    targets = tuple(spec["targets"])
    target_names = tuple(value.removeprefix("target_") for value in targets)
    base = args.assignments_root / args.dataset
    table = pd.read_csv(base / "eligible_trials.tsv.gz", sep="\t").sort_values("trial_uid").reset_index(drop=True)
    table["tensor_index"] = np.arange(len(table))
    raw_uids = np.load(args.data_root / spec["uids"], allow_pickle=False).astype(str)
    raw_index = pd.Index(raw_uids).get_indexer(table.trial_uid)
    if (raw_index < 0).any():
        raise ValueError("Fine-tuning trial identity mismatch")
    raw = np.load(args.data_root / spec["tensors"], mmap_mode="r", allow_pickle=False)
    import torch
    tensors = torch.as_tensor(np.asarray(raw[raw_index], dtype=np.float32), device="cuda")
    if args.dataset == "ds005540":
        channels = list(EMO_64_CHANNELS)
        scale_factor = 1.0
    else:
        meta = json.loads((args.data_root / "derived/final_robustness_v9_inputs/urban_100hz.json").read_text())
        channels = meta["channel_order"]
        if meta.get("units") != "physical volts":
            raise ValueError("Unexpected Urban tensor units")
        scale_factor = 1_000_000.0
    if len(channels) != tensors.shape[1]:
        raise ValueError("Channel metadata does not match tensors")

    args.output.mkdir(parents=True, exist_ok=True)
    source_paths = [args.config, Path(__file__), Path(inspect.getfile(population_targets)),
                    Path(inspect.getfile(prediction_table)), args.checkpoint]
    manifest = {
        "dataset_id": args.dataset,
        "model_repository_commit": repository_commit(args.model_repository),
        "sources": {path.name: sha256_file(path) for path in source_paths},
        "scope": "Residual LaBraM last-block fine-tuning at the four p/s resource corners.",
    }
    write_json(args.output / "manifest.json", manifest)
    rows, training = [], []
    for fold in range(cfg["fold_count"]):
        assignment_seed = cfg["design_seed"] + fold
        folder = base / "assignments" / f"seed-{assignment_seed}"
        for stimulus_dose in (0, 8):
            assignment0 = pd.read_csv(folder / f"participant-00_stimulus-{stimulus_dose:02d}.tsv.gz", sep="\t")
            resources = population_targets(table, assignment0, targets)
            cache = args.output / f"fold-{fold}_stimulus-{stimulus_dose}.pt"
            prediction, meta = fit_residual(
                tensors,
                resources["train"].tensor_index.to_numpy(), resources["train_y"] - resources["train_loo"],
                resources["validation"].tensor_index.to_numpy(), resources["val_y"] - resources["val_prior"],
                resources["fit"].tensor_index.to_numpy(), resources["fit_y"] - resources["fit_loo"],
                channels=channels, repository=args.model_repository, checkpoint=args.checkpoint,
                seed=assignment_seed + 900000, crop_samples=spec["crop_samples"],
                scale_factor=scale_factor, output=cache,
            )
            training.append({"fold": fold, "stimulus_dose": stimulus_dose, **meta})
            for participant_dose in (0, 8):
                assignment = pd.read_csv(folder / f"participant-{participant_dose:02d}_stimulus-{stimulus_dose:02d}.tsv.gz", sep="\t")
                zero = np.zeros_like(prediction)
                frame = prediction_table(table, assignment, targets, resources["all_prior"], zero, prediction)
                frame.insert(0, "representation", "labram_lastblock_residual")
                frame.insert(1, "seed", assignment_seed)
                frame.insert(2, "participant_dose", participant_dose)
                frame.insert(3, "stimulus_dose", stimulus_dose)
                rows.append(frame)
            write_json(args.output / "training.json", training)
            print(args.dataset, fold, stimulus_dose, meta["selected_epochs"], flush=True)
    result = pd.concat(rows, ignore_index=True)
    destination = args.output / "identity_exposure_predictions.tsv.gz"
    result.to_csv(destination, sep="\t", index=False)
    write_json(args.output / "completion.json", {
        "status": "complete", "rows": len(result), "predictions_sha256": sha256_file(destination),
        "boundary": "Key-corner sensitivity only; no full 5x5 fine-tuned surface or retraining interval.",
    })


if __name__ == "__main__":
    main()
