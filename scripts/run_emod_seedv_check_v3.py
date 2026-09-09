"""Bounded official-task EMOD check with strict backbone loading and epoch resume."""
from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
from torch.utils.data import Dataset, DataLoader

from openaffect_eeg.artifacts import sha256_file


class Windows(Dataset):
    def __init__(self, root, manifest, split):
        self.arrays, tables = [], []
        for i, record in enumerate(manifest["records"]):
            array = np.load(root/record["tensor_file"], mmap_mode="r", allow_pickle=False)
            table = pd.read_csv(root/record["table_file"], sep="\t")
            table["array"] = i
            table["offset"] = np.arange(len(table))
            self.arrays.append(array)
            tables.append(table.loc[table["split"].eq(split)])
        self.table = pd.concat(tables, ignore_index=True)
        self.index = self.table[["array", "offset", "label"]].to_numpy(int)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        array, offset, label = self.index[index]
        return torch.from_numpy(np.array(self.arrays[array][offset], copy=True)), int(label)


def evaluate(model, loader, channels, loss_fn):
    model.eval()
    labels, probabilities, loss_sum = [], [], 0.
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            logits = model(x, channels[None].expand(len(x), -1))
            if not torch.isfinite(logits).all():
                raise ValueError("Non-finite official-task logits")
            loss_sum += loss_fn(logits, y).item()*len(x)
            labels.extend(y.cpu().tolist())
            probabilities.append(logits.softmax(-1).cpu().numpy())
    p = np.concatenate(probabilities)
    predicted = p.argmax(-1)
    return {"loss": loss_sum/len(labels), "accuracy": accuracy_score(labels, predicted),
            "balanced_accuracy": balanced_accuracy_score(labels, predicted),
            "kappa": cohen_kappa_score(labels, predicted),
            "macro_f1": f1_score(labels, predicted, average="macro")}, p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(2)
    random.seed(2025)
    np.random.seed(2025)
    torch.manual_seed(2025)
    # Official adaptive pooling has no deterministic CUDA backward implementation.
    torch.use_deterministic_algorithms(True, warn_only=True)
    sys.path.insert(0, str(args.source))
    cls = importlib.import_module("model.EMOD").EMOD
    checkpoint_path = args.source/"Pretraining_weights/EMOD_pretrained.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint = {k.removeprefix("module."): v for k, v in checkpoint.items()}
    manifest_path = args.prepared/"prepared_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    model = cls(n_filters=128, depth=3, dropout=.25, frequency=200, num_classes=5, eeg_len=200, channels=62, dataset="SEEDV")
    state = model.state_dict()
    backbone_names = [k for k in state if k.startswith(("embd.", "axial_transformer.", "pos_embeddings."))]
    if any(k not in checkpoint or checkpoint[k].shape != state[k].shape for k in backbone_names):
        raise ValueError("Official checkpoint lacks a complete matching backbone")
    for k in backbone_names:
        state[k] = checkpoint[k]
    model.load_state_dict(state, strict=True)
    channels = torch.tensor(manifest["channel_ids"], device="cuda")
    datasets = {split: Windows(args.prepared, manifest, split) for split in ("train", "validation", "test")}
    if any(set(datasets[a].table.trial_uid) & set(datasets[b].table.trial_uid)
           for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Window split leaks complete trials")
    loaders = {name: DataLoader(data, batch_size=args.batch_size, shuffle=name == "train", num_workers=0,
                pin_memory=True) for name, data in datasets.items()}
    model.cuda()
    optimizer = torch.optim.Adam([
        {"params": model.embd.parameters(), "lr": .0005},
        {"params": model.axial_transformer.parameters(), "lr": .0005},
        {"params": model.pos_embeddings.parameters(), "lr": .0005},
        {"params": model.classifier.parameters(), "lr": .001}], lr=.001, weight_decay=.0001)
    opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    if {id(p) for p in model.parameters() if p.requires_grad} != opt_ids:
        raise ValueError("Trainable parameters missing from the official optimizer groups")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=1e-8)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=.1)
    args.output.mkdir(parents=True, exist_ok=True)
    fingerprint = {"prepared_manifest_sha256": sha256_file(manifest_path), "checkpoint_sha256": sha256_file(checkpoint_path),
        "script_sha256": sha256_file(Path(__file__)), "source_sha256": sha256_file(args.source/"model/EMOD.py"),
        "epochs": args.epochs, "batch_size": args.batch_size}
    resume = args.output/"resume.pt"
    history, best, start_epoch = [], -float("inf"), 0
    if resume.exists():
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        if saved["fingerprint"] != fingerprint:
            raise ValueError("Training resume fingerprint mismatch")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        history, best, start_epoch = saved["history"], saved["best"], saved["epoch"]+1
        torch.set_rng_state(saved["cpu_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    for epoch in range(start_epoch, args.epochs):
        begun = time.monotonic()
        model.train()
        total, correct, count, norms = 0., 0, 0, []
        for x, y in loaders["train"]:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, channels[None].expand(len(x), -1))
            loss = loss_fn(logits, y)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            norms.append(float(norm))
            optimizer.step()
            total += loss.item()*len(x)
            correct += int((logits.argmax(-1) == y).sum())
            count += len(x)
        scheduler.step()
        validation, _ = evaluate(model, loaders["validation"], channels, loss_fn)
        if validation["kappa"] >= best:
            best = validation["kappa"]
            torch.save(model.state_dict(), args.output/"best.pt")
        history.append({"epoch": epoch+1, "train_loss": total/count, "train_accuracy": correct/count,
            "validation": validation, "mean_gradient_norm": float(np.mean(norms)), "seconds": time.monotonic()-begun})
        temporary = args.output/"resume.tmp.pt"
        torch.save({"fingerprint": fingerprint, "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "best": best,
            "history": history, "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, temporary)
        temporary.replace(resume)
        (args.output/"history.json").write_text(json.dumps(history, indent=2)+"\n")
        print("EMOD SEEDV", epoch+1, f"train={correct/count:.4f} val_kappa={validation['kappa']:.4f}", flush=True)
    model.load_state_dict(torch.load(args.output/"best.pt", weights_only=True), strict=True)
    result, probabilities = evaluate(model, loaders["test"], channels, loss_fn)
    prediction = datasets["test"].table.drop(columns=["array", "offset"]).copy()
    for k in range(5):
        prediction[f"probability_{k}"] = probabilities[:, k]
    prediction.to_csv(args.output/"private_test_predictions.tsv.gz", sep="\t", index=False)
    result.update(fingerprint=fingerprint, loaded_backbone_tensors=len(backbone_names),
        parameter_count=sum(p.numel() for p in model.parameters()), best_validation_kappa=best,
        split_counts={k: len(v) for k, v in datasets.items()}, epochs_completed=len(history),
        determinism="fixed seeds; deterministic algorithms warn-only; adaptive_avg_pool2d CUDA backward is nondeterministic",
        boundary="One-seed native within-participant official-task check, not a full reported-score replication or independent-person affect inference")
    (args.output/"task_check.json").write_text(json.dumps(result, indent=2)+"\n")
    print("EMOD task check finished", result, flush=True)


if __name__ == "__main__":
    main()
