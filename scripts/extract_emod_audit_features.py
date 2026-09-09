"""Audit the official EMOD checkpoint as frozen representations, not full replication."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from openaffect_eeg.artifacts import sha256_file
from openaffect_eeg.labram import EMO_64_CHANNELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("tensors", type=Path)
    parser.add_argument("uids", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    sys.path.insert(0, str(source))
    model_type = importlib.import_module("model.EMOD").EMOD_pretrain
    mapping = importlib.import_module("data_preprocess.channel_idx").get_channel_idx()
    checkpoint = source / "Pretraining_weights/EMOD_pretrained.ckpt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    state = {k.removeprefix("module."): v for k, v in state.items()}
    renamed = {}
    if "attnpool_c.attn.weight" in state:
        prefixes = {"attnpool_c.": "projection_head.0.",
                    "attnpool_t.": "projection_head.1.",
                    "projection_head.0.": "projection_head.3.",
                    "projection_head.1.": "projection_head.4.",
                    "projection_head.3.": "projection_head.6."}
        mapped = {}
        for key, value in state.items():
            target = next((new + key[len(old):] for old, new in prefixes.items() if key.startswith(old)), key)
            if target in mapped:
                raise ValueError("Checkpoint key mapping collision")
            mapped[target] = value
            if target != key:
                renamed[key] = target
        state = mapped
    width = int(state["pos_embeddings.weight"].shape[1])
    print("checkpoint width", width, "keys", len(state), flush=True)
    # Depth is resolved by strict matching, not a partial checkpoint load.
    model = None
    failures = []
    for depth in (2, 3):
        candidate = model_type(n_filters=width, depth=depth)
        try:
            candidate.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            failures.append(str(exc))
            continue
        model = candidate
        break
    if model is None:
        raise ValueError("Official checkpoint did not strictly match supported architecture: " + failures[-1])
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    print("strict checkpoint match", width, depth, commit, flush=True)
    if args.inspect_only:
        return
    tensors = np.load(args.tensors, mmap_mode="r", allow_pickle=False)
    uids = np.load(args.uids, allow_pickle=False).astype(str)
    if tensors.ndim != 3 or tensors.shape[1:] != (64, 3000) or len(uids) != len(tensors):
        raise ValueError("This adapter requires the existing EmoEEG 64-channel 100-Hz 30-s archive")
    keep = [i for i, ch in enumerate(EMO_64_CHANNELS) if ch.upper() in mapping]
    removed = [ch for ch in EMO_64_CHANNELS if ch.upper() not in mapping]
    channel_ids = torch.tensor([mapping[EMO_64_CHANNELS[i].upper()] for i in keep], device="cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    input_hashes = {p.name: sha256_file(p) for p in (args.tensors, args.uids, checkpoint)}
    source_hashes = {str(p.relative_to(source)): sha256_file(p) for p in
                     (source / "model/EMOD.py", source / "model/AxialTransformer.py", source / "data_preprocess/channel_idx.py")}
    for initialization in ("pretrained", "random"):
        dest = args.output / f"ds005540_emod_{initialization}_frozen.npz"
        meta_path = dest.with_suffix(".json")
        if dest.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta["input_sha256"] != input_hashes or meta["source_sha256"] != source_hashes or meta["output_sha256"] != sha256_file(dest):
                raise ValueError("Cached EMOD artifact hash mismatch")
            continue
        torch.manual_seed(20260902)
        encoder = model if initialization == "pretrained" else model_type(n_filters=width, depth=depth)
        encoder = encoder.eval().cuda()
        feature = np.empty((len(uids), width), dtype=np.float32)
        started = time.monotonic()
        with torch.inference_mode():
            for start in range(0, len(uids), args.batch_size):
                end = min(len(uids), start + args.batch_size)
                trial = np.array(tensors[start:end, keep, :], dtype=np.float32)
                if not np.isfinite(trial).all():
                    raise ValueError("Nonfinite raw EEG")
                vectors = []
                for offset in (0, 1000, 2000):
                    x = torch.from_numpy(trial[:, :, offset:offset + 1000].copy()).cuda()
                    vectors.append(encoder(x, channel_ids[None, :].expand(end - start, -1)).cpu().numpy())
                feature[start:end] = np.mean(vectors, axis=0)
                if start % (args.batch_size * 25) == 0:
                    print(initialization, end, "/", len(uids), flush=True)
        if not np.isfinite(feature).all():
            raise ValueError("Nonfinite EMOD embeddings")
        np.savez_compressed(dest, trial_uid=uids, features=feature)
        meta = dict(model="EMOD", initialization=initialization, width=width, depth=depth,
            checkpoint_strict_match=True, checkpoint_key_renaming=renamed,
            official_commit=commit, source_sha256=source_hashes,
            input_sha256=input_hashes, output_sha256=sha256_file(dest), seed=20260902,
            input="Benchmark 100-Hz archive, no new amplitude normalization, three nonoverlapping 10-s crops",
            output="Mean of official L2-normalized projection embeddings over crops",
            retained_channels=[EMO_64_CHANNELS[i] for i in keep], removed_unknown_channels=removed,
            boundary="Frozen official weights with explicit legacy-head name mapping, benchmark preprocessing and Ridge readout; not replication of original fine-tuning results or a current SOTA ranking claim",
            elapsed_seconds=time.monotonic()-started, torch_version=torch.__version__, cuda_version=torch.version.cuda,
            cuda_library_root=os.environ.get("OPENAFFECT_CUDA_LIB_ROOT"))
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        print(initialization, "features verified", feature.shape, flush=True)
        encoder.cpu()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
